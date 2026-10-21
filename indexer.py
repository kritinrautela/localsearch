"""Index the folders in config.toml into index.sqlite: text (FTS5 + EmbeddingGemma 2 vectors), OCR of scans and
screenshots, and photo, video-frame and soundtrack vectors.

Read-only on the source tree: files are opened for reading or handed to pdftotext/textutil/pdftoppm/ffmpeg, which
write to stdout or a temporary folder. Nothing under the roots is ever written, moved or deleted, and OCR text is
kept in the index only, never written back into a PDF.

    uv run python indexer.py                    # scan, extract, OCR, embed text, embed media
    uv run python indexer.py --no-embed         # scan and extract only (FTS works, vectors later)
    uv run python indexer.py --stages extract,ocr,embed     # any subset, in this order
    uv run python indexer.py --paths ~/Desktop/x --no-wait  # only these folders (the watcher's quick update)
"""

import argparse
import bisect
import csv
import fcntl
import html
import io
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

import common
import media
import ocr
from common import HERE, HOME, Embedder, connect, expand, load_config, version_key

TEXT_EXT = {"txt", "md", "markdown", "text", "rst"}
DOC_EXT = {"docx", "doc", "rtf", "odt", "html", "htm"}
SHEET_EXT = {"xlsx", "csv"}
ALL_EXT = TEXT_EXT | DOC_EXT | SHEET_EXT | {"pdf", "pptx"}
MEDIA_EXT = media.IMAGE_EXT | media.VIDEO_EXT | media.AUDIO_EXT
SF_DATALESS = 0x40000000   # an iCloud / Drive file whose bytes are still in the cloud: reading it would download it
STAGES = ["extract", "ocr", "embed", "media"]

LOG = None
SOFFICE_LOCK = threading.Lock()
PATHS = None     # --paths: only these folders are scanned...
ONLY = None      # ...and only the files found new or changed there go through OCR, embedding and media
LOCK = None      # the run lock; let go while waiting for power, so quick folder updates can run meanwhile
NO_WAIT = False  # --no-wait: on battery, skip embedding instead of waiting for power (a later run catches up)


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


# --- walking ---------------------------------------------------------------------------------------------------------

def skipped_name(name, cfg):
    s = cfg["skip"]
    low = name.lower()
    return (name.startswith(".") or low in s["names"] or any(low.startswith(p) for p in s["name_prefixes"])
            or any(low.endswith(x) for x in s["suffixes"]))


def roots(cfg):
    """[(folder, media_only)]: the document roots, then the extra photo/video roots. With --paths, the given folders
    that lie inside a root, each carrying its root's media_only."""
    out = [(expand(r), False) for r in cfg["roots"]]
    out += [(expand(r), True) for r in cfg.get("media", {}).get("roots", []) if expand(r).exists()]
    if PATHS is None:
        return out
    picked = []
    for p in PATHS:
        for r, media_only in out:
            if p == r or str(p).startswith(str(r) + "/"):
                picked.append((p, media_only))
                break
    return picked


def walk(cfg, counts):
    """Yield (absolute path, ext, stat) for indexable files under the roots, applying the skip rules."""
    under = [str(expand(u)) for u in cfg["skip"]["under"]]
    bundles = tuple(cfg["skip"]["bundle_suffixes"])
    max_bytes = cfg["extract"]["max_file_mb"] * 1_000_000
    media_bytes = cfg.get("media", {}).get("max_mb", 4000) * 1_000_000
    keep_dirs = set(cfg.get("media", {}).get("keep_dirs", []))
    skip_types = {t.lower().lstrip(".") for t in cfg["skip"].get("types", [])}
    for root, media_only in roots(cfg):
        if any(str(root) == u or str(root).startswith(u + "/") for u in under):
            continue
        exts = MEDIA_EXT if media_only else ALL_EXT | MEDIA_EXT
        for dirpath, dirs, files in os.walk(root):
            keep = []
            for d in dirs:
                full = os.path.join(dirpath, d)
                if d.startswith(".") or d.lower().endswith(bundles) or any(full == u or full.startswith(u + "/") for u in under):
                    counts["skipped_dirs"] += 1
                elif os.path.islink(full):
                    counts["skipped_links"] += 1
                else:
                    keep.append(d)
            dirs[:] = keep
            images = [f for f in files if f.rsplit(".", 1)[-1].lower() in media.IMAGE_EXT]
            sample = [os.path.join(dirpath, f) for f in images[:: max(1, len(images) // 20)][:20]]
            dataset = media.is_dataset(images, sample, os.path.basename(dirpath) in keep_dirs
                                       or ".photoslibrary" in dirpath)
            if dataset:
                counts["skipped_dataset_dirs"] += 1
                log(f"scan: dataset folder skipped for images ({len(images)} machine-named, no camera EXIF): "
                    f"{os.path.relpath(dirpath, HOME)}")
            for f in files:
                full = os.path.join(dirpath, f)
                if skipped_name(f, cfg):
                    counts["skipped_secret_or_hidden"] += 1
                    continue
                ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
                if ext not in exts:
                    counts["skipped_other_type"] += 1
                    continue
                if ext in skip_types:
                    counts["skipped_by_choice"] += 1
                    continue
                if dataset and ext in media.IMAGE_EXT:
                    counts["skipped_dataset_images"] += 1
                    continue
                if os.path.islink(full):
                    counts["skipped_links"] += 1
                    continue
                try:
                    st = os.stat(full)
                except OSError:
                    counts["unreadable"] += 1
                    continue
                if getattr(st, "st_flags", 0) & SF_DATALESS:
                    counts["skipped_cloud_only"] += 1
                    continue
                if st.st_size > (media_bytes if ext in MEDIA_EXT else max_bytes):
                    counts["skipped_too_big"] += 1
                    continue
                yield full, ext, st


# --- extraction ------------------------------------------------------------------------------------------------------

def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").replace("\ufeff", "")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def run(cmd, cfg):
    r = subprocess.run(cmd, capture_output=True, timeout=cfg["extract"]["timeout_s"], stdin=subprocess.DEVNULL)
    if r.returncode != 0 and not r.stdout:
        raise RuntimeError(r.stderr.decode("utf-8", "ignore").strip()[:200] or f"exit {r.returncode}")
    return r.stdout.decode("utf-8", "ignore")


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def pptx_pages(path):
    z = zipfile.ZipFile(path)
    slides = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
    slides.sort(key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
    pages = []
    for i, n in enumerate(slides, 1):
        xml = z.read(n).decode("utf-8", "ignore")
        runs = re.findall(r"<a:t>([^<]*)</a:t>|(</a:p>)", xml)
        text = "".join("\n" if br else html.unescape(t) + " " for t, br in runs)
        pages.append((i, text))
    return pages


def local(tag):
    return tag.rsplit("}", 1)[-1]


def xlsx_pages(path):
    """One page per sheet; each row as 'a | b | c'. Shared and inline strings resolved; formulas give their value."""
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    shared = []
    if "xl/sharedStrings.xml" in names:
        for _, el in ET.iterparse(z.open("xl/sharedStrings.xml")):
            if local(el.tag) == "si":
                shared.append("".join(t.text or "" for t in el.iter() if local(t.tag) == "t"))
                el.clear()
    rels = {}
    if "xl/_rels/workbook.xml.rels" in names:
        for el in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")):
            rels[el.get("Id")] = el.get("Target", "").lstrip("/").removeprefix("xl/")
    sheets = []
    for el in ET.fromstring(z.read("xl/workbook.xml")).iter():
        if local(el.tag) == "sheet":
            rid = next((v for k, v in el.attrib.items() if local(k) == "id"), None)
            target = "xl/" + rels.get(rid, "")
            if target in names:
                sheets.append((el.get("name", ""), target))
    pages = []
    for i, (name, target) in enumerate(sheets, 1):
        lines = [f"Sheet: {name}"]
        row, cell_type, value = [], None, None
        for event, el in ET.iterparse(z.open(target), events=("start", "end")):
            tag = local(el.tag)
            if event == "start":
                if tag == "c":
                    cell_type, value = el.get("t"), None
                continue
            if tag == "v":
                value = el.text
            elif tag == "is":
                value = "".join(t.text or "" for t in el.iter() if local(t.tag) == "t")
            elif tag == "c":
                if value not in (None, ""):
                    if cell_type == "s":
                        try:
                            value = shared[int(value)]
                        except (ValueError, IndexError):
                            pass
                    row.append(value.strip())
                el.clear()
            elif tag == "row":
                if any(row):
                    lines.append(" | ".join(v for v in row if v))
                row = []
                el.clear()
        pages.append((i, "\n".join(lines)))
    return pages


def csv_pages(path):
    text = read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = csv.reader(io.StringIO(text), dialect)
    return [(None, "\n".join(" | ".join(c.strip() for c in r if c.strip()) for r in rows if any(c.strip() for c in r)))]


def extract(path, ext, cfg):
    """[(page or None, text)] for one file."""
    if ext == "pdf":
        out = run([cfg["extract"]["pdftotext"], "-enc", "UTF-8", path, "-"], cfg)
        return [(i, t) for i, t in enumerate(out.split("\f"), 1)]
    if ext in DOC_EXT:
        # textutil reads newer .doc files; older ones (e.g. a 2014 Chinese-locale Word file) it treats as plain text and
        # returns the binary, so the format is forced and LibreOffice reads what textutil refuses.
        try:
            text = run([cfg["extract"]["textutil"], "-format", "html" if ext == "htm" else ext, "-convert", "txt",
                        "-stdout", path], cfg)
        except (RuntimeError, subprocess.TimeoutExpired):
            text = ""
        if text.strip():
            return [(None, text)]
        with SOFFICE_LOCK:
            return [(None, run([cfg["extract"]["soffice"], f"-env:UserInstallation=file://{HERE}/.lo-profile",
                                "--headless", "--cat", path], cfg))]
    if ext == "pptx":
        return pptx_pages(path)
    if ext == "xlsx":
        return xlsx_pages(path)
    if ext == "csv":
        return csv_pages(path)
    return [(None, read_text(path))]


def chunk(pages, size, overlap):
    """Fixed-size chunks with overlap, broken at whitespace (DigUp's scheme); each tagged with its starting page."""
    parts, starts, page_of = [], [], []
    pos = 0
    for page, text in pages:
        text = clean(text)
        if not text:
            continue
        starts.append(pos)
        page_of.append(page)
        parts.append(text)
        pos += len(text) + 2
    full = "\n\n".join(parts)
    out = []
    start = 0
    while start < len(full):
        end = min(start + size, len(full))
        if end < len(full):
            cut = max(full.rfind(" ", start + size - overlap, end), full.rfind("\n", start + size - overlap, end))
            if cut > start:
                end = cut
        piece = full[start:end].strip()
        if piece:
            page = page_of[bisect.bisect_right(starts, start) - 1] if starts else None
            out.append((page, piece))
        if end >= len(full):
            break
        nxt = end - overlap
        sp = full.find(" ", nxt, end)
        start = sp + 1 if sp != -1 else nxt
    return out, len(full)


def work(path, ext, cfg):
    try:
        pages = extract(path, ext, cfg)
        chunks, chars = chunk(pages, cfg["chunk"]["size"], cfg["chunk"]["overlap"])
        return chunks, chars, None
    except Exception as e:  # one bad file must not stop the run
        return [], 0, f"{type(e).__name__}: {e}"[:300]


# --- stages ----------------------------------------------------------------------------------------------------------

def remove_file(db, file_id):
    rows = db.execute("SELECT id, text FROM chunks WHERE file = ?", (file_id,)).fetchall()
    ids = [r[0] for r in rows]
    db.executemany("DELETE FROM fts WHERE rowid = ?", [(i,) for i in ids])
    # tri is an external-content table: a delete must carry the old text, before the chunk row goes.
    db.executemany("INSERT INTO tri (tri, rowid, text) VALUES ('delete', ?, ?)", rows)
    for table in ("media", "media_done", "ocr_done"):
        db.execute(f"DELETE FROM {table} WHERE file = ?", (file_id,))
    db.executemany("DELETE FROM vectors WHERE chunk = ?", [(i,) for i in ids])
    db.execute("DELETE FROM chunks WHERE file = ?", (file_id,))
    db.execute("DELETE FROM files WHERE id = ?", (file_id,))


def extract_stage(db, cfg, stats):
    counts = Counter()
    known = {p: (i, s, m) for i, p, s, m in db.execute("SELECT id, path, size, mtime FROM files")}
    todo, seen = [], set()
    for full, ext, st in walk(cfg, counts):
        rel = os.path.relpath(full, HOME)
        seen.add(rel)
        counts["found_" + ext] += 1
        old = known.get(rel)
        if old and old[1] == st.st_size and old[2] == st.st_mtime:
            counts["unchanged"] += 1
            continue
        todo.append((full, rel, ext, st))
    if PATHS is not None:
        global ONLY
        ONLY = {rel for _, rel, _, _ in todo}
    prefixes = tuple(os.path.relpath(r, HOME) + "/" for r, _ in roots(cfg))
    # A full run also drops files of a folder taken out of the settings (a media root that is only unplugged stays).
    named = tuple(os.path.relpath(expand(r), HOME) + "/" for r in cfg["roots"] + cfg.get("media", {}).get("roots", []))
    gone = [i for p, (i, _, _) in known.items() if p not in seen
            and (p.startswith(prefixes) or (PATHS is None and not p.startswith(named)))]
    for i in gone:
        remove_file(db, i)
    counts["removed_gone"] = len(gone)
    log(f"scan: {sum(v for k, v in counts.items() if k.startswith('found_'))} indexable files, "
        f"{len(todo)} new or changed, {len(gone)} gone; " + json.dumps(dict(sorted(counts.items()))))

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(4) as ex:
        # Photos and videos have no text to extract: they are registered here and handled by the ocr/media stages.
        futs = {ex.submit(work, full, ext, cfg) if ext not in MEDIA_EXT else ex.submit(lambda: ([], 0, None)):
                (full, rel, ext, st) for full, rel, ext, st in todo}
        for fut in as_completed(futs):
            full, rel, ext, st = futs[fut]
            chunks, chars, err = fut.result()
            if rel in known:
                remove_file(db, known[rel][0])
            status = "error" if err else ("ok" if chunks else ("media" if ext in MEDIA_EXT else "empty"))
            counts["status_" + status] += 1
            cur = db.execute(
                "INSERT INTO files (path, ext, size, mtime, chunks, chars, status, error, indexed_at, grp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rel, ext, st.st_size, st.st_mtime, len(chunks), chars, status, err, time.time(),
                 media_key(rel) if ext in MEDIA_EXT else version_key(rel)))
            fid = cur.lastrowid
            for ordinal, (page, text) in enumerate(chunks):
                cid = db.execute("INSERT INTO chunks (file, ord, page, text) VALUES (?, ?, ?, ?)",
                                 (fid, ordinal, page, text)).lastrowid
                db.execute("INSERT INTO fts (rowid, name, body) VALUES (?, ?, ?)", (cid, rel, text))
                db.execute("INSERT INTO tri (rowid, text) VALUES (?, ?)", (cid, text))
            done += 1
            if done % 100 == 0:
                db.commit()
                log(f"extract: {done}/{len(todo)} files, {time.time() - t0:.0f}s")
    db.commit()
    backfill(db)
    stats["extract_s"] = round(time.time() - t0, 1)
    stats["scan"] = dict(sorted(counts.items()))
    log(f"extract: done {done} files in {stats['extract_s']}s")


def backfill(db):
    """Bring an index built before Phase 3b up to date: version groups and the trigram table."""
    rows = db.execute("SELECT id, path FROM files WHERE grp IS NULL").fetchall()
    if rows:
        db.executemany("UPDATE files SET grp = ? WHERE id = ?", [(version_key(p), i) for i, p in rows])
        log(f"backfill: version groups for {len(rows)} files")
    n_chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    n_tri = db.execute("SELECT count(*) FROM tri_docsize").fetchone()[0]
    if n_tri != n_chunks:
        t = time.time()
        db.execute("INSERT INTO tri (tri) VALUES ('rebuild')")
        log(f"backfill: trigram index rebuilt over {n_chunks} chunks in {time.time() - t:.0f}s")
    db.commit()


def on_battery():
    try:
        out = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "Battery Power" in out


def wait_for_power():
    """Embedding is the heavy part: hold it while the Mac runs on battery, resume on AC."""
    if NO_WAIT or not on_battery():
        return 0
    log("power: on battery, embedding paused (checking every 60 s)")
    t = time.time()
    if LOCK:
        fcntl.flock(LOCK, fcntl.LOCK_UN)
    while on_battery():
        time.sleep(60)
    if LOCK:
        fcntl.flock(LOCK, fcntl.LOCK_EX)
    log(f"power: back on AC after {(time.time() - t) / 60:.0f} min, resuming")
    return time.time() - t


def has_words(text):
    """DigUp's SearchText.hasWords: letters are at least a fifth of the non-space characters. A table of numbers
    (model outputs, coordinates, logs of figures) gets keyword search only: its vector would match nothing."""
    letters = others = 0
    for c in text:
        if c.isspace():
            continue
        if c.isalpha():
            letters += 1
        else:
            others += 1
    return letters > 0 and letters * 4 >= others


def embed_stage(db, cfg, stats):
    m = cfg["model"]
    # Every chunk with words gets a vector (embed_first_chunks = 0 lifts DigUp's first-pass cap); numbers-only chunks
    # stay keyword-only. Resumable: only chunks without a vector are picked up, and each block of 512 is committed.
    cap = m.get("embed_first_chunks", 0) or 10**9
    rows = db.execute("SELECT c.id, c.text, f.path FROM chunks c JOIN files f ON f.id = c.file "
                      "LEFT JOIN vectors v ON v.chunk = c.id WHERE v.chunk IS NULL AND c.ord < ? ORDER BY c.id",
                      (cap,)).fetchall()
    if NO_WAIT and on_battery():
        log("embed: on battery, left for the next run on mains power")
        stats["embed_s"] = 0
        return
    todo = [r for r in rows if has_words(r[1]) and (ONLY is None or r[2] in ONLY)]
    stats["embed_skipped_no_words"] = len(rows) - len(todo)
    stats["chunks_fts_only"] = db.execute("SELECT count(*) FROM chunks c LEFT JOIN vectors v ON v.chunk = c.id "
                                          "WHERE v.chunk IS NULL").fetchone()[0] - len(todo)
    total = len(todo)
    have = db.execute("SELECT count(*) FROM vectors").fetchone()[0]
    log(f"embed: {total} chunks need vectors ({have} already done, {len(rows) - len(todo)} numbers-only "
        "chunks stay keyword-only)")
    stats["embed_chunks"] = total
    if not total:
        stats["embed_s"] = 0
        return
    t0, last_log, done, failed, paused = time.time(), 0, 0, 0, 0.0
    paused += wait_for_power()  # before llama-server starts, so nothing sits in memory while on battery
    emb = Embedder(cfg, log)
    try:
        with ThreadPoolExecutor(m["workers"]) as ex:
            for start in range(0, total, 512):
                paused += wait_for_power()
                rows = todo[start:start + 512]
                batches = [rows[i:i + m["batch"]] for i in range(0, len(rows), m["batch"])]
                texts = [[m["doc_prefix"].format(title=Path(p).stem or "none", text=t) for _, t, p in b]
                         for b in batches]
                for b, vecs in zip(batches, ex.map(emb.embed, texts)):
                    for (cid, _, _), v in zip(b, vecs):
                        if v is None:
                            failed += 1
                            continue
                        db.execute("INSERT INTO vectors (chunk, vec) VALUES (?, ?)",
                                   (cid, v.astype(np.float16).tobytes()))
                    done += len(b)
                db.commit()
                if time.time() - last_log > 30:
                    last_log = time.time()
                    rate = done / max(time.time() - t0 - paused, 1)
                    log(f"embed: {done}/{total} ({rate:.1f} chunks/s, ETA {(total - done) / rate / 60:.1f} min), "
                        f"failed {failed}, llama-server RSS {emb.peak_server_kb / 1024:.0f} MB peak")
    finally:
        emb.close()
    stats["embed_s"] = round(time.time() - t0, 1)
    stats["embed_failed"] = failed
    stats["embed_paused_s"] = round(paused, 1)
    stats["embed_rate"] = round(done / max(stats["embed_s"] - paused, 0.1), 1)
    stats["llama_server_peak_rss_mb_sampled"] = round(emb.peak_server_kb / 1024)
    log(f"embed: done {done} chunks in {stats['embed_s']}s, {failed} failed")


# --- OCR: image-only PDF pages and screenshots -------------------------------------------------------------------------

def add_chunks(db, fid, rel, pages, cfg):
    """Append chunks for (page, text) pairs to a file already in the index; returns how many."""
    start = db.execute("SELECT coalesce(max(ord) + 1, 0) FROM chunks WHERE file = ?", (fid,)).fetchone()[0]
    chunks, chars = chunk(pages, cfg["chunk"]["size"], cfg["chunk"]["overlap"])
    for ordinal, (page, text) in enumerate(chunks, start):
        cid = db.execute("INSERT INTO chunks (file, ord, page, text) VALUES (?, ?, ?, ?)",
                         (fid, ordinal, page, text)).lastrowid
        db.execute("INSERT INTO fts (rowid, name, body) VALUES (?, ?, ?)", (cid, rel, text))
        db.execute("INSERT INTO tri (rowid, text) VALUES (?, ?)", (cid, text))
    db.execute("UPDATE files SET chunks = chunks + ?, chars = chars + ?, status = CASE WHEN status IN ('empty', "
               "'media') THEN 'ocr' ELSE status END WHERE id = ?", (len(chunks), chars, fid))
    return len(chunks)


def ocr_file(full, ext, cfg):
    """(pages OCR'd, [(page, text)], error). PDFs: only pages without a text layer; images: the whole picture."""
    try:
        if ext == "pdf":
            _, pages = ocr.image_only_pages(full, cfg["ocr"]["min_chars"])
            texts = ocr.ocr_pdf_pages(full, pages, cfg["ocr"]["engine"]) if pages else {}
            return len(pages), sorted(texts.items()), None
        return 1, [(None, ocr.ENGINES[cfg["ocr"]["engine"]](full))], None
    except (OSError, subprocess.SubprocessError) as e:
        return 0, [], str(e)[:300]


def ocr_stage(db, cfg, stats):
    """Read text from image-only PDF pages and screenshots into the index (keyword, exact-code table, and vectors via
    the embed stage). Nothing is written to the PDFs or images; renders go to a temporary folder."""
    shots = {os.path.relpath(p, HOME) for p in media.screenshots([r for r, _ in roots(cfg)])}
    rows = db.execute("SELECT f.id, f.path, f.ext FROM files f LEFT JOIN ocr_done o ON o.file = f.id "
                      "WHERE o.file IS NULL AND f.status != 'error'").fetchall()
    todo = [(fid, rel, ext) for fid, rel, ext in rows if (ONLY is None or rel in ONLY) and ext == "pdf" or (
        ONLY is None or rel in ONLY) and (
        ext in media.IMAGE_EXT and (rel in shots or media.SCREENSHOT_NAME.match(os.path.basename(rel))))]
    log(f"ocr: {len(todo)} files to check ({sum(e != 'pdf' for _, _, e in todo)} screenshots), "
        f"engine {cfg['ocr']['engine']}")
    def timed(row):
        t = time.time()
        return ocr_file(str(HOME / row[1]), row[2], cfg), time.time() - t

    t0, pages_done, added, files_hit, paused = time.time(), 0, 0, 0, 0.0
    with ThreadPoolExecutor(2) as ex:
        for start in range(0, len(todo), 50):
            paused += wait_for_power()
            block = todo[start:start + 50]
            for (fid, rel, ext), ((n, texts, err), secs) in zip(block, ex.map(timed, block)):
                texts = [(p, t) for p, t in texts if t.strip()]
                n_chunks = add_chunks(db, fid, rel, texts, cfg) if texts else 0
                chars = sum(len(t) for _, t in texts)
                db.execute("INSERT OR REPLACE INTO ocr_done VALUES (?, ?, ?, ?, ?, ?)",
                           (fid, cfg["ocr"]["engine"], n, chars, round(secs, 2), err))
                pages_done += n
                added += n_chunks
                files_hit += bool(n_chunks)
                if err:
                    log(f"ocr: error {rel}: {err}")
            db.commit()
            log(f"ocr: {min(start + 50, len(todo))}/{len(todo)} files, {pages_done} pages, {added} chunks, "
                f"{time.time() - t0 - paused:.0f}s")
    stats["ocr_s"] = round(time.time() - t0, 1)
    stats["ocr_paused_s"] = round(paused, 1)
    stats["ocr_pages"] = pages_done
    stats["ocr_files_with_text"] = files_hit
    stats["ocr_chunks"] = added
    log(f"ocr: done {pages_done} pages, {added} chunks from {files_hit} files in {stats['ocr_s']}s")


# --- photos, video frames, soundtrack ---------------------------------------------------------------------------------

def media_key(rel):
    """Version group for a photo or video: only true copies group ("x (1).jpg", "x copy.jpg", "x 2.jpg"). The text rule
    would strip the year from DSC_2019.JPG and merge a whole camera roll."""
    folder, _, name = rel.rpartition("/")
    stem, _, ext = name.rpartition(".")
    stem = re.sub(r"(\s*\(\d+\)|\s+copy(\s+\d+)?|\s+\d)$", "", stem.lower())
    return f"{folder}/{stem}.{ext.lower()}"


def why_unreadable(full):
    try:
        with open(full, "rb") as f:
            f.read(1)
    except PermissionError:
        attrs = subprocess.run(["/usr/bin/xattr", full], capture_output=True, text=True).stdout.split()
        return "app-restricted (com.apple.macl)" if "com.apple.macl" in attrs else "permission denied"
    except OSError as e:
        return str(e)[:200]
    with open(full, "rb") as f:
        head = f.read(16)
    if not head:
        return "empty file"
    if head.startswith(b"bplist00"):
        return "not an image: empty placeholder left by an interrupted AirDrop"
    if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
        return "not an image: a saved web page"
    return None


def media_file(emb, full, ext):
    """([(kind, t, t_end, vector)], note): note is None, or why the file gave nothing (icon, unreadable, ...)."""
    if ext in media.IMAGE_EXT:
        try:
            w, h, _ = media.image_info(full)
            if max(w, h) < media.MIN_PX:
                return [], f"skipped: icon {w}x{h}"
            v = emb.embed_media(media.thumbnail(full))
        except OSError as e:
            return [], why_unreadable(full) or str(e)
        return ([("image", None, None, v)], None) if v is not None else ([], "server refused the image")
    if why := why_unreadable(full):
        return [], why
    out = []
    try:
        duration, has_video, has_audio = media.probe(full)
        if ext in media.AUDIO_EXT:
            has_video = False   # an mp3's cover art shows up as a one-frame video stream
        with tempfile.TemporaryDirectory(prefix="localsearch-media-") as tmp:
            if has_video:
                for t, jpg in media.video_frames(full, tmp):
                    v = emb.embed_media(jpg.read_bytes())
                    if v is not None:
                        out.append(("frame", t, None, v))
            if has_audio:
                for t, t_end, wav in media.audio_windows(full, tmp, duration):
                    v = emb.embed_media(wav.read_bytes())
                    if v is not None:
                        out.append(("audio", t, t_end, v))
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return out, str(e)[:300]
    return out, None if out else "no frames or sound decoded"


def media_stage(db, cfg, stats):
    """Vectors for photos (448 px thumbnails), one video frame per shot (at least every 10 s) and 30 s soundtrack
    windows, in the same space as the text. Sources are only read; thumbnails and clips live in a temp folder."""
    rows = db.execute(f"SELECT f.id, f.path, f.ext FROM files f LEFT JOIN media_done d ON d.file = f.id "
                      f"WHERE (d.file IS NULL OR d.error LIKE 'permission%' OR d.error LIKE 'app-restricted%' OR d.error LIKE 'unreadable%' OR d.error LIKE 'no frames%') AND f.ext IN ({','.join('?' * len(MEDIA_EXT))}) "
                      f"ORDER BY f.ext IN ({','.join('?' * len(media.VIDEO_EXT))}), f.id",
                      (*MEDIA_EXT, *media.VIDEO_EXT)).fetchall()
    if ONLY is not None:
        rows = [row for row in rows if row[1] in ONLY]
    if NO_WAIT and on_battery() and rows:
        log(f"media: on battery, {len(rows)} files left for the next run on mains power")
        rows = []
    n_img = sum(e in media.IMAGE_EXT for _, _, e in rows)
    log(f"media: {n_img} images and {len(rows) - n_img} videos to embed")
    stats["media_files"] = len(rows)
    if not rows:
        stats["media_s"] = 0
        return
    t0, last_log, done, vecs, paused = time.time(), 0, 0, Counter(), 0.0
    notes = Counter()
    paused += wait_for_power()
    emb = Embedder(cfg, log, media=True)

    def one(row):
        t = time.time()
        out, note = media_file(emb, str(HOME / row[1]), row[2])
        return out, note, time.time() - t

    try:
        with ThreadPoolExecutor(cfg["model"]["workers"]) as ex:
            for start in range(0, len(rows), 32):
                paused += wait_for_power()
                block = rows[start:start + 32]
                for (fid, rel, ext), (out, note, secs) in zip(block, ex.map(one, block)):
                    for kind, t, t_end, v in out:
                        db.execute("INSERT INTO media (file, kind, t, t_end, vec) VALUES (?, ?, ?, ?, ?)",
                                   (fid, kind, t, t_end, v.astype(np.float16).tobytes()))
                        vecs[kind] += 1
                    db.execute("INSERT OR REPLACE INTO media_done VALUES (?, ?, ?, ?, ?)",
                               (fid, sum(k == "frame" for k, *_ in out), sum(k == "audio" for k, *_ in out),
                                round(secs, 2), note))
                    if note:
                        notes[note.split(":")[0].split(" (")[0]] += 1
                        if not note.startswith("skipped"):
                            log(f"media: {rel}: {note}")
                    done += 1
                db.commit()
                if time.time() - last_log > 30:
                    last_log = time.time()
                    rate = done / max(time.time() - t0 - paused, 1)
                    log(f"media: {done}/{len(rows)} files ({rate:.2f}/s, ETA {(len(rows) - done) / rate / 60:.0f} "
                        f"min), {dict(vecs)}, llama-server RSS {emb.peak_server_kb / 1024:.0f} MB peak")
    finally:
        emb.close()
    stats["media_s"] = round(time.time() - t0, 1)
    stats["media_paused_s"] = round(paused, 1)
    stats["media_vectors"] = dict(vecs)
    stats["media_notes"] = dict(notes)
    stats["media_server_peak_rss_mb_sampled"] = round(emb.peak_server_kb / 1024)
    log(f"media: done {done} files in {stats['media_s']}s, {dict(vecs)}, {dict(notes)}")


def stop(*_):
    """SIGTERM: stop llama-server and leave at once. sys.exit would wait for every OCR or media job already queued
    in the worker pools (minutes); the uncommitted block is rolled back by SQLite and redone on the next run."""
    for emb in list(common.ACTIVE):
        emb.close()
    log("stopped (SIGTERM); uncommitted work is redone on the next run")
    os._exit(143)


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true", help="extract and OCR only")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset of " + ",".join(STAGES))
    ap.add_argument("--rebuild", action="store_true", help="empty the index first (source files are never touched)")
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    ap.add_argument("--paths", nargs="+", help="only these folders, recursively (new, changed and removed files)")
    ap.add_argument("--no-wait", action="store_true", help="on battery, skip embedding instead of waiting")
    args = ap.parse_args()
    global PATHS, NO_WAIT, LOCK
    NO_WAIT = args.no_wait
    if args.paths:
        PATHS = [Path(os.path.abspath(os.path.expanduser(p))) for p in args.paths]
    # One indexer at a time: a quick folder update gives way to a running refresh, a refresh waits for an update.
    (HERE / "state").mkdir(exist_ok=True)   # absent in a fresh copy
    lock = open(HERE / "state" / "indexer.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if PATHS else 0))
    except BlockingIOError:
        print("another indexer is running", flush=True)
        return 75
    LOCK = lock
    cfg = load_config(Path(args.config))
    logdir = expand(cfg["log_dir"])
    logdir.mkdir(parents=True, exist_ok=True)
    LOG = open(logdir / "index.log", "a")
    t0 = time.time()
    stats = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "roots": cfg["roots"]}
    log(f"index start: roots {cfg['roots']} -> {expand(cfg['index'])}")
    signal.signal(signal.SIGTERM, stop)
    db = connect(cfg)
    if args.rebuild:
        db.execute("INSERT INTO tri (tri) VALUES ('delete-all')")
        for table in ("fts", "vectors", "chunks", "files", "meta"):
            db.execute(f"DELETE FROM {table}")
        db.commit()
        db.execute("VACUUM")
        log("rebuild: index emptied")
    stages = [st for st in STAGES if st in args.stages.split(",")]
    if args.no_embed:
        stages = [st for st in stages if st not in ("embed", "media")]
    stats["stages"] = stages
    for st in stages:
        {"extract": extract_stage, "ocr": ocr_stage, "embed": embed_stage, "media": media_stage}[st](db, cfg, stats)
    stats["total_s"] = round(time.time() - t0, 1)
    stats["indexer_peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576)
    stats["children_peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576)
    q = lambda s: db.execute(s).fetchone()[0]
    stats["files"] = dict(db.execute("SELECT status, count(*) FROM files GROUP BY status").fetchall())
    stats["files_by_ext"] = dict(db.execute("SELECT ext, count(*) FROM files GROUP BY ext").fetchall())
    stats["chunks"] = q("SELECT count(*) FROM chunks")
    stats["vectors"] = q("SELECT count(*) FROM vectors")
    stats["media_vectors_total"] = dict(db.execute("SELECT kind, count(*) FROM media GROUP BY kind").fetchall())
    stats["pdf_no_text"] = q("SELECT count(*) FROM files WHERE ext = 'pdf' AND status = 'empty'")
    stats["db_mb"] = round(os.path.getsize(expand(cfg["index"])) / 1e6, 1)
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    full = stats["stages"] == STAGES and PATHS is None
    if full:
        db.execute("INSERT OR REPLACE INTO meta VALUES ('last_index', ?)", (json.dumps(stats),))
        db.commit()
    # eval.py reads reports/index_stats.json, so only a full run on the default config writes it there; a partial
    # run (--stages) or another config keeps its stats next to its own log
    out = HERE / "reports" if full and Path(args.config).resolve() == HERE / "config.toml" else logdir
    out.mkdir(exist_ok=True)
    (out / "index_stats.json").write_text(json.dumps(stats, indent=1))
    log("index done: " + json.dumps(stats))


if __name__ == "__main__":
    sys.exit(main())
