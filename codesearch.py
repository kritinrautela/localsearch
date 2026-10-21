"""Code search: a separate small index of source files, split at functions and classes, searched by keyword and by
meaning together. The search window's code: prefix uses it; a result opens the editor at its line.

    uv run python codesearch.py --index            # new and changed files; embedding waits for mains power
    uv run python codesearch.py "where do we parse the hotkey"

Reads files only. Never runs git or any tool inside a repository (a repository's config could run code).
"""

import argparse
import fcntl
import fnmatch
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from common import HERE, HOME, Embedder, expand, fold, load_config

CODE_EXT = {"py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "go", "rs", "swift", "java", "kt", "c", "h", "cc", "cpp",
            "hpp", "m", "mm", "rb", "php", "sh", "zsh", "bash", "sql", "lua", "r", "scala", "cs", "vue", "svelte",
            "html", "css", "scss", "toml", "yaml", "yml", "ipynb"}
DOC_NAMES = ("readme", "changelog", "contributing", "architecture", "design")   # a repo's own docs, as DigUp reads
SKIP_DIRS = {"node_modules", "__pycache__", "dist", "build", "target", "vendor", "Pods", "DerivedData", "venv",
             "site-packages", "out", "coverage", "work"}
DEFAULT_ROOTS = ["~/Projects", "~/Developer", "~/localsearch", "~/Desktop", "~/Documents"]   # [code] roots overrides
# A chunk holding something that looks like a key or password is never indexed, so it can never show in a preview.
SECRET = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{30,}|\bxox[abprs]-[A-Za-z0-9-]{10,}|\bAIza[0-9A-Za-z_-]{35}|\bhf_[A-Za-z0-9]{30,}"
    r"|(?i:(?:api[_-]?key|secret|passw(?:or)?d|token|bearer)[\"']?\s*[:=]\s*[\"'][^\"'\s]{12,}[\"'])")
MAX_BYTES = 400_000
MAX_LINES = 60
# A line that starts a unit worth its own chunk: a definition at the left margin (or one indent in, for methods).
UNIT = re.compile(r"^(\s{0,4})(?:export\s+)?(?:default\s+)?(?:pub(?:\(\w+\))?\s+)?(?:async\s+)?"
                  r"(?:def|class|function|func|fn|struct|enum|impl|interface|trait|type|module|object|protocol|"
                  r"extension|const\s+\w+\s*=\s*(?:async\s*)?\(|let\s+\w+\s*=\s*(?:async\s*)?\()\b")
NAME = re.compile(r"(?:def|class|function|func|fn|struct|enum|impl|interface|trait|type|protocol|extension|"
                  r"const|let)\s+([A-Za-z_][\w]*)")

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, size INTEGER, mtime REAL);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, file INTEGER NOT NULL, line INTEGER, line_end INTEGER,
                                   name TEXT, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(name, body, tokenize="unicode61 tokenchars '_'");
CREATE TABLE IF NOT EXISTS vectors (chunk INTEGER PRIMARY KEY, vec BLOB NOT NULL);
"""


def name_words(name):
    """The words inside a name, so "playlist" finds make_playlist and makePlaylist: "make playlist"."""
    parts = [p for p in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", name or "") if p]
    return " ".join(p.lower() for p in parts) if len(parts) > 1 else ""


def index_path(cfg):
    # Next to this file by default, so a copy of the program (a test, a second install) never writes into another's.
    return expand(cfg.get("code", {}).get("index", str(HERE / "code_index.sqlite")))


def code_roots(cfg):
    return [expand(r) for r in cfg.get("code", {}).get("roots", DEFAULT_ROOTS) if expand(r).is_dir()]


def connect(cfg):
    db = sqlite3.connect(index_path(cfg), check_same_thread=False, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def skipped(name, cfg):
    s = cfg["skip"]
    low = name.lower()
    return (name.startswith(".") or low in s["names"] or any(low.startswith(p) for p in s["name_prefixes"])
            or any(low.endswith(x) for x in s["suffixes"]))


class GitIgnore:
    """The .gitignore files met on the way down, read as text (git itself is never run). Supports the common forms:
    globs, dir/ patterns, /anchored patterns, ** and ! re-includes, last match winning."""

    def __init__(self):
        self.rules = {}   # folder -> [(negate, dir_only, anchored, pattern)]

    def load(self, folder):
        try:
            lines = open(os.path.join(folder, ".gitignore"), encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            return
        rules = []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            neg = ln.startswith("!")
            ln = ln[1:] if neg else ln
            dir_only = ln.endswith("/")
            ln = ln.rstrip("/")
            anchored = "/" in ln
            rules.append((neg, dir_only, anchored, ln.lstrip("/")))
        if rules:
            self.rules[folder] = rules

    def ignored(self, path, is_dir):
        hit = False
        for folder, rules in self.rules.items():
            if not path.startswith(folder + "/"):
                continue
            rel = path[len(folder) + 1:]
            for neg, dir_only, anchored, pat in rules:
                if dir_only and not is_dir:
                    continue
                target = rel if anchored else os.path.basename(rel)
                if fnmatch.fnmatchcase(target, pat) or (anchored and fnmatch.fnmatchcase(rel, pat.replace("**/", ""))):
                    hit = not neg
        return hit


def notebook_text(full):
    """A notebook's cells as plain text, outputs left out."""
    import json
    nb = json.load(open(full, encoding="utf-8", errors="replace"))
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if src.strip():
            parts.append(("# %% [markdown]\n" if cell.get("cell_type") == "markdown" else "# %%\n") + src)
    return "\n\n".join(parts)


def walk(cfg, only=None):
    under = [str(expand(u)) for u in cfg["skip"]["under"]]
    bundles = tuple(cfg["skip"]["bundle_suffixes"])
    seen = set()
    ignore = GitIgnore()
    for root in (only or code_roots(cfg)):
        for dirpath, dirs, files in os.walk(root):
            ignore.load(dirpath)
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS
                       and not d.lower().endswith(bundles) and not any(os.path.join(dirpath, d).startswith(u)
                                                                       for u in under)
                       and not os.path.islink(os.path.join(dirpath, d))
                       and not ignore.ignored(os.path.join(dirpath, d), True)]
            for f in files:
                ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
                full = os.path.join(dirpath, f)
                doc = ext == "md" and (f.lower().startswith(DOC_NAMES) or os.path.basename(dirpath).lower() == "docs")
                if (ext not in CODE_EXT and not doc) or skipped(f, cfg) or f.endswith((".min.js", ".min.css", ".lock")):
                    continue
                if ignore.ignored(full, False) or getattr(os.stat(full, follow_symlinks=False), "st_flags", 0) & 0x40000000:
                    continue
                if full in seen or os.path.islink(full):
                    continue
                seen.add(full)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                if 0 < st.st_size <= MAX_BYTES:
                    yield full, st


COMMENT = ("#", "//", "/*", "@")


def lead_in(lines, i, floor):
    """Where a definition's chunk starts: above it, over its own comments and decorators (and the blank lines between
    a section comment and it), so that comment is searched with the code it describes, not the code before it."""
    top = k = i
    while k - 1 > floor:
        s = lines[k - 1].strip()
        if s.startswith(COMMENT) and not s.startswith("#!"):
            k -= 1
            top = k
        elif not s:
            k -= 1
        else:
            break
    return top


def split(text):
    """[(first line, last line, name, text)]: a chunk per definition, long ones cut into windows of MAX_LINES."""
    lines = text.splitlines()
    starts, units = [], {}   # chunk start -> the definition line it leads into (the name is read there)
    for i, ln in enumerate(lines):
        if UNIT.match(ln):
            top = lead_in(lines, i, starts[-1] if starts else -1)
            starts.append(top)
            units[top] = i
    if not starts or starts[0] > 0:
        starts = [0] + starts
    out = []
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        name_m = NAME.search(lines[units.get(a, a)]) if a < len(lines) else None
        name = name_m[1] if name_m else ""
        for s in range(a, b, MAX_LINES - 10):
            e = min(b, s + MAX_LINES)
            body = "\n".join(lines[s:e]).strip()
            if len(body) >= 20:
                out.append((s + 1, e, name, body))
            if e >= b:
                break
    return out


def index(cfg, only=None, log=print):
    db = connect(cfg)
    known = {p: (i, m, s) for i, p, m, s in db.execute("SELECT id, path, mtime, size FROM files")}
    found, changed = set(), 0
    for full, st in walk(cfg, only):
        rel = os.path.relpath(full, HOME)
        found.add(rel)
        old = known.get(rel)
        if old and old[1] == st.st_mtime and old[2] == st.st_size:
            continue
        try:
            text = (notebook_text(full) if full.endswith(".ipynb")
                    else open(full, encoding="utf-8", errors="replace").read())
        except (OSError, ValueError):
            continue
        if "\x00" in text[:2000]:
            continue
        if old:
            drop(db, old[0])
        fid = db.execute("INSERT INTO files (path, size, mtime) VALUES (?, ?, ?)",
                         (rel, st.st_size, st.st_mtime)).lastrowid
        for line, end, name, body in split(text):
            if SECRET.search(body):
                continue
            cid = db.execute("INSERT INTO chunks (file, line, line_end, name, text) VALUES (?, ?, ?, ?, ?)",
                             (fid, line, end, name, body)).lastrowid
            db.execute("INSERT INTO fts (rowid, name, body) VALUES (?, ?, ?)",
                       (cid, os.path.basename(full) + " " + name + " " + name_words(name), body))
        changed += 1
    scope = [str(r) for r in (only or code_roots(cfg))]
    # A full run also drops files of a folder taken out of the code roots.
    gone = [(i, p) for p, (i, _, _) in known.items() if p not in found
            and (only is None or any(str(HOME / p).startswith(r.rstrip("/") + "/") for r in scope))]
    for fid, _ in gone:
        drop(db, fid)
    # chunks indexed before the secret check existed
    bad = [cid for cid, body in db.execute("SELECT id, text FROM chunks") if SECRET.search(body)]
    db.executemany("DELETE FROM fts WHERE rowid = ?", [(i,) for i in bad])
    db.executemany("DELETE FROM vectors WHERE chunk = ?", [(i,) for i in bad])
    db.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in bad])
    db.commit()
    log(f"code: {len(found)} files, {changed} new or changed, {len(gone)} gone")
    return db


def drop(db, fid):
    ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE file = ?", (fid,))]
    db.executemany("DELETE FROM fts WHERE rowid = ?", [(i,) for i in ids])
    db.executemany("DELETE FROM vectors WHERE chunk = ?", [(i,) for i in ids])
    db.execute("DELETE FROM chunks WHERE file = ?", (fid,))
    db.execute("DELETE FROM files WHERE id = ?", (fid,))


def embed(db, cfg, log=print):
    rows = db.execute("SELECT c.id, c.text, f.path FROM chunks c JOIN files f ON f.id = c.file "
                      "LEFT JOIN vectors v ON v.chunk = c.id WHERE v.chunk IS NULL").fetchall()
    if not rows:
        return
    log(f"code: embedding {len(rows)} chunks")
    m = cfg["model"]
    emb = Embedder(cfg, log)
    try:
        for i in range(0, len(rows), m["batch"]):
            block = rows[i:i + m["batch"]]
            vecs = emb.embed([m["doc_prefix"].format(title=os.path.basename(p), text=t[:1500]) for _, t, p in block])
            for (cid, _, _), v in zip(block, vecs):
                if v is not None:
                    db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (cid, v.astype(np.float16).tobytes()))
            if i % 512 == 0:
                db.commit()
        db.commit()
    finally:
        emb.close()


class CodeSearcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.db = sqlite3.connect(f"file:{index_path(cfg)}?mode=ro", uri=True, check_same_thread=False)
        rows = self.db.execute("SELECT chunk, vec FROM vectors ORDER BY chunk").fetchall()
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.matrix = (np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), -1)
                       if rows else np.zeros((0, cfg["model"]["dim"]), np.float16))

    def search(self, q, qvec=None, k=30):
        words = [w for w in re.split(r"[^\w]+", q) if len(w) >= 2]
        ranks = {}
        if words:
            toks = ['"' + w.replace('"', "") + '"*' for w in words]
            joined = '"' + "_".join(w.replace('"', "") for w in words) + '"*'   # hotkey label -> hotkey_label
            for pat, weight in (("name : " + joined, 6.0), (joined, 3.0), (" AND ".join(toks), 2.0), (" OR ".join(toks), 1.0)):
                if weight == 2.0 and len(toks) < 2:
                    continue
                try:
                    for i, (cid,) in enumerate(self.db.execute(
                            "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 6.0, 1.0) LIMIT 300",
                            (pat,))):
                        ranks[cid] = ranks.get(cid, 0) + weight / (30 + i)
                except sqlite3.OperationalError:
                    pass
        if qvec is not None and len(self.matrix) and np.any(qvec):
            sims = self.matrix.astype(np.float32) @ qvec.astype(np.float32)
            for i, row in enumerate(np.argsort(-sims)[:300]):
                cid = int(self.ids[row])
                ranks[cid] = ranks.get(cid, 0) + 1 / (30 + i)
        lw = [fold(w) for w in words]
        out, per_file = [], {}
        for cid in sorted(ranks, key=lambda c: -ranks[c]):
            r = self.db.execute("SELECT f.path, c.line, c.line_end, c.name, c.text FROM chunks c "
                                "JOIN files f ON f.id = c.file WHERE c.id = ?", (cid,)).fetchone()
            if not r:
                continue
            path, line, end, name, text = r
            if per_file.get(path, 0) >= 2:
                continue
            per_file[path] = per_file.get(path, 0) + 1
            at = line   # the first line holding a query word, so the editor lands on it
            for j, ln in enumerate(text.splitlines()):
                if lw and any(w in fold(ln) for w in lw):
                    at = line + j
                    break
            out.append({"path": path, "line": at, "start": line, "end": end, "name": name, "text": text,
                        "score": ranks[cid]})
            if len(out) >= k:
                break
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--paths", nargs="+")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.index:
        (HERE / "state").mkdir(exist_ok=True)
        lock = open(HERE / "state" / "code.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("code: another run is going")
            return
        only = [Path(os.path.abspath(os.path.expanduser(p))) for p in args.paths] if args.paths else None
        t = time.time()
        db = index(cfg, only)
        from watch import on_battery
        if not args.no_embed and not on_battery():
            embed(db, cfg)
        print(f"code: done in {time.time() - t:.0f}s")
        return
    s = CodeSearcher(cfg)
    emb = Embedder(cfg, log=lambda m: None) if len(s.matrix) else None
    try:
        hits = s.search(args.query, emb.query(args.query) if emb else None, 10)
    finally:
        if emb:
            emb.close()
    for h in hits:
        print(f"{h['path']}:{h['line']}  {h['name']}\n    {h['text'].splitlines()[0][:120]}")


if __name__ == "__main__":
    main()
