"""LocalSearch self-test: run after every change, before calling it done. Prints PASS/FAIL lines and counts only, never
file names or text from the index.

    .venv/bin/python tools/selftest.py          # a fresh install on made-up files, in a temp folder (about 2 minutes)
    .venv/bin/python tools/selftest.py --live   # also this Mac's running app, its log and search quality

The fresh-install part copies the source into a temp folder, indexes made-up documents and code with the real model,
and checks meaning, keyword and code search, an empty index, an index without vectors yet, and that no llama-server is
ever left behind. Its servers use ports 18090-18093, away from the app's and the indexer's.
"""

import argparse
import copy
import json
import os
import py_compile
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from common import Embedder, load_config, orphan_server  # noqa: E402

SOURCES = ["app.py", "codesearch.py", "common.py", "indexer.py", "media.py", "ocr.py", "search.py", "watch.py"]
DOCS = {
    "garden.txt": "Quarterly budget for the garden project: seeds, soil and a new hose.",
    "trip.md": "Packing list for the Hong Kong trip: passport, charger, umbrella.",
    "curry.txt": "Recipe for butter chicken with garam masala and cream.",
    "lease.txt": "Flat lease agreement, rent and bond conditions.",
    "exams.txt": "Study timetable for final exams in chemistry.",
}
FILLER = ["Notes from the physics lecture on thermodynamics", "Car insurance renewal letter due in March",
          "Guitar chords for a slow acoustic song", "Minutes of the school committee meeting",
          "Workout plan with squats and a 5 km run", "Invoice for website hosting services",
          "Birthday party guest list and cake order", "How to repot a monstera plant",
          "Essay draft about ocean warming", "Phone bill summary with data usage",
          "Dentist appointment reminder for Tuesday", "Movie night list of comedies",
          "Cover letter for a marketing internship", "Yoga breathing exercises for sleep",
          "Electric bike comparison of range and price", "Wedding photography quote",
          "Instructions for the new wifi router", "Pet vaccination record for the cat",
          "Podcast notes about productivity habits", "Tax return checklist and receipts",
          "Laptop warranty and repair claim", "Volunteer roster for the community kitchen",
          "Piano practice log with scales", "Recipe for vegan banana bread", "Shopping list of milk and bread"]
MEANING = [("what do I need to bring when travelling", "trip.md"), ("gardening costs", "garden.txt"),
           ("cooking indian curry", "curry.txt"), ("renting an apartment contract", "lease.txt"),
           ("exam revision schedule", "exams.txt")]

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def guarded(name, fn):
    try:
        fn()
    except Exception as e:
        check(name, False, f"{type(e).__name__}: {e}")


def test_compile():
    bad = []
    for f in SOURCES + ["tools/selftest.py"]:
        try:
            py_compile.compile(str(HERE / f), doraise=True)
        except py_compile.PyCompileError:
            bad.append(f)
    check("every module compiles", not bad, ", ".join(bad))
    for sh in ["tools/install.sh", "tools/install_agent.sh", "tools/refresh.sh"]:
        r = subprocess.run(["/bin/zsh", "-n", str(HERE / sh)], capture_output=True)
        check(f"{sh} parses", r.returncode == 0)


def test_cloud_root(real):
    """A Google Drive root is searched although ~/Library is skipped; its "Mac Backups" folder and online-only files
    are passed over without being opened."""
    import collections
    import indexer
    tmp = Path(tempfile.mkdtemp(prefix="ls-cloud-"))
    try:
        drive = tmp / "Library" / "CloudStorage" / "Drive"
        (drive / "Mac Backups").mkdir(parents=True)
        (drive / "plan.txt").write_text("drive plan")
        (drive / "Mac Backups" / "copy.txt").write_text("a copy")
        cfg = copy.deepcopy(real)
        cfg.update(roots=[str(drive)], media={**real.get("media", {}), "roots": []})
        cfg["skip"] = {**real["skip"], "under": [str(tmp / "Library"), str(drive / "Mac Backups")]}
        saved, indexer.CLOUD = indexer.CLOUD, str(tmp / "Library" / "CloudStorage")
        try:
            found = [os.path.basename(f) for f, _, _ in indexer.walk(cfg, collections.Counter())]
        finally:
            indexer.CLOUD = saved
        check("a Google Drive folder is searched, its Mac Backups copy is not", found == ["plan.txt"], str(found))
        check("a file on this Mac counts as local, a missing one does not",
              indexer.on_this_mac(str(drive / "plan.txt")) and not indexer.on_this_mac(str(drive / "gone.txt")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_big_sheets():
    """A CSV keeps its first SHEET_CHARS of text and says how many rows were left out (one giant line too), and a
    file in a same_as copy folder is passed over only while the other copy has the same size."""
    import indexer
    tmp = Path(tempfile.mkdtemp(prefix="ls-sheet-"))
    saved = indexer.SHEET_CHARS
    try:
        indexer.SHEET_CHARS = 1000
        (tmp / "grid.csv").write_text("x,y,score\n" + "".join(f"{i},{i * 2},0.{i}\n" for i in range(5000)))
        (tmp / "line.csv").write_text("a," * 5000)
        grid = indexer.csv_pages(str(tmp / "grid.csv"))[0][1]
        line = indexer.csv_pages(str(tmp / "line.csv"))[0][1]
        check("a huge CSV keeps its start and notes the rows left out",
              grid.startswith("x | y | score") and len(grid) < 1100 and "more rows not indexed" in grid, str(len(grid)))
        check("a CSV of one giant line is cut to the limit too", len(line) <= 1000, str(len(line)))
        (tmp / "a").mkdir(), (tmp / "b").mkdir()
        (tmp / "a" / "f.txt").write_text("same"), (tmp / "b" / "f.txt").write_text("same")
        (tmp / "b" / "g.txt").write_text("only in the copy")
        m = {str(tmp / "b"): str(tmp / "a")}
        st = lambda n: os.stat(tmp / "b" / n)
        check("a file copied into a mirror folder is indexed once",
              indexer.copy_elsewhere(str(tmp / "b" / "f.txt"), st("f.txt"), m)
              and not indexer.copy_elsewhere(str(tmp / "b" / "g.txt"), st("g.txt"), m))
        db = sqlite3.connect(tmp / "t.sqlite")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE t (x)")
        db.executemany("INSERT INTO t VALUES (?)", [("x" * 4000,)] * 2000)
        db.commit()
        db.execute("DELETE FROM t")
        db.commit()
        before = os.path.getsize(tmp / "t.sqlite")
        ran = indexer.compact(db, min_free=1_000_000)
        check("a mostly-empty index file is shrunk after a run", ran and os.path.getsize(tmp / "t.sqlite") < before / 4,
              f"{before} -> {os.path.getsize(tmp / 't.sqlite')} bytes")
        check("a full index file is left alone", not indexer.compact(db, min_free=1_000_000))
        db.execute("CREATE VIRTUAL TABLE fts USING fts5(body)")
        db.execute("CREATE VIRTUAL TABLE tri USING fts5(text, tokenize='trigram')")
        for t in ("fts", "tri"):
            db.executemany(f"INSERT INTO {t} VALUES (?)", [("deleted text " * 50,)] * 500)
            db.commit()
            db.execute(f"DELETE FROM {t}")
            db.commit()
        indexer.REMOVED[0] = 10**9
        indexer.compact(db, min_free=1_000_000)
        check("after a big removal the word indexes are merged", indexer.REMOVED[0] == 0
              and db.execute("SELECT count(*) FROM tri_data").fetchone()[0] < 10)
        db.close()
    finally:
        indexer.SHEET_CHARS = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_spoken_dates():
    """"invoice from last week" filters by date like date:week; a phrase inside the query or a year in a name stays
    a search word."""
    from datetime import datetime
    import app
    day = lambda span: datetime.fromtimestamp(span[0]).strftime("%Y-%m-%d")
    now = datetime.now()
    spans = lambda q: [p for p in app.parse_filters(q)[2] if isinstance(p, tuple)]
    ok = [app.parse_filters("invoice from last week")[0] == "invoice" and len(spans("invoice from last week")) == 1,
          day(spans("notes yesterday")[0]) == (now.replace(hour=0) - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d"),
          day(spans("jorc table in sept 2025")[0]) == "2025-09-01",
          day(spans("trip in 2025")[0]) == "2025-01-01"]
    plain = ["today show notes", "tax return 2025", "notes in bali", "yesterday"]
    ok += [app.parse_filters(q)[0] == q and not spans(q) for q in plain]
    check("plain-English dates filter, plain words do not", all(ok), f"{sum(ok)}/{len(ok)}")
    line = "01:47:00 embed: 24064/45807 (2.3 chunks/s, ETA 160.1 min), failed 0"
    check("the menu says how long indexing has left, and when the Mac is busy",
          app.eta_words(line, load=0.5) == ", about 2 h 40 min left"
          and app.eta_words(line, load=3).endswith("(slower: the Mac is busy)") and app.eta_words("no eta") == "")


def test_app_typos():
    """An app name typed with one slip still finds the app as a Top Hit; everyday document words do not."""
    import re
    import app
    from common import fold

    class NoHistory:
        def count(self, p):
            return 0
    a = app.Apps.__new__(app.Apps)
    a.history, a.loading, a.loaded_at = NoHistory(), False, time.time()
    a.apps = []
    for name in ["Discord", "WhatsApp", "Google Chrome", "Spotify", "Obsidian", "Visual Studio Code", "Contacts",
                 "Tips", "Prime Video", "Microsoft PowerPoint", "Print Center", "Grammarly Desktop", "Notes"]:
        words = re.findall(r"[A-Za-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name))
        a.apps.append({"name": name, "path": f"/Applications/{name}.app", "low": fold(name),
                       "initials": "".join(w[0] for w in words).lower(), "words": [fold(w) for w in words]})
    top = lambda q: (a.match(q) or [{"name": None, "score": 0}])[0]
    slips = {"discrod": "Discord", "dicord": "Discord", "discords": "Discord", "chorme": "Google Chrome",
             "spotfy": "Spotify", "obsidain": "Obsidian", "watsapp": "WhatsApp"}
    ok = [top(q)["name"] == want and top(q)["score"] >= 75 for q, want in slips.items()]
    check("app names with a typo still open the app", all(ok), f"{sum(ok)}/{len(ok)}")
    # these may still list an app lower down, but only as a slip, which a document holding the word outranks
    words = ["visa", "trip", "deck", "core", "paint", "videos", "contract", "invoice", "report", "gold"]
    ok = [not (m := a.match(q)) or m[0]["score"] < 75 or m[0]["typo"] for q in words]
    check("everyday words are not taken for an app", all(ok), f"{sum(ok)}/{len(ok)}")
    ok = [top("discord")["name"] == "Discord" and not top("discord")["typo"], app.typo_distance("discrod", "discord") == 1,
          app.typo_prefix("xiscord", "discord") == 3]
    check("exact app names stay exact matches", all(ok), f"{sum(ok)}/{len(ok)}")
    tmp = Path(tempfile.mkdtemp())
    try:   # a throwaway History, so the real app_keys.json is never touched
        saved, app.STATE = app.STATE, tmp
        h = app.History()
        a.history = h
        before = top("dc")["name"]
        h.learn("dc", "/Applications/Discord.app")
        again = app.History()
        ok = [before != "Discord", top("dc")["name"] == "Discord", top("DC ")["name"] == "Discord",
              again.learned("dc") == {"/Applications/Discord.app": 1}]
        check("an app opened from short text comes first for that text next time", all(ok), f"{sum(ok)}/{len(ok)}")
    finally:
        app.STATE = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_pause_targets():
    """Pause stops the real indexer, not only the time/caffeinate wrappers (which then stuck every later refresh)."""
    import app
    ps = """  101 /usr/bin/time -l nice -n 10 .venv/bin/python indexer.py
  102 caffeinate -s /usr/bin/time -l nice -n 10 .venv/bin/python indexer.py
  103 /opt/homebrew/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python indexer.py
  104 /bin/zsh /Users/x/localsearch/tools/refresh.sh
  105 vim indexer.py
  106 /usr/bin/python3 other.py"""
    check("pause and resume reach the indexer itself", app.indexer_pids_in(ps) == [103]
          and app.indexer_pids_in(ps, wrappers=True) == [101, 102, 103])
    ok = [app.rename_problem("a.pdf", "b.pdf", str(HERE)) is None, app.rename_problem("a.pdf", "a.pdf", str(HERE)),
          app.rename_problem("a.pdf", "x/y.pdf", str(HERE)), app.rename_problem("a.pdf", ".hidden", str(HERE)),
          app.rename_problem("a.pdf", "app.py", str(HERE))]
    check("rename refuses empty, slashes, hidden names and overwriting", ok[0] and all(ok[1:]), f"{sum(map(bool, ok))}/5")
    sums = {"=2+2": "4", "2 + 2 =": "4", "100 m2 to sqft": "1,076.39 sq ft", "2 hours in minutes": "120 min",
            "1 gb to mb": "1,000 MB", "3 days in hours": "72 h"}
    got = {q: (app.answers(q) or [("",)])[0][0] for q in sums}
    check("calculator takes = and m2/m3; converts time and data sizes", got == sums,
          ", ".join(f"{q}->{v}" for q, v in got.items() if v != sums[q]))
    pf = [app.parse_filters(q)[:2] for q in ("pdf files", "my photos", "pdf -", "gold pdf", "pdf")]
    check("'pdf files' and 'my photos' list that kind; a lone minus is ignored; 'gold pdf' still searches gold",
          pf[0] == ("", {"pdf"}) and pf[1][0] == "" and pf[1][1] and pf[2][0] == "pdf" and pf[3] == ("gold", {"pdf"})
          and pf[4] == ("pdf", set()), str(pf))
    a, b = str(app.HOME / "x/a.pdf"), str(app.HOME / "y/b.pdf")
    rows = [{"path": a, "name": "a.pdf"}, {"path": str(app.HOME / "z.txt"), "name": "z.txt"}]
    moved, gone = app.follow_file(rows, a, b), app.follow_file(rows, a, None)
    rel = app.follow_file([{"path": "x/a.pdf"}], a, b, rel=True)
    check("lists follow a renamed, moved or binned file",
          moved[0] == {"path": b, "name": "b.pdf"} and len(gone) == 1 and rel == [{"path": "y/b.pdf"}])
    src = (HERE / "app.py").read_text()
    check("cmd-Backspace still edits the search text on a file row",
          'self.file_row(row):\n            self.trash(row)' not in src and '"trash", "\\x08"' not in src)
    sh = (HERE / "tools/refresh.sh").read_text()
    check("a paused leftover wrapper cannot block refreshes", "kill -CONT" in sh
          and sh.index("kill -CONT") < sh.index("already running"))


def test_rerank(real):
    """The second pass: re-orders documents among the top results, leaves photos in their places, keeps a code
    query's order, and changes nothing on a bad answer. Then one real round trip through the model on a test port."""
    import search
    from common import rerank_cfg
    doc = lambda n: {"path": f"d/{n}.txt", "snippet": n, "chunk": n, "versions": []}
    photo = {"path": "p/beach.jpg", "snippet": "photo", "kind": "image", "versions": []}
    hits = [doc(1), photo, doc(2), doc(3)]
    new = search.reranked(hits, [0, 2, 3], [0.1, 0.2, 9.0])
    # The blend is gentle: 3rd by today and 1st by the model ties with 1st by today, so only 2 and 3 swap.
    check("rerank moves documents and leaves photos in place",
          new[1] is photo and [h["chunk"] for h in (new[0], new[2], new[3])] == [1, 3, 2])
    check("rerank changes nothing on a bad answer", search.reranked(hits, [0, 2, 3], None) == hits
          and search.reranked(hits, [0, 2, 3], [1.0]) == hits)
    check("a code or figure query keeps its exact-match order",
          not search.wants_rerank("AE-1167 2,829,169", hits) and search.wants_rerank("lease bond refund", hits))
    rc = rerank_cfg(real)
    if rc is None:
        print("skip  reranker round trip (model not on this Mac or switched off)")
        return
    rc["model"]["port"] = 18094
    stop_port(18094)
    rr = Embedder(rc, log=lambda m: None)
    try:
        s = rr.rerank("recipe for butter chicken", ["Garden hose and seeds budget.",
                                                    "Butter chicken with garam masala and cream."])
        check("the reranker answers and ranks the right text first", bool(s) and s[1] > s[0])
    finally:
        rr.close()
        stop_port(18094)


def fresh_config(tmp, real):
    cfg = copy.deepcopy(real)
    cfg.update(roots=[str(tmp / "docs")], index=str(tmp / "ls" / "index.sqlite"), log_dir=str(tmp / "ls" / "logs"))
    cfg["model"]["port"] = 18090
    cfg["skip"].update(under=[], types=[])
    cfg["media"].update(roots=[], keep_dirs=[])
    cfg["code"] = {"roots": [str(tmp / "code")], "index": str(tmp / "ls" / "code_index.sqlite")}
    cfg["app"]["port"] = 18091
    return cfg


def toml(cfg):
    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        if isinstance(v, dict):   # an inline table, e.g. [skip] same_as
            return "{" + ", ".join(f"{json.dumps(k)} = {val(x)}" for k, x in v.items()) + "}"
        return json.dumps(v)
    top = [f"{k} = {val(v)}" for k, v in cfg.items() if not isinstance(v, dict)]
    secs = [f"\n[{k}]\n" + "\n".join(f"{kk} = {val(vv)}" for kk, vv in v.items())
            for k, v in cfg.items() if isinstance(v, dict)]
    return "\n".join(top) + "\n" + "\n".join(secs) + "\n"


def real_counts(real):
    """{index: (files, files from a self-test)} for this Mac's own indexes. Their timestamps say nothing (the running
    app updates them during a test); a test writing into them would add its own files or wipe the rest."""
    import codesearch
    out = {}
    for p in (Path(real["index"]).expanduser(), codesearch.index_path(real)):
        if p.exists():
            db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            out[p.name] = db.execute("SELECT count(*), sum(path LIKE '%localsearch-selftest-%') FROM files").fetchone()
            db.close()
    return out


def test_fresh_install(real):
    tmp = Path(tempfile.mkdtemp(prefix="localsearch-selftest-"))
    try:
        (tmp / "ls" / "tools").mkdir(parents=True)
        (tmp / "docs").mkdir()
        (tmp / "code").mkdir()
        for f in SOURCES:
            shutil.copy(HERE / f, tmp / "ls" / f)
        shutil.copy(HERE / "config.example.toml", tmp / "ls" / "config.example.toml")
        for name, text in DOCS.items():
            (tmp / "docs" / name).write_text(text + "\n")
        for i, text in enumerate(FILLER):
            (tmp / "docs" / f"note{i}.txt").write_text(text + ".\n")
        (tmp / "code" / "music.py").write_text('def make_playlist(songs):\n    """Shuffle songs."""\n'
                                               '    return sorted(songs)\n')

        # A fresh copy has no config.toml, state/ or logs/: the first load writes config.toml from the example.
        r = subprocess.run([sys.executable, "-c", "import common; common.load_config()"], cwd=tmp / "ls",
                           capture_output=True, text=True)
        check("fresh copy creates config.toml from the example",
              r.returncode == 0 and (tmp / "ls" / "config.toml").exists(), r.stderr.strip()[-200:])
        cfg = fresh_config(tmp, real)
        (tmp / "ls" / "config.toml").write_text(toml(cfg))

        from watch import on_battery   # on battery the indexer would wait for mains power and time out the test
        r = subprocess.run([sys.executable, "indexer.py"] + (["--no-wait"] if on_battery() else []), cwd=tmp / "ls",
                           capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        check("first index runs without errors", r.returncode == 0 and "Traceback" not in out,
              out.strip().splitlines()[-1][-200:] if r.returncode else "")
        wal = Path(cfg["index"] + "-wal")
        check("the -wal file is cut back after a run", not wal.exists() or wal.stat().st_size == 0,
              f"{wal.stat().st_size if wal.exists() else 0} bytes")
        db = sqlite3.connect(cfg["index"])
        files, chunks, vecs = db.execute("SELECT (SELECT count(*) FROM files), (SELECT count(*) FROM chunks), "
                                         "(SELECT count(*) FROM vectors)").fetchone()
        n = len(DOCS) + len(FILLER)
        check("every made-up file indexed", files == n, f"{files}/{n} files, {chunks} chunks")
        from watch import on_battery
        if on_battery():
            check("meaning vectors (skipped: on battery, embedding waits for mains)", True)
        else:
            check("meaning vectors made", vecs == chunks, f"{vecs}/{chunks}")

        import search
        s = search.Searcher(cfg, embedder=None)
        emb = Embedder(cfg, log=lambda m: None)
        try:
            if vecs:
                hits = 0
                for q, want in MEANING:
                    top = [h["path"].rsplit("/", 1)[-1] for h in s.search(q, "hybrid", 5, qvec=emb.query(q))[:3]]
                    hits += want in top
                check("meaning search finds the right file in the top 3", hits == len(MEANING),
                      f"{hits}/{len(MEANING)}")
            top = s.search("hose", "hybrid", 5, qvec=None)
            check("keyword search", bool(top) and top[0]["path"].endswith("garden.txt"))
            search.RESPELL_MIN = 1   # each made-up word is in one chunk
            fixed, fixes = s.respell("quaterly packign list")
            kept, none = s.respell("quarterly packing list AE-1167 sunst")
            search.RESPELL_MIN = 3
            check("a typo is searched as the word in your files; right words, codes and short words stay",
                  fixed == "quarterly packing list" and kept == "quarterly packing list AE-1167 sunst" and not none,
                  f"{fixes}")
            top = s.search(fixed, "hybrid", 5, qvec=None)
            check("a respelled search finds the file", bool(top) and top[0]["path"].endswith(("garden.txt", "trip.md")))
        finally:
            emb.close()

        r = subprocess.run([sys.executable, "codesearch.py", "--index"], cwd=tmp / "ls", capture_output=True,
                           text=True, timeout=600)
        check("code index builds", r.returncode == 0 and "Traceback" not in r.stdout + r.stderr)
        import codesearch
        found = codesearch.CodeSearcher(cfg).search("playlist", None, 5)
        check("code search finds a function by name", any(h["name"] == "make_playlist" for h in found))

        # Before the first index, and while the first index runs on battery: no vectors at all.
        empty = copy.deepcopy(cfg)
        empty["index"] = str(tmp / "empty.sqlite")
        e = search.Searcher(empty, embedder=None)
        ok = all(e.search("anything", m, 10, qvec=None) == [] for m in ("hybrid", "keyword"))
        check("an empty index searches without crashing", ok)
        novec = copy.deepcopy(cfg)
        novec["index"] = str(tmp / "novec.sqlite")
        dst = sqlite3.connect(novec["index"])
        db.backup(dst)
        dst.execute("DELETE FROM vectors")
        dst.commit()
        top = search.Searcher(novec, embedder=None).search("hose", "hybrid", 5, qvec=None)
        check("an index without vectors yet still finds words", bool(top) and top[0]["path"].endswith("garden.txt"))
        test_handoff(tmp, cfg, db)
    finally:
        for port in (18090, 18091):
            stop_port(port)
        shutil.rmtree(tmp, ignore_errors=True)


def test_handoff(tmp, cfg, db):
    """2026-10-30: a refresh embedding for hours held the lock, so every quick update of a changed folder was turned
    away (exit 75) and retried every 2 min until it finished. Now a refresh lets a waiting quick update go first."""
    ls = tmp / "ls"
    holder = subprocess.Popen([sys.executable, "-c",
                               "import fcntl, time, indexer\n"
                               "indexer.LOCK = open('state/indexer.lock', 'w')\n"
                               "fcntl.flock(indexer.LOCK, fcntl.LOCK_EX)\n"
                               "print('held', flush=True)\n"
                               "end = time.time() + 120\n"
                               "while time.time() < end:\n"
                               "    indexer.yield_lock()\n"
                               "    time.sleep(0.5)\n"],
                              cwd=ls, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        holder.stdout.readline()   # "held": the stand-in refresh owns the lock
        (tmp / "docs" / "quick.txt").write_text("Receipt for the zanzibarquokka kayak hire.\n")
        t = time.time()
        r = subprocess.run([sys.executable, "indexer.py", "--no-wait", "--no-embed", "--paths", str(tmp / "docs")],
                           cwd=ls, capture_output=True, text=True, timeout=180)
        secs = time.time() - t
        import search
        top = search.Searcher(cfg, embedder=None).search("zanzibarquokka", "hybrid", 5, qvec=None)
        check("a quick update gets in while a refresh holds the lock",
              r.returncode == 0 and secs < 60 and bool(top) and top[0]["path"].endswith("quick.txt"),
              f"exit {r.returncode}, {secs:.0f}s")
        check("the refresh takes the lock back afterwards", holder.poll() is None)
    finally:
        holder.kill()
        holder.wait()

    sys.path.insert(0, str(ls))
    import indexer
    done = db.execute("SELECT c.id, c.text, f.path FROM chunks c JOIN files f ON f.id = c.file "
                      "JOIN vectors v ON v.chunk = c.id LIMIT 2").fetchall()
    gone = [(10**9, "no such chunk", "x.txt")]
    check("a block drops chunks embedded or removed meanwhile", indexer.still_unembedded(db, done + gone) == [])
    bad = tmp / "docs" / "raster.tif"
    bad.write_bytes(b"II*\x00" + b"\x07" * 4096)
    _, note = indexer.media_file(None, str(bad), "tif")
    check("an image macOS cannot decode is not retried every run", (note or "").startswith("undecodable"),
          (note or "")[:40])
    big = tmp / "big.log"
    big.write_bytes(b"x" * 2048)
    import common
    common.rotate(big, max_bytes=1024)
    check("a log past its size limit is rotated", not big.exists() and (tmp / "big.log.1").exists())


def test_orphan(real):
    """A llama-server whose starter died (parent launchd) is replaced by the next Embedder, then stopped by it."""
    cfg = copy.deepcopy(real)
    cfg["model"]["port"] = 18092
    m = cfg["model"]
    cmd = " ".join(f"'{a}'" for a in [m["server"], "-m", str(Path(m["gguf"]).expanduser()), *m["args"],
                                         "--port", "18092"])
    stop_port(18092)   # a leftover from an earlier, interrupted self-test
    subprocess.run(["/bin/sh", "-c", cmd + " >/dev/null 2>&1 &"])   # the shell exits: the server is orphaned
    try:
        probe = Embedder.__new__(Embedder)
        probe.url = "http://127.0.0.1:18092"
        for _ in range(240):   # until it has loaded and answers, like one left by a killed run
            if orphan_server(18092) and probe._healthy():
                break
            time.sleep(0.5)
        had = orphan_server(18092) is not None
        emb = Embedder(cfg, log=lambda m: None)
        owns = emb.proc is not None
        emb.close()
        time.sleep(1)
        check("a left-behind server is replaced and stopped", had and owns and orphan_server(18092) is None)
    finally:
        stop_port(18092)


def stop_port(port):
    """Stop whatever listens on a test port, and wait until it has gone."""
    out = subprocess.run(["/usr/sbin/lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True,
                         text=True).stdout.split()
    for pid in map(int, out):
        os.kill(pid, 15)
        for _ in range(80):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.25)


def test_live(real):
    r = subprocess.run(["pgrep", "-f", "localsearch/app.py$"], capture_output=True)
    check("the app is running", r.returncode == 0)
    log = HERE / "logs" / "app.log"
    tb = log.read_text(errors="replace").count("Traceback") if log.exists() else 0
    check("app.log has no errors", tb == 0, f"{tb} tracebacks")
    from common import RERANK
    rr_port = {**RERANK, **real.get("rerank", {})}["port"]
    left = [p for p in (real["model"]["port"], real["app"]["port"], rr_port) if orphan_server(p)]
    check("no left-behind servers on the app's, indexer's or reranker's port", not left, str(left))
    queries = HERE / "evals" / "queries.json"
    if not queries.exists():
        print("skip  search quality (no evals/queries.json on this Mac)")
        return
    from eval import first_rank, metrics
    import search
    cfg = copy.deepcopy(real)
    cfg["model"]["port"] = 18093
    from common import rerank_cfg
    rc = rerank_cfg(real)
    emb, rr = Embedder(cfg, log=lambda m: None), None
    try:
        if rc is not None:   # score what the window shows: the first order, then the reranker's second pass
            rc["model"]["port"] = 18094
            rr = Embedder(rc, log=lambda m: None)
        s = search.Searcher(real, embedder=emb)
        qs = [x for x in json.load(open(queries))["queries"] if x["tag"] not in ("image", "video")]
        ranks = []
        for x in qs:
            hits = s.search(x["q"], "hybrid", 20, qvec=emb.query(x["q"]))
            if rr is not None and search.wants_rerank(x["q"], hits):
                slots, texts = search.rerank_docs(s.db, hits)
                hits = search.reranked(hits, slots, rr.rerank(x["q"], texts))
            ranks.append(first_rank(hits, set(x["expect"]) | set(x.get("expect_extra", []))))
    finally:
        emb.close()
        if rr is not None:
            rr.close()
    m = metrics(ranks)
    slow = []
    for x in json.load(open(queries))["queries"]:
        t = time.time()
        s.words(x["q"])
        slow.append(time.time() - t)
    # 2026-10-30: a word found in 1.2 M CSV chunks made one keyword search take 90 s; the fix brought the worst to ~1 s
    check("the slowest keyword search stays quick", max(slow) < 8, f"worst {max(slow):.1f}s of {len(slow)}")
    # 2026-10-22 baseline: top-1 72%, top-5 97%, MRR 0.831 on 58 text queries. By 2026-10-30 the index had grown and
    # the first order alone gave top-5 95%, MRR 0.778; with the reranker top-1 72%, top-5 97%, MRR 0.814.
    check("search quality holds", m["r5"] >= 0.95 and m["mrr"] >= 0.80,
          f"top1 {m['r1']:.0%}, top5 {m['r5']:.0%}, mrr {m['mrr']:.3f} on {len(qs)} queries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also check this Mac's running app and search quality")
    args = ap.parse_args()
    real = load_config()
    t = time.time()
    test_compile()
    before = real_counts(real)
    guarded("spoken dates", test_spoken_dates)
    guarded("app typos", test_app_typos)
    guarded("pause targets", test_pause_targets)
    guarded("cloud folders", lambda: test_cloud_root(real))
    guarded("big sheets", test_big_sheets)
    guarded("reranker", lambda: test_rerank(real))
    guarded("fresh install", lambda: test_fresh_install(real))
    after = real_counts(real)
    ok = all((after[k][1] or 0) == 0 and after[k][0] >= 0.9 * n for k, (n, _) in before.items() if k in after)
    check("this Mac's own indexes untouched by the tests", ok and before.keys() == after.keys(),
          ", ".join(f"{k} {before[k][0]} -> {after.get(k, (0,))[0]} files" for k in before))
    guarded("left-behind server", lambda: test_orphan(real))
    if args.live:
        guarded("live checks", lambda: test_live(real))
    bad = results.count(False)
    print(f"\n{len(results) - bad}/{len(results)} passed in {time.time() - t:.0f}s" + ("" if bad else ": all good"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
