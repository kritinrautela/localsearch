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

        r = subprocess.run([sys.executable, "indexer.py"], cwd=tmp / "ls", capture_output=True, text=True,
                           timeout=900)
        out = r.stdout + r.stderr
        check("first index runs without errors", r.returncode == 0 and "Traceback" not in out,
              out.strip().splitlines()[-1][-200:] if r.returncode else "")
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
    finally:
        for port in (18090, 18091):
            stop_port(port)
        shutil.rmtree(tmp, ignore_errors=True)


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
    left = [p for p in (real["model"]["port"], real["app"]["port"]) if orphan_server(p)]
    check("no left-behind servers on the app's or indexer's port", not left, str(left))
    queries = HERE / "evals" / "queries.json"
    if not queries.exists():
        print("skip  search quality (no evals/queries.json on this Mac)")
        return
    from eval import first_rank, metrics
    import search
    cfg = copy.deepcopy(real)
    cfg["model"]["port"] = 18093
    emb = Embedder(cfg, log=lambda m: None)
    try:
        s = search.Searcher(real, embedder=emb)
        qs = [x for x in json.load(open(queries))["queries"] if x["tag"] not in ("image", "video")]
        ranks = [first_rank(s.search(x["q"], "hybrid", 20, qvec=emb.query(x["q"])),
                            set(x["expect"]) | set(x.get("expect_extra", []))) for x in qs]
    finally:
        emb.close()
    m = metrics(ranks)
    # 2026-10-22 baseline: top-1 72%, top-5 97%, MRR 0.831 on 58 text queries
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
