"""Search index.sqlite: hybrid (DigUp's scoring), RRF, or vector-only.

Photos, video frames and soundtrack windows (Phase 4) share the text vector space. Their cosines are put on the text
chunks' scale for the query (z against the text distribution), so a picture competes with documents on equal terms;
a video hit carries the time of its best frame or sound window.

Hybrid and RRF put exact matches of a code-like query (AE-1167, CD/KNG/RCCM/19-B-02266, 2,829,169 t) first, using the
trigram table. Results are collapsed by version group: copies and _v32/_v33/(1)/BACKUP variants of one document show
as one result, newest first, with the other versions listed under it.

    uv run python search.py "rental agreement for the flat"
    uv run python search.py "INV-2041" --mode rrf -k 10
    uv run python search.py "..." --mode vector --json
"""

import argparse
import json
import re
import time

import numpy as np

from common import Embedder, connect, fold, load_config

# DigUp's stopword list (English part), Searcher.swift
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or", "my", "me", "that", "this",
    "these", "those", "from", "about", "where", "what", "which", "who", "when", "is", "was", "are", "were", "be",
    "it", "its", "some", "any", "find", "show", "file", "files", "photo", "photos", "picture", "image", "images",
    "screenshot", "screenshots", "video", "videos", "pdf", "document", "documents", "recording",
}
Z_GATE = 2.0
WORD_WEIGHT = 2.0
RRF_K = 60
EXACT_MAX_FILES = 30   # a short plain phrase counts as a name only if it is this rare
BM25_TOP = 10          # the best keyword matches earn a rank bonus in hybrid...
BM25_WEIGHT = 2.0      # ...of BM25_WEIGHT / rank (doubled for lookups); tuned on evals/queries.json, Phase 4
MEDIA_Z = 4.0          # a photo, frame or soundtrack this far above the text scale is a clearly visual match...
MEDIA_BONUS = 2.0      # ...and earns this in hybrid, standing in for the word bonus it can never get; tuned, Phase 4
MEDIA_ONLY_EXT = {"jpg", "jpeg", "png", "heic", "webp", "gif", "tif", "tiff", "bmp", "mp4", "mov", "m4v", "avi", "mkv",
                  "webm", "mts", "3gp", "mp3", "m4a", "wav", "aac", "flac", "aiff", "aif", "opus", "ogg", "caf",
                  "dng", "cr2", "cr3", "nef", "arw", "raf", "orf", "rw2"}
MOMENTS = 3            # a video lists up to this many more matching moments...
MOMENT_GAP_S = 10.0    # ...at least this far apart...
MOMENT_DROP = 1.5      # ...each within this much z of the video's best moment


def code_tokens(q):
    """Tokens that look like codes, IDs or figures: they hold a digit, or join letters with / - _ , ."""
    out = []
    for t in q.split():
        t = t.strip("\"'()[]{}:;?!")
        if sum(c.isalnum() for c in t) >= 3 and (any(c.isdigit() for c in t) or re.search(r"\w[/\-_,.]\w", t)):
            out.append(t)
    return out


def terms(q):
    seen, out = set(), []
    for w in re.split(r"[^\w]+|_", fold(q)):
        if len(w) >= 2 and w not in STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def is_lookup(ts):
    """Codes and long numbers: weigh words double and meaning half (DigUp's isLookup)."""
    return bool(ts) and all(any(c.isdigit() for c in t) and (any(c.isalpha() for c in t) or len(t) >= 4) for t in ts)


def clock(t):
    t = int(t or 0)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


class Searcher:
    def __init__(self, cfg=None, embedder=None):
        self.cfg = cfg or load_config()
        self.db = connect(self.cfg)
        self.emb = embedder
        t = time.time()
        rows = self.db.execute("SELECT v.chunk, c.file, v.vec FROM vectors v JOIN chunks c ON c.id = v.chunk "
                               "ORDER BY v.chunk").fetchall()
        self.row_chunk = np.array([r[0] for r in rows], dtype=np.int64)
        self.row_file = np.array([r[1] for r in rows], dtype=np.int64)
        self.matrix = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float16).reshape(
            len(rows), self.cfg["model"]["dim"])   # an explicit width: a fresh index has no vectors yet
        rows = self.db.execute("SELECT file, kind, t, t_end, vec FROM media ORDER BY id").fetchall()
        audio_w = self.cfg.get("media", {}).get("audio_weight", 0.5)
        self.media_file = np.array([r[0] for r in rows], dtype=np.int64)
        self.media_info = [(r[1], r[2], r[3]) for r in rows]
        self.media_weight = np.array([audio_w if r[1] == "audio" else 1.0 for r in rows], dtype=np.float32)
        self.media = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float16).reshape(
            len(rows), self.cfg["model"]["dim"])
        self.paths, self.grp, self.mtime = {}, {}, {}
        for f, path, grp, mtime in self.db.execute("SELECT id, path, grp, mtime FROM files"):
            self.paths[f], self.grp[f], self.mtime[f] = path, grp or path, mtime
        self.load_s = time.time() - t
        self.last_mz = None

    def embedder(self):
        if self.emb is None:
            self.emb = Embedder(self.cfg, log=lambda m: None)
        return self.emb

    # --- the two signals ---------------------------------------------------------------------------------------------

    @staticmethod
    def cosines(matrix, qvec):
        out = np.empty(len(matrix), dtype=np.float32)
        q = qvec.astype(np.float32)
        for i in range(0, len(matrix), 65536):
            out[i:i + 65536] = matrix[i:i + 65536].astype(np.float32) @ q
        return out

    @staticmethod
    def best_per_file(scores, row_file):
        """{file: (score, row)} keeping each file's best row."""
        order = np.argsort(-scores, kind="stable")
        best = {}
        for row in order:
            f = int(row_file[row])
            if f not in best:
                best[f] = (float(scores[row]), int(row))
        return best

    def meaning(self, qvec):
        """{file: (z, hit-args)}: each file's best text chunk or media vector, as z on the text chunks' scale."""
        scores = self.cosines(self.matrix, qvec) if len(self.matrix) else np.zeros(0, np.float32)
        mscores = self.cosines(self.media, qvec) if len(self.media) else np.zeros(0, np.float32)
        base = scores if len(scores) > 1 else mscores
        if not len(base):
            return {}
        mean, dev = float(base.mean()), max(float(base.std()), 0.02)
        out = {f: ((c - mean) / dev, {"row": row}) for f, (c, row) in self.best_per_file(scores, self.row_file).items()}
        if len(mscores):
            mz = (mscores - mean) / dev
            mz = np.where(mz > 0, mz * self.media_weight, mz)
            self.last_mz = mz
            for f, (zz, row) in self.best_per_file(mz, self.media_file).items():
                if f not in out or zz > out[f][0]:
                    out[f] = (zz, {"media": row})
        return out

    def indexed(self, pattern):
        return self.db.execute("SELECT 1 FROM fts WHERE fts MATCH ? LIMIT 1", (pattern,)).fetchone() is not None

    def words(self, q):
        """FTS5 matches: ([(file, chunk, snippet)] best-first one per file, {file: coverage}, patterns)."""
        ts = terms(q)
        patterns = ['"' + t.replace('"', "") + '"' for t in ts]
        for i, t in enumerate(ts):  # a word in no file as typed matches as a prefix (DigUp)
            if len(t) >= 3 and not self.indexed(patterns[i]):
                patterns[i] += "*"
        if not patterns:
            return [], {}, ts
        hits, seen = [], set()
        for f, cid, snip in self.db.execute(
                "SELECT c.file, c.id, snippet(fts, 1, '«', '»', '…', 12) FROM fts JOIN chunks c ON c.id = fts.rowid "
                "WHERE fts MATCH ? ORDER BY bm25(fts, 3.0, 1.0) LIMIT 1000", (" OR ".join(patterns),)):
            if f not in seen and len(hits) < 100:
                seen.add(f)
                hits.append((f, cid, snip))
        coverage = {}
        if hits:
            files = ",".join(str(h[0]) for h in hits)
            for p in patterns:
                for (f,) in self.db.execute(
                        f"SELECT DISTINCT c.file FROM fts JOIN chunks c ON c.id = fts.rowid "
                        f"WHERE fts MATCH ? AND c.file IN ({files})", (p,)):
                    coverage[f] = coverage.get(f, 0) + 1 / len(patterns)
        return hits, coverage, ts

    def phrase_files(self, phrase, limit=2000):
        """{file: (chunk, snippet, occurrences)} for chunks holding `phrase` as written (case and accents ignored), not
        as part of a longer code: "AE-1167" does not match "AE-11670"."""
        if len(phrase) < 3:
            return {}
        pat = re.compile(r"(?<![0-9a-z])" + re.escape(fold(phrase)) + r"(?![0-9a-z])")
        out = {}
        for cid, f, text in self.db.execute(
                "SELECT c.id, c.file, c.text FROM tri JOIN chunks c ON c.id = tri.rowid WHERE tri MATCH ? LIMIT ?",
                ('"' + phrase.replace('"', '""') + '"', limit)):
            folded = fold(text)
            found = list(pat.finditer(folded))
            if not found:
                continue
            prev = out.get(f)
            n = len(found) + (prev[2] if prev else 0)
            if prev:
                out[f] = (prev[0], prev[1], n)
            else:
                a, b = found[0].span()  # fold() keeps length for Latin text, so offsets line up with the original
                out[f] = (cid, ("…" if a > 90 else "") + text[max(0, a - 90):a] + "«" + text[a:b] + "»"
                          + text[b:b + 120] + "…", n)
        return out

    def exact(self, q):
        """Files that hold the whole query as written; failing that, every code-like token in it. Applies to code-like
        queries, and to short phrases rare enough to be a name ("Kifica Benfica"), never to ordinary questions."""
        q = q.strip()
        codes = code_tokens(q)
        words = q.split()
        if not codes and not (2 <= len(words) <= 5):
            return {}
        hits = self.phrase_files(q)
        if not codes and len(hits) > EXACT_MAX_FILES:
            return {}
        if hits or len(codes) < 1 or codes == [q]:
            return hits
        common = None
        for c in codes:
            h = self.phrase_files(c)
            common = h if common is None else {f: v for f, v in common.items() if f in h}
        return common or {}

    # --- versions ----------------------------------------------------------------------------------------------------

    def collapse(self, ranked, k, tier=frozenset()):
        """ranked: [(file, score, hit-args)] best first. One result per version group, at the position of its best
        member; the newest matching member (by mtime) in that member's tier leads, the rest are listed as versions."""
        groups, order = {}, []
        for item in ranked:
            g = self.grp[item[0]]
            if g not in groups:
                groups[g] = []
                order.append(g)
            groups[g].append(item)
        out = []
        for g in order[:k]:
            members = groups[g]
            top_tier = members[0][0] in tier
            lead = max((m for m in members if (m[0] in tier) == top_tier), key=lambda m: self.mtime[m[0]])
            h = self.hit(lead[0], members[0][1], **lead[2])
            others = sorted((m[0] for m in members if m is not lead), key=lambda f: -self.mtime[f])
            h["versions"] = [self.paths[f] for f in others]
            out.append(h)
        return out

    # --- rankers -----------------------------------------------------------------------------------------------------

    def search(self, q, mode="hybrid", k=10, qvec=None, collapse=True, exts=None):
        """exts: only files with these extensions. Photos and videos alone are ranked by meaning without the hybrid
        gate, since a picture never holds the query's words."""
        if exts and exts <= MEDIA_ONLY_EXT and mode == "hybrid":
            mode = "vector"
        # Keyword matches read the live database, so a file indexed after this Searcher loaded can turn up: skip it
        # until the next reload knows its path and version group.
        ranked = [item for item in self.rank(q, mode, qvec) if item[0] in self.grp]
        if exts:
            ranked = [item for item in ranked if self.paths[item[0]].rsplit(".", 1)[-1].lower() in exts]
        tier = set()
        if mode != "vector":
            ex = {f: v for f, v in self.exact(q).items() if f in self.grp}
            if ex:
                tier = set(ex)
                base = {f: (score, args) for f, score, args in ranked}
                first = sorted(ex, key=lambda f: (-base.get(f, (0,))[0], -ex[f][2], f))
                ranked = ([(f, base.get(f, (0,))[0], {"word": (0, ex[f][0], ex[f][1]), "prefer_word": True})
                           for f in first] + [item for item in ranked if item[0] not in tier])
        if not collapse:
            return [self.hit(f, score, **args) for f, score, args in ranked[:k]]
        return self.collapse(ranked, k, tier)

    def similar(self, path, k=40, exts=None):
        """Files that mean the same as the file at path (relative to home), itself and its versions left out."""
        f = next((i for i, p in self.paths.items() if p == path), None)
        if f is None:
            return None
        rows = self.media[self.media_file == f] if len(self.media) else []
        if not len(rows):
            rows = self.matrix[self.row_file == f] if len(self.matrix) else []
        if not len(rows):
            return None
        vec = rows.astype(np.float32).mean(axis=0)
        vec /= max(float(np.linalg.norm(vec)), 1e-6)
        ranked = [item for item in self.rank("", "vector", vec) if self.grp[item[0]] != self.grp[f]]
        if exts:
            ranked = [item for item in ranked if self.paths[item[0]].rsplit(".", 1)[-1].lower() in exts]
        return self.collapse(ranked, k)

    def rank(self, q, mode, qvec=None):
        """[(file, score, hit-args)] best first, every candidate."""
        if qvec is None and (len(self.matrix) or len(self.media)):
            qvec = self.embedder().query(q)
        z = self.meaning(qvec) if qvec is not None else {}
        if mode == "vector":
            ranked = sorted(z, key=lambda f: (-z[f][0], f))[:2000]
            return [(f, z[f][0], dict(z[f][1])) for f in ranked]

        hits, coverage, ts = self.words(q)
        word_rank = {f: (i + 1, cid, snip) for i, (f, cid, snip) in enumerate(hits)}
        if mode == "rrf":
            vec_rank = {f: i + 1 for i, f in enumerate(sorted(z, key=lambda f: (-z[f][0], f))[:1000])}
            fused = {}
            for f, r in vec_rank.items():
                fused[f] = fused.get(f, 0) + 1 / (RRF_K + r)
            for f, (r, _, _) in word_rank.items():
                fused[f] = fused.get(f, 0) + 1 / (RRF_K + r)
            ranked = sorted(fused, key=lambda f: (-fused[f], f))[:2000]
            return [(f, fused[f], {**z.get(f, (0, {}))[1], "word": word_rank.get(f),
                                   "prefer_word": vec_rank.get(f, 10**9) > word_rank.get(f, (10**9,))[0]})
                    for f in ranked]

        # hybrid: DigUp's Searcher.search. z = cosine on this query's scale; files qualify by z >= 2, by holding at
        # least half the query's words, or by being a top-10 BM25 match; score = meaning z + a bonus growing with
        # word coverage + a bonus for BM25 rank. Without the last term a short brief that is the best keyword match
        # sinks under long reports that merely contain every query word somewhere.
        lookup = is_lookup(ts)
        mw, ww = (0.5, WORD_WEIGHT * 2) if lookup else (1.0, WORD_WEIGHT)
        bw = BM25_WEIGHT * (2 if lookup else 1)
        top = {f: i + 1 for i, (f, _, _) in enumerate(hits[:BM25_TOP])}
        cands = ({f for f, (zz, _) in z.items() if zz >= Z_GATE} | {f for f, c in coverage.items() if c >= 0.5}
                 | set(top))
        scored = {f: mw * max(z.get(f, (0, None))[0], 0) + ww * coverage.get(f, 0) ** 1.5
                  + (bw / top[f] if f in top else 0) for f in cands}
        for f in scored:
            zz, args = z.get(f, (0, {}))
            if "media" in args and zz >= MEDIA_Z:
                scored[f] += MEDIA_BONUS
        out = []
        for f in sorted(scored, key=lambda f: (-scored[f], f)):
            zz, args = z.get(f, (None, {}))
            use_meaning = zz is not None and (zz >= Z_GATE or f not in word_rank)
            out.append((f, scored[f], {**args, "word": word_rank.get(f), "prefer_word": not use_meaning}))
        return out

    def moments(self, f, best):
        """[{"t", "kind"}]: other places in the video that match nearly as well as its best one, best first."""
        mz = self.last_mz
        if mz is None or len(mz) != len(self.media_file):
            return []
        rows = np.nonzero(self.media_file == f)[0]
        if len(rows) < 2:
            return []
        top = float(mz[best])
        taken, out = [float(self.media_info[best][1] or 0)], []
        for row in rows[np.argsort(-mz[rows])]:
            kind, t, _ = self.media_info[row]
            if kind == "image" or mz[row] < max(Z_GATE, top - MOMENT_DROP):
                break
            t = float(t or 0)
            if all(abs(t - u) >= MOMENT_GAP_S for u in taken):
                taken.append(t)
                out.append({"t": t, "kind": kind})
                if len(out) >= MOMENTS:
                    break
        return out

    def hit(self, f, score, row=None, word=None, prefer_word=False, media=None):
        if media is not None and not (word and prefer_word):
            kind, t, t_end = self.media_info[media]
            label = {"image": "photo", "frame": f"video frame at {clock(t)}",
                     "audio": f"soundtrack {clock(t)}-{clock(t_end)}"}[kind]
            out = {"path": self.paths[f], "page": None, "score": round(score, 4), "snippet": label, "kind": kind,
                   "t": t, "versions": []}
            if kind != "image":
                out["moments"] = self.moments(f, media)
            return out
        if word and (prefer_word or row is None):
            cid, snippet = word[1], word[2]
        else:
            cid = int(self.row_chunk[row])
            snippet = None
        page, text = self.db.execute("SELECT page, text FROM chunks WHERE id = ?", (cid,)).fetchone()
        if snippet is None:
            snippet = text[:240]
        return {"path": self.paths[f], "page": page, "score": round(score, 4),
                "snippet": re.sub(r"\s+", " ", snippet).strip(), "versions": [], "chunk": cid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["hybrid", "rrf", "vector"], default="hybrid")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all-versions", action="store_true", help="do not collapse versions of one document")
    args = ap.parse_args()
    s = Searcher()
    t = time.time()
    try:
        hits = s.search(args.query, args.mode, args.k, collapse=not args.all_versions)
    finally:
        if s.emb:
            s.emb.close()
    took = time.time() - t
    if args.json:
        print(json.dumps({"query": args.query, "mode": args.mode, "seconds": round(took, 3), "hits": hits}, indent=1,
                         ensure_ascii=False))
        return
    for i, h in enumerate(hits, 1):
        page = f"  p.{h['page']}" if h["page"] else ""
        more = f"  +{len(h['versions'])} versions" if h["versions"] else ""
        print(f"{i:2}. {h['path']}{page}  [{h['score']}]{more}\n    {h['snippet'][:220]}")
    print(f"({args.mode}, {took:.2f}s, {len(s.matrix)} text and {len(s.media)} media vectors loaded in "
          f"{s.load_s:.2f}s)")


if __name__ == "__main__":
    main()
