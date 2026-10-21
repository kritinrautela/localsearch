"""Shared pieces: config, the SQLite index, and the llama-server embedding client."""

import base64
import ipaddress
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import threading
import time
import tomllib
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HOME = Path.home()


# --- network guard: this process may only talk to loopback (llama-server on 127.0.0.1) -------------------------------

def _loopback(address):
    if not isinstance(address, tuple):  # AF_UNIX path
        return True
    host = address[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo


def _guarded_connect(self, address):
    if not _loopback(address):
        raise PermissionError(f"localsearch: blocked non-loopback connection to {address!r}")
    return _connect(self, address)


def _guarded_connect_ex(self, address):
    if not _loopback(address):
        raise PermissionError(f"localsearch: blocked non-loopback connection to {address!r}")
    return _connect_ex(self, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    # Refuse to resolve anything but loopback names, so no DNS query ever leaves the Mac either.
    if host not in (None, "localhost") and not _loopback((host if isinstance(host, str) else host.decode(), 0)):
        raise PermissionError(f"localsearch: blocked name lookup for {host!r}")
    return _getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.getaddrinfo = _guarded_getaddrinfo


# --- text helpers ----------------------------------------------------------------------------------------------------

def fold(text):
    """Lower case without accents: "Orçamento" -> "orcamento"."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()


VERSION_WORDS = (r"copy(\s*\d+)?|backup|bak|old|new|final|draft|latest|clean|edited|updated|revised|unsigned|signed"
                 r"|blank|beautified|v\d+([._]\d+)*[a-z]?|rev\s*\d*|\d{4}[-_.]?\d{2}[-_.]?\d{2}|(19|20)\d{2}"
                 r"|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s_-]*((19|20)?\d{2})?")
GENERIC_STEMS = {"readme", "index", "notes", "note", "log", "data", "output", "outputs", "report", "summary",
                 "results", "config", "main", "untitled", "document", "table", "test", "todo", "changelog",
                 "license", "manifest", "metadata", "requirements", "claude", "agents", "memory"}


def version_key(rel):
    """Files that are versions of one document share a key: the name without its extension and without trailing
    version marks (_v33, (1), copy, BACKUP, _DRAFT, a date, .backup), letters and digits only. A generic or one-word
    name ("README", "Configuration") only groups within its own folder."""
    folder, _, name = rel.rpartition("/")
    stem = fold(name.rsplit(".", 1)[0] if "." in name else name)
    prev = None
    while prev != stem:
        prev = stem
        stem = re.sub(r"\s*\([^)]*\)$", "", stem)                           # (1), (Final), (Revised, Unsigned)
        stem = re.sub(rf"[\s._-]+({VERSION_WORDS})$", "", stem)             # _v33, copy 2, BACKUP, _2026-08-12
        stem = re.sub(r"[\s_]\d{1,2}$", "", stem)                           # _1, " 2"
    key = re.sub(r"[^0-9a-z]+", "", stem)
    one_word = not re.search(r"[0-9a-z][^0-9a-z]+[0-9a-z]", stem)
    if len(key) < 6 or key in GENERIC_STEMS or one_word:
        key = folder + "/" + key
    return key


def expand(p):
    return Path(os.path.expanduser(p))


def load_config(path=HERE / "config.toml"):
    path = Path(path)
    if path == HERE / "config.toml" and not path.exists():   # a fresh copy: start from the shipped defaults
        path.write_bytes((HERE / "config.example.toml").read_bytes())
    with open(path, "rb") as f:
        return tomllib.load(f)


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,      -- relative to the home folder
    ext TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    chunks INTEGER NOT NULL DEFAULT 0,
    chars INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,           -- ok | empty | error
    error TEXT,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file INTEGER NOT NULL REFERENCES files(id),
    ord INTEGER NOT NULL,
    page INTEGER,                   -- PDF page, slide or sheet number (1-based)
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file);
CREATE TABLE IF NOT EXISTS vectors (
    chunk INTEGER PRIMARY KEY REFERENCES chunks(id),
    vec BLOB NOT NULL               -- 768 x float16, L2-normalised
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(name, body, tokenize='porter unicode61');
-- Exact substrings (codes, IDs, figures as written): trigram index over chunks.text, no second copy of the text.
CREATE VIRTUAL TABLE IF NOT EXISTS tri USING fts5(text, content='chunks', content_rowid='id',
                                                  tokenize='trigram remove_diacritics 1');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
-- Phase 4: photo, video-frame and soundtrack vectors, in the same 768-d space as the text vectors.
CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY,
    file INTEGER NOT NULL REFERENCES files(id),
    kind TEXT NOT NULL,             -- image | frame | audio
    t REAL,                         -- seconds into the video (frame time, or audio window start)
    t_end REAL,                     -- audio window end
    vec BLOB NOT NULL               -- 768 x float16, L2-normalised
);
CREATE INDEX IF NOT EXISTS media_file ON media(file);
CREATE TABLE IF NOT EXISTS media_done (file INTEGER PRIMARY KEY, frames INTEGER, windows INTEGER, seconds REAL,
                                       error TEXT);
CREATE TABLE IF NOT EXISTS ocr_done (file INTEGER PRIMARY KEY, engine TEXT, pages INTEGER, chars INTEGER,
                                     seconds REAL, error TEXT);
"""


def connect(cfg):
    path = expand(cfg["index"])
    db = sqlite3.connect(path, check_same_thread=False, timeout=60)  # wait out a short lock instead of failing
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    if "grp" not in {r[1] for r in db.execute("PRAGMA table_info(files)")}:
        db.execute("ALTER TABLE files ADD COLUMN grp TEXT")   # version group (common.version_key)
        db.commit()
    return db


# No proxy handler: requests go straight to 127.0.0.1 even if HTTP(S)_PROXY or a system proxy is set.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def rss_kb(pid):
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
        return int(out.strip() or 0)
    except ValueError:
        return 0


ACTIVE = set()   # Embedders whose llama-server this process started; stopped on SIGTERM by indexer.py


def orphan_server(port):
    """pid of a llama-server listening on port whose starter has died (its parent is now launchd), or None."""
    out = subprocess.run(["/usr/sbin/lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout.split()
    for pid in out:
        info = subprocess.run(["ps", "-o", "ppid=,comm=", "-p", pid], capture_output=True, text=True).stdout.split()
        if len(info) == 2 and info[0] == "1" and info[1].endswith("llama-server"):
            return int(pid)
    return None


class Embedder:
    """Talks to llama-server; starts it if nothing is listening on the port, and stops it on close()."""

    def __init__(self, cfg, log=print, media=False):
        self.cfg = cfg["model"]
        self.media = media              # start the server with the image/audio projector (mmproj)
        self.marker = None
        self.url = f"http://127.0.0.1:{self.cfg['port']}"   # the only network endpoint this code uses
        self.log = log
        self.proc = None
        self.peak_server_kb = 0
        self._stop = threading.Event()
        if self._healthy() and (pid := orphan_server(self.cfg["port"])):
            # Left behind by a run killed before it could stop it. Reusing it would keep it alive for good (and it
            # may lack --mmproj): replace it with a server this process owns and stops.
            os.kill(pid, signal.SIGTERM)
            for _ in range(40):
                if not self._healthy():
                    break
                time.sleep(0.25)
        if media:
            # A text-only server another run is using (code search embedding): wait for it to finish, then start one
            # with the projector.
            for _ in range(180):
                if not self._healthy() or self._vision():
                    break
                time.sleep(5)
        if not self._healthy():
            self._start(cfg)
        if media:
            props = json.load(OPENER.open(self.url + "/props", timeout=10))
            if not props.get("modalities", {}).get("vision"):
                raise RuntimeError(f"llama-server on port {self.cfg['port']} has no --mmproj; stop it and rerun")
            self.marker = props["media_marker"]   # random per server start

    def _vision(self):
        try:
            return bool(json.load(OPENER.open(self.url + "/props", timeout=10)).get("modalities", {}).get("vision"))
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _healthy(self):
        try:
            with OPENER.open(self.url + "/health", timeout=2) as r:
                return b"ok" in r.read()
        except (urllib.error.URLError, OSError):
            return False

    def _start(self, cfg):
        gguf = expand(self.cfg["gguf"])
        cmd = [self.cfg["server"], "-m", str(gguf), *self.cfg["args"], "--port", str(self.cfg["port"])]
        if self.media:
            cmd += ["--mmproj", str(expand(self.cfg["mmproj"]))]
        logf = open(expand(cfg["log_dir"]) / "llama-server.log", "a")
        self.log("starting llama-server: " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        ACTIVE.add(self)
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited; see logs/llama-server.log")
            if self._healthy():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("llama-server did not become healthy in 120 s")
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        while not self._stop.is_set() and self.proc and self.proc.poll() is None:
            self.peak_server_kb = max(self.peak_server_kb, rss_kb(self.proc.pid))
            self._stop.wait(2)

    def close(self):
        ACTIVE.discard(self)
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.peak_server_kb = max(self.peak_server_kb, rss_kb(self.proc.pid))
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _post(self, inputs):
        body = json.dumps({"input": inputs}).encode()
        req = urllib.request.Request(self.url + "/v1/embeddings", body, {"Content-Type": "application/json"})
        with OPENER.open(req, timeout=600) as r:
            data = json.load(r)["data"]
        data.sort(key=lambda d: d["index"])
        return np.asarray([d["embedding"] for d in data], dtype=np.float32)

    def embed(self, texts):
        """Vectors for texts, L2-normalised; a row is None when the server refused that text or returned junk."""
        try:
            vecs = self._post(texts)
            rows = list(vecs)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError):
            if len(texts) == 1:
                return [None]
            rows = []
            for t in texts:  # find the one that broke the batch
                rows.extend(self.embed([t]))
            return rows
        out = []
        for v in rows:
            if v is None or v.shape[0] != self.cfg["dim"] or not np.isfinite(v).all():
                out.append(None)
                continue
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else None)
        return out

    def embed_media(self, data):
        """Vector for one image (JPEG bytes) or sound clip (WAV bytes) through the projector; None if refused."""
        body = json.dumps({"content": {"prompt_string": self.marker,
                                       "multimodal_data": [base64.b64encode(data).decode()]}}).encode()
        req = urllib.request.Request(self.url + "/embeddings", body, {"Content-Type": "application/json"})
        try:
            with OPENER.open(req, timeout=600) as r:
                d = json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
            return None
        v = np.asarray((d[0] if isinstance(d, list) else d["data"][0])["embedding"], dtype=np.float32)
        v = v.reshape(-1, self.cfg["dim"]).mean(0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 and np.isfinite(v).all() else None

    def query(self, q):
        v = self.embed([self.cfg["query_prefix"].format(q=q)])[0]
        if v is None:
            raise RuntimeError("could not embed query")
        return v
