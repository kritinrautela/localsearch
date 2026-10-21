"""Keep the index current without being asked: an FSEvents watch on the indexed folders runs a quick update of just the
folders that changed (indexer.py --paths ... --no-wait), and a full refresh runs once a night on mains power.

Runs inside the search window's process (app.py), on its main run loop. The indexer only reads source files, so
watching them changes nothing; the watcher never touches a file itself.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import FSEvents

import codesearch
from common import HERE, expand

QUIET_S = 6            # wait this long after the last change, so a file being saved or copied is complete
CODE_QUIET_S = 45      # code changes come in bursts while you edit; DigUp waits about a minute
MAX_WAIT_S = 60        # ...but never longer than this while changes keep coming
NIGHT_HOURS = range(2, 6)
STATE = HERE / "state" / "watch.json"
INDEXED_EXT = {"txt", "md", "markdown", "text", "rst", "docx", "doc", "rtf", "odt", "html", "htm", "xlsx", "csv",
               "pdf", "pptx", "jpg", "jpeg", "png", "heic", "webp", "gif", "tif", "tiff", "bmp", "dng", "cr2", "cr3", "nef", "arw", "raf", "orf", "rw2",
               "mp4", "mov", "m4v",
               "mp3", "m4a", "wav", "aac", "flac", "aiff", "aif", "opus", "ogg", "caf"}   # indexer.ALL_EXT | MEDIA_EXT


def indexer_running():
    out = subprocess.run(["/usr/bin/pgrep", "-f", r"python indexer\.py"], capture_output=True, text=True).stdout
    return bool(out.split())


def on_battery():
    try:
        out = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "Battery Power" in out


class Watcher:
    def __init__(self, cfg, on_updated, log=print):
        self.cfg = cfg
        self.on_updated, self.log = on_updated, log
        roots = [str(expand(r)) for r in cfg["roots"]]
        roots += [str(expand(r)) for r in cfg.get("media", {}).get("roots", [])]
        self.roots = [r for r in roots if os.path.isdir(r)]
        self.code_roots = [str(r) for r in codesearch.code_roots(cfg)]
        self.code_pending, self.code_last, self.code_proc = set(), 0.0, None
        self.under = [str(expand(u)) for u in cfg["skip"]["under"]]
        self.bundles = tuple(cfg["skip"]["bundle_suffixes"])
        self.pending, self.first_event, self.last_event = set(), 0.0, 0.0
        self.proc, self.running_paths = None, []
        self.retry_at = 0.0
        self.state = self._read_state()
        self.stream = None
        self._start_stream()

    # state ------------------------------------------------------------------------------------------------------------
    def _read_state(self):
        try:
            return json.loads(STATE.read_text())
        except (OSError, ValueError):
            return {}

    def _save_state(self):
        try:
            STATE.parent.mkdir(exist_ok=True)
            STATE.write_text(json.dumps(self.state))
        except OSError:
            pass

    # events -----------------------------------------------------------------------------------------------------------
    def _start_stream(self):
        def callback(stream, info, n, paths, flags, ids):
            for p in paths:
                self._event(str(p))
        flags = (FSEvents.kFSEventStreamCreateFlagUseCFTypes | FSEvents.kFSEventStreamCreateFlagWatchRoot
                 | FSEvents.kFSEventStreamCreateFlagFileEvents)
        watched = sorted(set(self.roots) | set(self.code_roots))
        self.stream = FSEvents.FSEventStreamCreate(None, callback, None, watched,
                                                   FSEvents.kFSEventStreamEventIdSinceNow, 5.0, flags)
        FSEvents.FSEventStreamScheduleWithRunLoop(self.stream, FSEvents.CFRunLoopGetMain(),
                                                  FSEvents.kCFRunLoopDefaultMode)
        FSEvents.FSEventStreamStart(self.stream)

    def _event(self, path):
        path = path.rstrip("/")
        ext = os.path.splitext(path)[1][1:].lower()
        is_dir = not ext and (os.path.isdir(path) or not os.path.exists(path))
        croot = next((r for r in self.code_roots if path == r or path.startswith(r + "/")), None)
        if croot is not None and (is_dir or ext in codesearch.CODE_EXT) and not any(
                part.startswith(".") or part in codesearch.SKIP_DIRS for part in path[len(croot):].split("/")):
            self.code_pending.add(path if is_dir else os.path.dirname(path))
            self.code_last = time.time()
        if not is_dir:
            if ext not in INDEXED_EXT:
                return
            path = os.path.dirname(path)
        root = next((r for r in self.roots if path == r or path.startswith(r + "/")), None)
        if root is None or any(path == u or path.startswith(u + "/") for u in self.under):
            return
        for part in path[len(root):].split("/"):
            if part.startswith(".") or part.lower().endswith(self.bundles):
                return
        now = time.time()
        if not self.pending:
            self.first_event = now
        self.pending.add(path)
        self.last_event = now

    # the timer, every 15 s ---------------------------------------------------------------------------------------------
    def tick(self):
        self._code_tick()
        if self.proc is not None:
            code = self.proc.poll()
            if code is None:
                return
            self.proc = None
            if code == 75:   # a refresh is writing the index: try again in two minutes
                self.pending.update(self.running_paths)
                self.retry_at = time.time() + 120
            else:
                self.state["last_update"] = time.time()
                self.state["last_update_files"] = self._changed_count()
                self._save_state()
                self.on_updated()
            return
        now = time.time()
        if self.pending and (now - self.last_event > QUIET_S or now - self.first_event > MAX_WAIT_S):
            if now >= self.retry_at:   # the indexer's lock decides; a refresh waiting for power lets go of it
                self._update(sorted(self.pending))
                self.pending.clear()
            return
        self._nightly()

    def _code_tick(self):
        if self.code_proc is not None:
            if self.code_proc.poll() is None:
                return
            self.code_proc = None
        if self.code_pending and time.time() - self.code_last > CODE_QUIET_S:
            top = sorted(self.code_pending)
            top = [p for p in top if not any(p != q and p.startswith(q + "/") for q in top)]
            top = [p if os.path.isdir(p) else os.path.dirname(p) for p in top]
            self.code_pending.clear()
            self.code_proc = subprocess.Popen([sys.executable, str(HERE / "codesearch.py"), "--index", "--paths",
                                               *sorted(set(top))], cwd=HERE, start_new_session=True,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _changed_count(self):
        try:
            with open(HERE / "logs" / "watch.out") as f:
                for line in f:
                    if " scan: " in line and "new or changed" in line:
                        return int(line.split(" new or changed")[0].rsplit(", ", 1)[-1])
        except (OSError, ValueError):
            pass
        return None

    def _update(self, paths):
        # Folders inside another changed folder are covered by its walk.
        top = [p for p in paths if not any(p != q and p.startswith(q + "/") for q in paths)]
        self.running_paths = top
        self.log(f"watch: updating {len(top)} folder(s): " + ", ".join(os.path.relpath(p, os.path.expanduser("~"))
                                                                         for p in top[:4]))
        with open(HERE / "logs" / "watch.out", "w") as out:
            self.proc = subprocess.Popen([sys.executable, str(HERE / "indexer.py"), "--no-wait", "--paths", *top],
                                         cwd=HERE, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    def _nightly(self):
        """A full refresh once a night between 2 and 6, on mains power: it catches what the watch cannot (a folder
        renamed while the Mac slept), and embeds anything the quick updates left for mains power. A laptop asleep
        through those hours gets it the next time it is on mains power instead, once 36 hours have gone by."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        missed = time.time() - self.state.get("last_nightly_at", 0) > 36 * 3600
        if ((now.hour not in NIGHT_HOURS and not missed) or self.state.get("last_nightly") == today or on_battery()
                or indexer_running()):
            return
        self.state["last_nightly"] = today
        self.state["last_nightly_at"] = time.time()
        self._save_state()
        self.log("watch: nightly refresh")
        subprocess.Popen(["/bin/zsh", str(HERE / "tools" / "refresh.sh")], cwd=HERE, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def restart(self, cfg):
        """New folders from the menu: watch the new set."""
        if self.stream is not None:
            FSEvents.FSEventStreamStop(self.stream)
            FSEvents.FSEventStreamInvalidate(self.stream)
            FSEvents.FSEventStreamRelease(self.stream)
        self.__init__(cfg, self.on_updated, self.log)

    def status(self):
        """One line for the menu."""
        if self.proc is not None:
            return "Updating changed folders…"
        if self.pending:
            return f"{len(self.pending)} changed folder(s), updating shortly"
        last = self.state.get("last_update")
        if last:
            mins = int((time.time() - last) / 60)
            return f"Watching {len(self.roots)} folders · last update {'just now' if mins < 1 else f'{mins} min ago'}"
        return f"Watching {len(self.roots)} folders for changes"
