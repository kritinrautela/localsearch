"""The search window (Phase 5): Spotlight and DigUp in one panel.

Spotlight's half answers on every keystroke: applications (ranked by how often you open them from here), a
calculator, files by name from macOS's own Spotlight index (mdfind), and a web-search row. DigUp's half comes from
search.py's hybrid engine: meaning, exact codes, OCR'd scans, photos and video frames, versions collapsed. The
preview pane shows the selected result where it matched: a PDF at the page with the query's words highlighted, a
video cued to the second, the passage of a document, a photo at full size.

Keys: return opens (a video at the matched second, a PDF in Preview with the words already searched), cmd-return
or cmd-R shows in Finder, cmd-C copies the file (text when some is selected), cmd-shift-C copies the path, space
(once you have moved into the results) or cmd-Y Quick Look, cmd-1..9 opens that row, tab cycles All / Documents /
Photos & Video / Apps / Clipboard, escape clears then closes. Rows drag out to Finder, Mail or anywhere files go.

Prefixes: name: (file names only), code: (the code index, opens your editor at the line), clip: (clipboard
history). Words like "video", "photos", "pdf" or "mp4" next to other words filter by kind, as kind: does; date:today,
date:week, date:month, date:year, date:2025 or date:2025-08 keep files changed then. cmd-K opens every action for the
selected result (Open With, Find Similar, Copy Path...); cmd-L finds files like it. Your Shortcuts run from here too.

The index keeps itself current: watch.py runs a quick update of any indexed folder that changes, and a full
refresh once a night on mains power.

Hotkey: [app] hotkey in config.toml (cmd+space). While Spotlight still holds cmd-space, the window uses [app]
fallback_hotkey and takes cmd-space by itself once Spotlight's shortcut is moved (checked every 10 s).

Local only: the engine's network guard (common.py) is imported first; the query embedder is a llama-server of its
own on 127.0.0.1 ([app] port) that stops after [app] idle_stop_min without a search. Nothing reaches the internet
unless you choose the web-search row, which hands a URL to your browser.

    uv run python app.py                    # menu bar item + hotkey; tools/install_agent.sh starts it at login
    uv run python app.py --demo "query" --shot out.png [--down N]   # show, search, save the window, quit
"""

import ast
import copy
import functools
import ctypes
import json
import math
import operator
import os
import plistlib
import queue
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import common  # noqa: F401  (network guard before anything else opens a socket)
from common import HERE, HOME, Embedder, expand, fold, load_config
from search import Searcher, clock, code_tokens, terms
import codesearch

import AppKit
import AVFoundation
import AVKit
import CoreMedia
import CoreServices
import Foundation
import objc
import Quartz
from Foundation import NSURL, NSMakeRange, NSMakeRect, NSObject
from PyObjCTools import AppHelper

WIDTH = 860
LIST_W = 400
FIELD_H = 56
BODY_H = 452
FOOT_H = 30
RADIUS = 28    # the bar alone is a capsule, as Spotlight's is
META_H = 166   # the preview's title and Kind / Size / Modified grid
ROW_H = {"header": 28, "hit": 50, "item": 34}
K = 30
DEBOUNCE_S = 0.11
THUMB_PX = 80
STATE = HERE / "state"

IMAGE_EXT = {"jpg", "jpeg", "png", "heic", "webp", "gif", "tif", "tiff", "bmp", "dng", "cr2", "cr3", "nef", "arw", "raf", "orf", "rw2"}
VIDEO_EXT = {"mp4", "mov", "m4v", "avi", "mkv", "webm", "mts", "3gp"}
AUDIO_EXT = {"mp3", "m4a", "wav", "aac", "flac", "aiff", "aif", "opus", "ogg", "caf"}
IINA_CLI = "/Applications/IINA.app/Contents/MacOS/iina-cli"
QUICKTIME_EXT = {"mp4", "mov", "m4v", "mp3", "m4a", "wav", "aac", "aiff", "aif", "caf"}
SCOPES = ["All", "Documents", "Photos & Video", "Apps", "Clipboard"]
GRID_COLS = 4
BIG = (1240, 760, 660)   # the expanded window (⌘O): width, grid width, body height
BIG_GRID_COLS = 6
GRID_PX = 240
DEBUG = bool(os.environ.get("LOCALSEARCH_DEBUG"))


def dbg(*a):
    if DEBUG:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def ago(t):
    s = max(0, time.time() - t)
    if s < 90:
        return "just now"
    if s < 5400:
        return f"{s / 60:.0f} min ago"
    if s < 172800:
        return f"{s / 3600:.0f} h ago"
    return f"{s / 86400:.0f} days ago"


def day(t):
    return time.strftime("%-d %b %Y", time.localtime(t))


def size_text(n):
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024


def ext_of(path):
    name = os.path.basename(path)
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def is_media(path):
    return ext_of(path) in IMAGE_EXT | VIDEO_EXT


def tidy(text):
    """Snippet text without runs of rule characters (----, ====, ....) and with the first «match» in view."""
    text = re.sub(r"([-—–_=─━•●*#|~.·])(\s*\1){3,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    a = text.find("«")
    if a > 40:
        cut = text.find(" ", a - 40)
        text = "…" + text[cut + 1 if 0 <= cut < a else a - 40:]
    return text


def short_path(path):
    p = str(path)
    home = str(HOME)
    drive = home + "/Library/CloudStorage/GoogleDrive-"
    if p.startswith(drive):
        rest = p[len(drive):].split("/", 1)
        p = "Google Drive" + ("/" + rest[1] if len(rest) > 1 else "")
        p = p.replace("Google Drive/My Drive", "Google Drive")
    elif p.startswith(home):
        p = "~" + p[len(home):]
    return p


# --- calculator ----------------------------------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "log": math.log10, "ln": math.log, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "abs": abs, "round": round}
_CONSTS = {"pi": math.pi, "e": math.e}


def calculate(q):
    """The value of an arithmetic expression (1,250 * 1.08, 2^10, sqrt(2), 15% of 80) as text, or None."""
    s = q.strip().lower().replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    if re.fullmatch(r"\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", s):
        return None   # a date, not a subtraction
    s = re.sub(r"(\d+(?:\.\d+)?)\s*% of\s*", r"(\1/100)*", s)
    s = re.sub(r"(\d)\s*x\s*(\d)", r"\1*\2", s)
    if not re.search(r"\d", s) or not re.fullmatch(r"[\d\s.+\-*/()%a-z]+", s):
        return None
    if not re.search(r"[+\-*/%]|\b(sqrt|log|ln|sin|cos|tan|abs|round)\b", s) or re.fullmatch(r"-?[\d.]+", s.strip()):
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 1000 or abs(a) > 1e6):
                raise ValueError("too big")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and len(n.args) == 1:
            return _FUNCS[n.func.id](ev(n.args[0]))
        raise ValueError("not arithmetic")
    try:
        v = ev(ast.parse(s, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        if v.is_integer() and abs(v) < 1e15:
            v = int(v)
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.10g}" if abs(v) >= 1e-4 else f"{v:.6g}"


# --- unit conversions: "10 oz to g", "5 km in miles", "30 c to f" ---------------------------------------------------

_UNITS, _LABEL = {}, {}


def _u(dim, factor, *names):
    for n in names:
        _UNITS[n] = (dim, factor)
        _LABEL[n] = names[0]


_u("mass", 1, "g", "gram", "grams", "gm")
_u("mass", 1000, "kg", "kilo", "kilos", "kilogram", "kilograms")
_u("mass", 0.001, "mg", "milligram", "milligrams")
_u("mass", 1e6, "t", "tonne", "tonnes", "ton", "tons")
_u("mass", 453.59237, "lb", "lbs", "pound", "pounds")
_u("mass", 28.349523125, "oz", "ounce", "ounces")
_u("mass", 31.1034768, "troy oz", "ozt", "toz", "troy ounce", "troy ounces")
_u("length", 0.001, "mm", "millimetre", "millimetres", "millimeter", "millimeters")
_u("length", 0.01, "cm", "centimetre", "centimetres", "centimeter", "centimeters")
_u("length", 1, "m", "metre", "metres", "meter", "meters")
_u("length", 1000, "km", "kilometre", "kilometres", "kilometer", "kilometers")
_u("length", 0.0254, "in", "inch", "inches")
_u("length", 0.3048, "ft", "foot", "feet")
_u("length", 0.9144, "yd", "yard", "yards")
_u("length", 1609.344, "mi", "mile", "miles")
_u("area", 1, "m²", "m2", "sqm", "square metres", "square meters")
_u("area", 1e4, "ha", "hectare", "hectares")
_u("area", 1e6, "km²", "km2", "sq km", "square kilometres", "square kilometers")
_u("area", 4046.8564224, "acres", "acre", "ac")
_u("area", 0.09290304, "sq ft", "sqft", "ft2", "square feet")
_u("volume", 1, "L", "l", "litre", "litres", "liter", "liters")
_u("volume", 0.001, "mL", "ml", "millilitre", "millilitres")
_u("volume", 3.785411784, "gal", "gallon", "gallons")
_u("volume", 1000, "m³", "m3", "cubic metres", "cubic meters")
_TEMPS = {"c": "°C", "°c": "°C", "celsius": "°C", "f": "°F", "°f": "°F", "fahrenheit": "°F", "k": "K", "kelvin": "K"}


def fmt_number(v):
    if abs(v) >= 1e15 or (v and abs(v) < 1e-4):
        return f"{v:.6g}"
    return f"{round(v, 4):,.10g}" if abs(v) < 1000 else f"{v:,.2f}".rstrip("0").rstrip(".")


def convert(q):
    """Conversions as [(result, expression, copyable number)], or None. A plain ounce is shown both ways, since an
    ounce of gold is a troy ounce."""
    m = re.fullmatch(r"\s*(-?[\d.,]+)\s*([a-z°²³ 0-9]*?[a-z°²³])\s+(?:to|in|into|as|=)\s+([a-z°²³ 0-9]*?[a-z°²³])\s*",
                     q.lower())
    if not m:
        return None
    try:
        v = float(m[1].replace(",", ""))
    except ValueError:
        return None
    a, b = m[2].strip(), m[3].strip()
    if a in _TEMPS and b in _TEMPS:
        ta, tb = _TEMPS[a], _TEMPS[b]
        c = {"°C": v, "°F": (v - 32) * 5 / 9, "K": v - 273.15}[ta]
        r = {"°C": c, "°F": c * 9 / 5 + 32, "K": c + 273.15}[tb]
        return [(f"{fmt_number(r)} {tb}", f"{fmt_number(v)} {ta}", f"{r:.6g}")]
    ua, ub = _UNITS.get(a), _UNITS.get(b)
    if not ua or not ub or ua[0] != ub[0]:
        return None
    out = []
    pairs = [(a, b)]
    if "oz" in (_LABEL[a], _LABEL[b]):
        pairs.append(tuple("troy oz" if _LABEL[x] == "oz" else x for x in (a, b)))
    for x, y in pairs:
        r = v * _UNITS[x][1] / _UNITS[y][1]
        out.append((f"{fmt_number(r)} {_LABEL[y]}", f"{fmt_number(v)} {_LABEL[x]}", f"{r:.10g}"))
    return out


LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
         "magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
         "consequat.")
ZONES = {"perth": "Australia/Perth", "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
         "india": "Asia/Kolkata", "delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata", "dubai": "Asia/Dubai",
         "uae": "Asia/Dubai", "angola": "Africa/Luanda", "luanda": "Africa/Luanda", "uige": "Africa/Luanda",
         "drc": "Africa/Kinshasa", "kinshasa": "Africa/Kinshasa", "lubumbashi": "Africa/Lubumbashi",
         "london": "Europe/London", "uk": "Europe/London", "new york": "America/New_York", "nyc": "America/New_York",
         "toronto": "America/Toronto", "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong",
         "tokyo": "Asia/Tokyo", "johannesburg": "Africa/Johannesburg", "utc": "UTC"}


def run_out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def instant(q):
    """[(value, what it is, what Return copies)] for the few words that are a question about this Mac or a tool:
    ip, uuid, battery, disk, a #colour, lorem, "time in dubai"."""
    ql = fold(q).strip()
    if ql in ("ip", "my ip", "ip address", "local ip"):
        ip = run_out(["/usr/sbin/ipconfig", "getifaddr", "en0"]) or run_out(["/usr/sbin/ipconfig", "getifaddr", "en1"])
        return [(ip, "Local IP address (Wi-Fi)", ip)] if ip else [("Offline", "No local IP address", "")]
    if ql in ("uuid", "guid", "new uuid"):
        u = str(uuid.uuid4())
        return [(u, "A new random UUID", u)]
    if ql in ("battery", "battery level", "charge"):
        out = run_out(["/usr/bin/pmset", "-g", "batt"])
        m = re.search(r"(\d+)%;\s*([^;]+);\s*([^\n]*)", out)
        if m:
            left = m[3].split(" present")[0].strip()
            return [(f"{m[1]}%", f"Battery · {m[2].strip()}" + (f" · {left}" if left and "no estimate" not in left
                                                                 else ""), f"{m[1]}%")]
    if ql in ("disk", "disk space", "storage", "free space"):
        import shutil
        du = shutil.disk_usage("/")
        return [(f"{du.free / 1e9:.0f} GB free", f"of {du.total / 1e9:.0f} GB on this Mac", f"{du.free / 1e9:.0f} GB")]
    if ql in ("lorem", "lorem ipsum", "placeholder text"):
        return [(LOREM[:60] + "…", "Placeholder paragraph", LOREM)]
    m = re.fullmatch(r"#?([0-9a-f]{6}|[0-9a-f]{3})", ql)
    if m and (q.strip().startswith("#") or re.search(r"[a-f]", ql) and re.search(r"\d", ql)):
        h = m[1] if len(m[1]) == 6 else "".join(c * 2 for c in m[1])
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return [(f"rgb({r}, {g}, {b})", f"#{h.upper()} as RGB", f"rgb({r}, {g}, {b})")]
    m = re.fullmatch(r"rgb\(?\s*(\d{1,3})[ ,]+(\d{1,3})[ ,]+(\d{1,3})\s*\)?", ql)
    if m and all(int(x) <= 255 for x in m.groups()):
        h = "#" + "".join(f"{int(x):02X}" for x in m.groups())
        return [(h, f"rgb({', '.join(m.groups())}) as hex", h)]
    m = re.fullmatch(r"(?:what time is it in|time in|time|clock)\s+(.+)", ql)
    if m and m[1] in ZONES:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(ZONES[m[1]]))
        diff = (now.utcoffset() - datetime.now().astimezone().utcoffset()).total_seconds() / 3600
        rel = "same as here" if not diff else f"{abs(diff):g} h {'ahead' if diff > 0 else 'behind'}"
        return [(now.strftime("%H:%M"), f"{m[1].title()} · {now:%a %-d %b} · {rel}", now.strftime("%H:%M"))]
    return []


def answers(q):
    """Calculator or conversion results for q, or [] when q is neither."""
    v = calculate(q)
    if v:
        return [(v, q, v.replace(",", ""))]
    return convert(q) or []


# --- System Settings pages -------------------------------------------------------------------------------------------

SETTINGS_APP = "/System/Applications/System Settings.app"
SETTINGS = [
    ("Wi-Fi", "com.apple.wifi-settings-extension", "wifi wireless internet"),
    ("Bluetooth", "com.apple.BluetoothSettings", "airpods headphones"),
    ("Network", "com.apple.Network-Settings.extension", "vpn ethernet firewall"),
    ("Battery", "com.apple.Battery-Settings.extension", "power energy charging low power"),
    ("Displays", "com.apple.Displays-Settings.extension", "monitor screen resolution brightness night shift"),
    ("Sound", "com.apple.Sound-Settings.extension", "volume audio speaker microphone output input"),
    ("Keyboard", "com.apple.Keyboard-Settings.extension", "shortcuts input sources dictation"),
    ("Trackpad", "com.apple.Trackpad-Settings.extension", "gestures scroll"),
    ("Mouse", "com.apple.Mouse-Settings.extension", ""),
    ("Notifications", "com.apple.Notifications-Settings.extension", "alerts banners"),
    ("Focus", "com.apple.Focus-Settings.extension", "do not disturb"),
    ("Privacy & Security", "com.apple.settings.PrivacySecurity.extension",
     "permissions camera microphone full disk access filevault location"),
    ("Desktop & Dock", "com.apple.Desktop-Settings.extension", "stage manager mission control hot corners"),
    ("Wallpaper", "com.apple.Wallpaper-Settings.extension", "background"),
    ("Appearance", "com.apple.Appearance-Settings.extension", "dark mode light accent colour color"),
    ("General", "com.apple.systempreferences.GeneralSettings", "about this mac"),
    ("Software Update", "com.apple.Software-Update-Settings.extension", "upgrade macos"),
    ("Storage", "com.apple.settings.Storage", "disk space"),
    ("Login Items", "com.apple.LoginItems-Settings.extension", "startup background apps"),
    ("Lock Screen", "com.apple.Lock-Screen-Settings.extension", "screen saver sleep"),
    ("Touch ID & Password", "com.apple.Touch-ID-Settings.extension", "fingerprint"),
    ("Users & Groups", "com.apple.Users-Groups-Settings.extension", "accounts"),
    ("Accessibility", "com.apple.Accessibility-Settings.extension", "zoom voiceover"),
    ("Control Center", "com.apple.ControlCenter-Settings.extension", "menu bar"),
    ("Spotlight", "com.apple.Spotlight-Settings.extension", ""),
    ("Time Machine", "com.apple.Time-Machine-Settings.extension", "backup"),
    ("Date & Time", "com.apple.Date-Time-Settings.extension", "clock time zone"),
    ("Language & Region", "com.apple.Localization-Settings.extension", "language region"),
    ("Printers & Scanners", "com.apple.Print-Scan-Settings.extension", "printer scanner"),
    ("Screen Time", "com.apple.Screen-Time-Settings.extension", ""),
    ("Passwords", "com.apple.Passwords-Settings.extension", ""),
]


def match_settings(q, n=3):
    q = fold(q.strip())
    if len(q) < 2:
        return []
    out = []
    for name, pane, extra in SETTINGS:
        low = fold(name)
        words = re.findall(r"[a-z0-9]+", low)
        if low.startswith(q) or low.replace("-", "").startswith(q):
            s = 100 - len(low) * 0.1
        elif any(w.startswith(q) for w in words):
            s = 80
        elif len(q) >= 3 and any(w.startswith(q) for w in extra.split()) or (len(q) >= 5 and q in extra):
            s = 60
        else:
            continue
        out.append({"kind": "setting", "name": name, "pane": pane, "score": s})
    out.sort(key=lambda x: -x["score"])
    return out[:n]


# --- dictionary: macOS's own offline dictionaries ---------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def define(word):
    word = word.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]{2,29}", word):
        return None
    try:
        text = CoreServices.DCSCopyTextDefinition(None, word, (0, len(word)))
    except Exception:
        return None
    return str(text) if text else None


def definition_gist(text):
    """The sense text after 'word | pronunciation |'."""
    parts = text.split("|")
    return (parts[-1] if len(parts) >= 3 else text).strip()


def definition_layout(text):
    return re.sub(r"\s(?=(?:\d{1,2} [a-z(]|• |▸|PHRASES|PHRASAL VERBS|DERIVATIVES|ORIGIN|USAGE))", "\n\n", text)


# --- query filters: kind:pdf, kind:photo, in:Work ---------------------------------------------------------------------

_DOCS = {"doc", "docx", "pages", "rtf", "txt", "md", "odt"}
_SHEETS = {"xls", "xlsx", "xlsm", "csv", "numbers", "tsv"}
_SLIDES = {"ppt", "pptx", "key"}
KINDS = {"pdf": {"pdf"}, "doc": _DOCS, "docs": _DOCS, "word": {"doc", "docx"}, "text": {"txt", "md"},
         "sheet": _SHEETS, "sheets": _SHEETS, "excel": {"xls", "xlsx", "xlsm"}, "spreadsheet": _SHEETS,
         "csv": {"csv"}, "slides": _SLIDES, "deck": _SLIDES, "ppt": _SLIDES, "powerpoint": _SLIDES,
         "photo": IMAGE_EXT, "photos": IMAGE_EXT, "image": IMAGE_EXT, "images": IMAGE_EXT, "picture": IMAGE_EXT,
         "video": VIDEO_EXT, "videos": VIDEO_EXT, "movie": VIDEO_EXT, "movies": VIDEO_EXT,
         "audio": AUDIO_EXT, "sound": AUDIO_EXT, "song": AUDIO_EXT, "songs": AUDIO_EXT, "music": AUDIO_EXT,
         "recording": AUDIO_EXT, "podcast": AUDIO_EXT, "html": {"html", "htm"}, "web": {"html", "htm"},
         **{e: {e} for e in AUDIO_EXT | {"odt", "md", "txt", "docx", "xlsx", "pptx"}}}
# Plain words that filter like kind: when they sit next to other words ("gold video", "licence pdf").
IMPLICIT = {"pdf": {"pdf"}, "pdfs": {"pdf"}, "video": VIDEO_EXT, "videos": VIDEO_EXT, "movie": VIDEO_EXT,
            "movies": VIDEO_EXT, "footage": VIDEO_EXT, "photo": IMAGE_EXT, "photos": IMAGE_EXT,
            "picture": IMAGE_EXT, "pictures": IMAGE_EXT, "pics": IMAGE_EXT, "image": IMAGE_EXT,
            "images": IMAGE_EXT, "screenshot": IMAGE_EXT, "screenshots": IMAGE_EXT, "spreadsheet": _SHEETS,
            "spreadsheets": _SHEETS, "excel": {"xls", "xlsx", "xlsm"}, "powerpoint": _SLIDES, "slides": _SLIDES,
            "audio": AUDIO_EXT, "song": AUDIO_EXT, "songs": AUDIO_EXT, "recording": AUDIO_EXT,
            "recordings": AUDIO_EXT, "podcast": AUDIO_EXT, "podcasts": AUDIO_EXT,
            **{e: {e} for e in IMAGE_EXT | VIDEO_EXT | AUDIO_EXT | {"docx", "xlsx", "pptx", "csv"}}}


RECENT_SKIP_EXT = {"pyc", "pyo", "log", "tmp", "lock", "db", "sqlite", "sqlite3", "wal", "shm", "part", "crdownload",
                   "plist", "o", "so", "dylib"}
RECENT_SKIP_DIRS = ("/__pycache__/", "/node_modules/", "/site-packages/", "/venv/", "/build/", "/dist/")


def parse_filters(q):
    """(query without filters, extensions wanted, folder words wanted)."""
    exts, places = set(), []

    def take(m):
        key, val = m[1].lower(), m[2].lower()
        if key == "kind" and val in KINDS:
            exts.update(KINDS[val])
            return " "
        if key == "in":
            places.append(fold(val))
            return " "
        if key == "date":
            span = date_span(val)
            if span:
                places.append(span)
                return " "
        if key == "tag":
            places.append(("tag", fold(m[2].replace("_", " "))))
            return " "
        return m[0]
    clean = re.sub(r"(?<!\S)(kind|in|date|tag):(\S+)", take, q, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not exts:
        words = clean.split()
        # Only kind words at either end count ("gold video", "pdf licence"), never one inside a phrase.
        lead = 0
        while lead < len(words) and words[lead].lower() in IMPLICIT:
            lead += 1
        tail = len(words)
        while tail > lead and words[tail - 1].lower() in IMPLICIT:
            tail -= 1
        sets = [IMPLICIT[w.lower()] for w in words[:lead] + words[tail:]]
        rest = words[lead:tail]
        if sets and rest:
            both = set.intersection(*map(set, sets))
            exts = both or set.union(*map(set, sets))
            clean = " ".join(rest)
    return clean, exts, places


def prefix_mode(q):
    """("name" | "code" | "clip" | "like" | "path" | "emoji" | None, the rest of the query)."""
    if q == "~" or q.startswith(("~/", "/")):
        return "path", q
    if re.match(r":[a-z]", q, re.IGNORECASE):
        return "emoji", q[1:]
    if re.match(r"emoji\s", q, re.IGNORECASE):
        return "emoji", q.split(None, 1)[1]
    if re.match(r"(snip|snippet|snippets)(\s|$)", q, re.IGNORECASE):
        return "snip", q.split(None, 1)[1] if " " in q.strip() else ""
    m = re.match(r"(name|code|clip|like):\s*(.*)$", q, re.IGNORECASE | re.DOTALL)
    return (m[1].lower(), m[2]) if m else (None, q)


def date_span(val):
    """(start, end) epoch seconds for date:today / yesterday / week / month / year / 2025 / 2025-08, else None."""
    now = datetime.now()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    spans = {"today": (day, now), "yesterday": (day - timedelta(days=1), day),
             "week": (day - timedelta(days=7), now), "month": (day - timedelta(days=31), now),
             "year": (day - timedelta(days=365), now)}
    if val in spans:
        a, b = spans[val]
    elif re.fullmatch(r"(19|20)\d\d", val):
        a, b = datetime(int(val), 1, 1), datetime(int(val) + 1, 1, 1)
    elif re.fullmatch(r"(19|20)\d\d-(0?[1-9]|1[0-2])", val):
        y, m = map(int, val.split("-"))
        a, b = datetime(y, m, 1), datetime(y + m // 12, m % 12 + 1, 1)
    else:
        return None
    return (a.timestamp(), b.timestamp())


def tag_filter(places):
    return next((p[1] for p in places if isinstance(p, tuple) and p[0] == "tag"), None)


def file_tags(path):
    """The file's Finder tags (Red, Important...), read through Foundation rather than an xattr parse."""
    ok, tags, _ = NSURL.fileURLWithPath_(path).getResourceValue_forKey_error_(None, Foundation.NSURLTagNamesKey, None)
    return list(tags or []) if ok else []


def passes(path, exts, places):
    if exts and ext_of(path) not in exts:
        return False
    for p in places:
        if isinstance(p, tuple) and p[0] == "tag":
            if not any(fold(t).startswith(p[1]) for t in file_tags(path)):
                return False
        elif isinstance(p, tuple):
            try:
                if not p[0] <= os.stat(path).st_mtime < p[1]:
                    return False
            except OSError:
                return False
        elif p not in fold(path):
            return False
    return True


def mark_words(text, query):
    """Wrap the query's words in «» wherever they occur in text (titles, file names)."""
    words = sorted({w for w in query.split() if len(w) >= 2}, key=len, reverse=True)
    if not words:
        return text
    rx = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)
    return rx.sub(lambda m: "«" + m.group(0) + "»", text)


# --- quick actions -----------------------------------------------------------------------------------------------------

ACTIONS = [   # (title, more words it answers to, symbol, action, what it does)
    ("Take Screenshot", "screenshot capture screen grab snip", "camera.viewfinder", "screenshot",
     "Opens the Screenshot tool"),
    ("New Note", "note notes write jot", "note.text.badge.plus", "note", "Makes a note in Notes"),
    ("New Reminder", "reminder remind todo task", "checklist", "reminder", "Adds a reminder to Reminders"),
    ("Keep Mac Awake for 1 Hour", "awake caffeinate amphetamine nosleep keep", "cup.and.saucer", "awake",
     "Holds off sleep for an hour"),
    ("Sleep Display", "sleep display screen off monitor", "moon.zzz", "displaysleep", "Turns the screen off now"),
    ("Toggle Dark Mode", "dark mode light appearance theme", "circle.lefthalf.filled", "dark",
     "Switches between light and dark"),
    ("Eject All Disks", "eject disks drives unmount usb", "eject", "eject", "Ejects every removable disk"),
    ("Update Search Index", "update index refresh reindex localsearch", "arrow.clockwise", "reindex",
     "Looks for new and changed files now"),
    ("Clear Clipboard History", "clear clipboard history forget copied", "clear", "clearclip",
     "Forgets everything copied since localsearch started"),
    ("Lock Screen", "lock screen away", "lock", "lock", "Locks the Mac at once"),
    ("Sleep", "sleep mac computer suspend", "powersleep", "powersleep", "Puts the Mac to sleep"),
    ("Restart", "restart reboot", "restart", "restart", "Restarts the Mac; apps are asked to quit first"),
    ("Shut Down", "shut down shutdown power off turn off", "power", "shutdown",
     "Shuts the Mac down; apps are asked to quit first"),
    ("Log Out", "log out logout sign out", "rectangle.portrait.and.arrow.right", "logout",
     "Logs you out; apps are asked to quit first"),
    ("Settings for localsearch", "localsearch settings preferences folders index options", "gearshape",
     "settings", "Choose the folders searched and the code editor"),
    ("Scratchpad", "scratchpad notepad scratch jot pad quick note", "square.and.pencil", "scratchpad",
     "A floating notepad that keeps what you type"),
    ("Clipboard History", "clipboard history copied paste pasteboard", "doc.on.clipboard", "clipboard",
     "What you copied lately, kept in memory only"),
    ("Save Clipboard as Snippet", "snippet save clipboard snip", "text.badge.plus", "savesnippet",
     "Keeps the copied text as a snippet; find it again with snip"),
]
CONFIRM = {"restart": "Restart", "shutdown": "Shut Down", "logout": "Log Out",
           "kill": "Force Quit"}   # Return twice, as a safety


SNIPPETS = STATE / "snippets.json"


class Snippets:
    """Saved bits of text (addresses, sign-offs, licence numbers) found with "snip <words>", copied with Return."""

    def __init__(self):
        try:
            self.items = json.loads(SNIPPETS.read_text())
        except (OSError, ValueError):
            self.items = []

    def save(self):
        SNIPPETS.parent.mkdir(exist_ok=True)
        SNIPPETS.write_text(json.dumps(self.items, ensure_ascii=False, indent=1))

    def add(self, text, name=None):
        name = name or tidy(text.strip().splitlines()[0])[:48]
        self.items = [s for s in self.items if s["text"] != text] + [{"name": name, "text": text, "at": time.time()}]
        self.save()
        return name

    def forget(self, item):
        self.items = [s for s in self.items if s is not item and s["text"] != item["text"]]
        self.save()

    def find(self, q):
        ql = fold(q).strip()
        found = [s for s in reversed(self.items) if not ql or all(w in fold(s["name"] + " " + s["text"])
                                                                  for w in ql.split())]
        return [{"kind": "snippet", "item": s, "name": s["name"], "value": s["text"]} for s in found[:30]]


class Scratchpad(NSObject):
    """A small floating notepad (Sol's scratchpad): what you type is kept in state/scratchpad.txt as you type."""

    FILE = STATE / "scratchpad.txt"

    def init(self):
        self = objc.super(Scratchpad, self).init()
        if self is None:
            return None
        style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable | AppKit.NSWindowStyleMaskResizable
                 | AppKit.NSWindowStyleMaskUtilityWindow)
        p = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 420, 360), style, AppKit.NSBackingStoreBuffered, False)
        p.setTitle_("Scratchpad")
        p.setFloatingPanel_(True)
        p.setHidesOnDeactivate_(False)
        p.setReleasedWhenClosed_(False)
        p.setFrameAutosaveName_("localsearch.scratchpad")
        scroll = AppKit.NSScrollView.alloc().initWithFrame_(p.contentView().bounds())
        scroll.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        scroll.setHasVerticalScroller_(True)
        tv = AppKit.NSTextView.alloc().initWithFrame_(scroll.bounds())
        tv.setAutoresizingMask_(AppKit.NSViewWidthSizable)
        tv.setRichText_(False)
        tv.setFont_(AppKit.NSFont.monospacedSystemFontOfSize_weight_(13, AppKit.NSFontWeightRegular))
        tv.setTextContainerInset_((10, 10))
        tv.setAutomaticQuoteSubstitutionEnabled_(False)
        tv.setAllowsUndo_(True)
        try:
            tv.setString_(self.FILE.read_text())
        except OSError:
            pass
        tv.setDelegate_(self)
        scroll.setDocumentView_(tv)
        p.setContentView_(scroll)
        p.center()
        self.panel, self.text = p, tv
        return self

    def textDidChange_(self, note):
        try:
            self.FILE.parent.mkdir(exist_ok=True)
            self.FILE.write_text(self.text.string())
        except OSError:
            pass

    def show(self):
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self.panel.makeKeyAndOrderFront_(None)
        self.panel.orderFrontRegardless()
        self.panel.makeFirstResponder_(self.text)


def duration(text):
    """(seconds, the words left over) for "5m", "1h 30m", "90s", "10 min tea", "2.5 hours"; a bare number is
    minutes. None when there is no duration."""
    units = {"h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600, "m": 60, "min": 60, "mins": 60,
             "minute": 60, "minutes": 60, "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1}
    total, rest = 0.0, text
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", text,
                         re.IGNORECASE):
        total += float(m[1]) * units[m[2].lower()]
        rest = rest.replace(m[0], " ", 1)
    if not total:
        m = re.match(r"\s*(\d+(?:\.\d+)?)\b", text)
        if not m:
            return None
        total, rest = float(m[1]) * 60, text[m.end():]
    if not 1 <= total <= 24 * 3600:
        return None
    return total, re.sub(r"\s+", " ", rest).strip(" :-")


def span_text(s):
    s = int(round(s))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    parts = ([f"{h} h"] if h else []) + ([f"{m} min"] if m else []) + ([f"{sec} s"] if sec and not h else [])
    return " ".join(parts) or "0 s"


def timer_rows(q, timers):
    """"timer 5m tea" / "5 min timer" start one; "timer" alone lists the running ones."""
    ql = q.strip()
    m = re.match(r"(?:timer|countdown|alarm in)\s+(.+)$", ql, re.IGNORECASE) or re.match(
        r"(.+?)\s+(?:timer|countdown)$", ql, re.IGNORECASE)
    out = []
    if m:
        d = duration(m[1])
        if d:
            secs, label = d
            out.append({"kind": "action", "name": f"Start {span_text(secs)} Timer" + (f": {label}" if label else ""),
                        "act": "timer", "arg": (secs, label), "sym": "timer", "label": "Timer", "score": 100,
                        "does": f"Notifies you with a sound in {span_text(secs)}"})
    if re.match(r"(timer|timers|countdown)", ql, re.IGNORECASE):
        for tid, (end, label) in sorted(timers.items(), key=lambda kv: kv[1][0]):
            left = max(0, end - time.time())
            out.append({"kind": "action", "name": f"{label or 'Timer'}: {clock(left)} left", "act": "canceltimer",
                        "arg": tid, "sym": "timer", "label": "Running", "score": 95,
                        "does": f"Ends at {time.strftime('%H:%M', time.localtime(end))}; Return cancels it"})
    return out


# --- browser bookmarks -------------------------------------------------------------------------------------------------

CHROMIUM_BOOKMARKS = (
    ("Brave", "Library/Application Support/BraveSoftware/Brave-Browser/Default/Bookmarks", "/Applications/Brave Browser.app"),
    ("Chrome", "Library/Application Support/Google/Chrome/Default/Bookmarks", "/Applications/Google Chrome.app"),
    ("Edge", "Library/Application Support/Microsoft Edge/Default/Bookmarks", "/Applications/Microsoft Edge.app"),
    ("Arc", "Library/Application Support/Arc/User Data/Default/Bookmarks", "/Applications/Arc.app"))
SAFARI_BOOKMARKS = "Library/Safari/Bookmarks.plist"


class Bookmarks:
    """Brave, Chrome, Edge, Arc and Safari bookmarks, read from their own files and re-read when they change."""

    def __init__(self):
        self.items, self.stamp = [], None
        self.refresh()

    def refresh(self):
        files = [HOME / f for _, f, _ in CHROMIUM_BOOKMARKS] + [HOME / SAFARI_BOOKMARKS]
        stamp = tuple(f.stat().st_mtime if f.exists() else 0 for f in files)
        if stamp != self.stamp:
            self.stamp = stamp
            threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        items, seen = [], set()

        def add(title, url, browser, app, folder):
            if url.startswith(("http://", "https://")) and (url, browser) not in seen:
                seen.add((url, browser))
                items.append({"title": title or url, "url": url, "browser": browser, "app": app, "folder": folder})
        for browser, rel, app in CHROMIUM_BOOKMARKS:
            try:
                data = json.load(open(HOME / rel, encoding="utf-8"))
            except (OSError, ValueError):
                continue

            def walk(node, folder):
                if node.get("type") == "url":
                    add(node.get("name"), node.get("url", ""), browser, app, folder)
                for child in node.get("children", []):
                    walk(child, node.get("name") or folder)
            for root in data.get("roots", {}).values():
                if isinstance(root, dict):
                    walk(root, root.get("name", ""))
        try:   # Safari's file needs Full Disk Access; without it Safari's bookmarks are simply left out
            data = plistlib.load(open(HOME / SAFARI_BOOKMARKS, "rb"))

            def walk_safari(node, folder):
                if node.get("WebBookmarkType") == "WebBookmarkTypeLeaf":
                    add(node.get("URIDictionary", {}).get("title"), node.get("URLString", ""), "Safari",
                        "/Applications/Safari.app", folder)
                for child in node.get("Children", []):
                    walk_safari(child, node.get("Title") or folder)
            walk_safari(data, "")
        except (OSError, ValueError, plistlib.InvalidFileException):
            pass
        self.items = items

    def match(self, q, n=4):
        ql = fold(q).strip()
        if len(ql) < 2:
            return []
        words = ql.split()
        out = []
        for b in self.items:
            t, u = fold(b["title"]), fold(b["url"].split("://", 1)[-1])
            tw = re.split(r"[^a-z0-9]+", t)
            if t.startswith(ql) or u.startswith(ql) or u.startswith("www." + ql):
                score = 100
            elif any(w.startswith(ql) for w in tw):
                score = 90
            elif all(any(w.startswith(x) for w in tw) for x in words):
                score = 80
            elif all(x in t or x in u for x in words) and len(ql) >= 3:
                score = 60
            else:
                continue
            out.append({"kind": "bookmark", "name": b["title"], "score": score, **b})
        out.sort(key=lambda b: (-b["score"], len(b["title"])))
        return out[:n]


# --- emoji -------------------------------------------------------------------------------------------------------------

EMOJI_NAMES = "/System/Library/PrivateFrameworks/CoreEmoji.framework/Versions/A/Resources/en.lproj/AppleName.strings"
EMOJI_ALIASES = {"laugh": "face with tears of joy", "lol": "face with tears of joy", "haha": "face with tears of joy",
                 "love": "red heart", "heart": "red heart", "ok": "ok hand", "thanks": "folded hands",
                 "pray": "folded hands", "please": "folded hands", "yes": "thumbs up", "like": "thumbs up",
                 "no": "thumbs down", "party": "party popper", "congrats": "party popper", "cry": "loudly crying face",
                 "sad": "crying face", "cool": "smiling face with sunglasses", "think": "thinking face",
                 "hmm": "thinking face", "smile": "smiling face with smiling eyes", "happy": "grinning face",
                 "wink": "winking face", "kiss": "face blowing a kiss", "angry": "pouting face", "wow": "astonished face",
                 "shock": "face screaming in fear", "done": "check mark button", "tick": "check mark",
                 "check": "check mark", "warning": "warning", "idea": "light bulb", "money": "money bag",
                 "gold": "1st place medal", "rocket": "rocket", "flex": "flexed biceps", "strong": "flexed biceps",
                 "clap": "clapping hands", "wave": "waving hand", "hi": "waving hand", "bye": "waving hand",
                 "100": "hundred points", "eyes": "eyes", "skull": "skull", "dead": "skull", "sun": "sun",
                 "india": "flag: india", "australia": "flag: australia", "angola": "flag: angola",
                 "congo": "flag: congo - kinshasa", "uae": "flag: united arab emirates", "dubai": "flag: united arab emirates"}


class Emoji:
    """Apple's own emoji names (CoreEmoji's AppleName.strings), loaded on first use."""

    def __init__(self):
        self.items = None

    def load(self):
        try:
            names = plistlib.load(open(EMOJI_NAMES, "rb"))
        except (OSError, ValueError, plistlib.InvalidFileException):
            names = {}
        # Skin-tone variants would show every hand five times over; keep the plain one.
        self.items = sorted(((ch, name) for ch, name in names.items()
                             if not any(0x1F3FB <= ord(c) <= 0x1F3FF for c in ch)), key=lambda x: len(x[1]))

    def match(self, q, n=30):
        if self.items is None:
            self.load()
        ql = fold(q).strip()
        if not ql:
            return []
        alias = EMOJI_ALIASES.get(ql)
        words = ql.split()
        scored = []
        for ch, name in self.items:
            nw = re.split(r"[^a-z0-9]+", name.lower())
            if alias and name == alias:
                score = 120
            elif name == ql:
                score = 110
            elif name.startswith(ql):
                score = 100
            elif all(any(w.startswith(x) for w in nw) for x in words):
                score = 80
            else:
                continue
            scored.append((score, ch, name))
        scored.sort(key=lambda s: -s[0])
        return [{"kind": "emoji", "value": ch, "name": name[:1].upper() + name[1:],
                 "code": " ".join(f"U+{ord(c):04X}" for c in ch if ord(c) not in (0xFE0F, 0x200D))}
                for _, ch, name in scored[:n]]


def emoji_image(ch, px=22):
    img = AppKit.NSImage.alloc().initWithSize_((px, px))
    img.lockFocus()
    s = AppKit.NSAttributedString.alloc().initWithString_attributes_(
        ch, {AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(px * 0.78)})
    sz = s.size()
    s.drawAtPoint_(((px - sz.width) / 2, (px - sz.height) / 2))
    img.unlockFocus()
    return img


class Shortcuts:
    """The names of the user's Shortcuts (the shortcuts tool), read in the background and refreshed hourly."""

    def __init__(self):
        self.names, self.read_at = [], 0.0
        self.refresh()

    def refresh(self):
        if time.time() - self.read_at < 3600:
            return
        self.read_at = time.time()

        def read():
            try:
                out = subprocess.run(["/usr/bin/shortcuts", "list"], capture_output=True, text=True,
                                     timeout=20).stdout
                self.names = [n.strip() for n in out.splitlines() if n.strip()]
            except (OSError, subprocess.SubprocessError):
                pass
        threading.Thread(target=read, daemon=True).start()


QUICKLINKS = {   # keyword -> (title, URL with {q}); config [quicklinks] adds more: yt = "https://...{q}"
    "g": ("Google", "https://www.google.com/search?q={q}"),
    "yt": ("YouTube", "https://www.youtube.com/results?search_query={q}"),
    "gh": ("GitHub", "https://github.com/search?q={q}"),
    "maps": ("Maps", "maps://?q={q}"),
    "wiki": ("Wikipedia", "https://en.wikipedia.org/wiki/Special:Search?search={q}"),
    "so": ("Stack Overflow", "https://stackoverflow.com/search?q={q}"),
    "ddg": ("DuckDuckGo", "https://duckduckgo.com/?q={q}"),
    "scholar": ("Google Scholar", "https://scholar.google.com/scholar?q={q}"),
    "hf": ("Hugging Face", "https://huggingface.co/search/full-text?q={q}"),
    "translate": ("Google Translate", "https://translate.google.com/?sl=auto&tl=en&text={q}"),
    "amazon": ("Amazon", "https://www.amazon.com.au/s?k={q}"),
}


def quicklink_rows(q, extra=None):
    """"yt gold price" -> Search YouTube for "gold price". Only ever opened when chosen."""
    links = {**QUICKLINKS, **{k: (k.title(), v) for k, v in (extra or {}).items()}}
    m = re.match(r"(\S+)\s+(.+)$", q.strip())
    if not m or m[1].lower() not in links:
        return []
    title, url = links[m[1].lower()]
    return [{"kind": "link", "name": f"Search {title} for “{m[2]}”", "url": url.replace("{q}", urllib.parse.quote(m[2])),
             "site": title, "score": 100}]


def process_rows(want):
    """Your own processes whose name starts with want, busiest first, for "kill <name>"."""
    out = run_out(["/bin/ps", "-U", str(os.getuid()), "-o", "pid=,pcpu=,rss=,comm="])
    rows = []
    for ln in out.splitlines():
        parts = ln.split(None, 3)
        if len(parts) < 4:
            continue
        pid, cpu, rss, comm = int(parts[0]), float(parts[1]), int(parts[2]), parts[3]
        name = os.path.basename(comm)
        if pid == os.getpid() or not fold(name).startswith(want):
            continue
        app = re.search(r"^(.*?\.app)/", comm)
        rows.append({"kind": "action", "name": f"Force Quit {name}", "act": "kill", "arg": pid, "sym": "bolt.slash",
                     "label": f"{cpu:.0f}% CPU · {rss / 1024:.0f} MB", "score": 100, "app_path": app[1] if app else None,
                     "does": f"Ends process {pid} at once; unsaved work in it is lost", "cpu": cpu})
    return sorted(rows, key=lambda r: -r["cpu"])[:8]


def match_actions(q, awake=False, shortcuts=(), timers=None):
    ql = fold(q).strip()
    if len(ql) < 2:
        return []
    words = ql.split()
    out = timer_rows(q, timers or {})
    if out and out[0]["act"] == "timer":
        return out
    if words[0] in ("kill", "forcequit", "force") and len(words) > 1:
        return process_rows(" ".join(w for w in words[1:] if w != "quit"))
    if words[0] == "quit" and len(words) > 1:
        want = " ".join(words[1:])
        for a in AppKit.NSWorkspace.sharedWorkspace().runningApplications():
            name = a.localizedName() or ""
            if (a.activationPolicy() == AppKit.NSApplicationActivationPolicyRegular and fold(name).startswith(want)
                    and a.processIdentifier() != os.getpid()):
                out.append({"kind": "action", "name": f"Quit {name}", "act": "quit", "sym": "xmark.circle",
                            "arg": a.processIdentifier(), "does": f"Asks {name} to quit, as ⌘Q would",
                            "score": 100, "app_path": a.bundleURL().path() if a.bundleURL() else None})
        return out[:4]
    for title, more, sym, act, does in ACTIONS:
        t = fold(title)
        arg = ""
        if len(words) > 1 and ((act == "note" and words[0] == "note")
                                or (act == "reminder" and words[0] in ("remind", "reminder", "todo"))):
            arg = q.strip().split(None, 1)[1]
            title = f"New {'Note' if act == 'note' else 'Reminder'}: “{arg}”"
            score = 100
        elif t.startswith(ql):
            score = 100
        elif any(w.startswith(ql) for w in t.split()):
            score = 90
        elif any(w.startswith(ql) for w in more.split()):
            score = 80
        elif all(any(w.startswith(x) for w in t.split() + more.split()) for x in words):
            score = 70
        else:
            continue
        if act == "awake" and awake:
            title, sym, does = "Let Mac Sleep Again", "bed.double", "Ends the hour of staying awake"
        out.append({"kind": "action", "name": title, "act": act, "sym": sym, "arg": arg, "does": does,
                    "score": score})
    for name in shortcuts:
        n = fold(name)
        if n.startswith(ql):
            score = 95
        elif any(w.startswith(ql) for w in n.split()):
            score = 85
        elif len(ql) >= 3 and all(any(w.startswith(x) for w in n.split()) for x in words):
            score = 75
        else:
            continue
        out.append({"kind": "action", "name": name, "act": "shortcut", "sym": "square.2.layers.3d", "arg": name,
                    "does": "Runs this shortcut", "score": score, "label": "Shortcut"})
    return sorted(out, key=lambda a: -a["score"])[:4]


# --- clipboard history: in memory only, never written to disk --------------------------------------------------------

class Clipboard:
    # Password managers mark what they copy with these (nspasteboard.org); such items are never kept.
    SKIP = {"org.nspasteboard.ConcealedType", "org.nspasteboard.TransientType", "org.nspasteboard.AutoGeneratedType",
            "com.agilebits.onepassword", "de.petermaurer.TransientPasteboardType", "com.typeit4me.clipping",
            "Pasteboard generator type"}
    LIMIT = 80

    def __init__(self):
        self.items = []
        self.pb = AppKit.NSPasteboard.generalPasteboard()
        self.count = -1
        self.poll()   # what is on the clipboard already is the first item
        AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.7, True, lambda t: self.poll())

    def poll(self):
        c = self.pb.changeCount()
        if c == self.count:
            return
        self.count = c
        types = set(self.pb.types() or [])
        if types & self.SKIP:
            return
        front = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        item = None
        urls = self.pb.readObjectsForClasses_options_(
            [NSURL], {AppKit.NSPasteboardURLReadingFileURLsOnlyKey: True}) or []
        if urls:
            paths = [u.path() for u in urls]
            item = {"type": "files", "paths": paths, "text": ", ".join(os.path.basename(x) for x in paths)}
        elif types & {"public.png", "public.tiff", "public.jpeg"} and AppKit.NSPasteboardTypeString not in types:
            img = AppKit.NSImage.alloc().initWithPasteboard_(self.pb)
            if img is not None:
                w, h = img.size()
                item = {"type": "image", "image": img, "text": f"Image {int(w)} × {int(h)}"}
        else:
            text = self.pb.stringForType_(AppKit.NSPasteboardTypeString)
            if text and text.strip():
                item = {"type": "text", "text": text[:200000]}
        if item is None:
            return
        item["at"] = time.time()
        item["app"] = front.localizedName() if front else ""
        same = (lambda i: i["type"] == item["type"] and i["text"] == item["text"]
                and i.get("paths") == item.get("paths"))
        self.items = [i for i in self.items if not same(i)]
        self.items.insert(0, item)
        del self.items[self.LIMIT:]
        images = [i for i in self.items if i["type"] == "image"]
        for old in images[12:]:
            self.items.remove(old)

    def restore(self, item):
        self.pb.clearContents()
        if item["type"] == "files":
            self.pb.writeObjects_([NSURL.fileURLWithPath_(x) for x in item["paths"]])
        elif item["type"] == "image":
            self.pb.writeObjects_([item["image"]])
        else:
            self.pb.setString_forType_(item["text"], AppKit.NSPasteboardTypeString)

    def forget(self, item=None):
        self.items = [i for i in self.items if i is not item] if item is not None else []

    def find(self, q):
        ql = fold(q.strip())
        return [i for i in self.items if not ql or ql in fold(i["text"])]


# --- history: what was opened from here, for ranking apps and for the recents list ---------------------------------

class History:
    def __init__(self):
        STATE.mkdir(exist_ok=True)
        self.path = STATE / "history.json"
        try:
            self.items = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.items = []
        self.upath = STATE / "uses.json"   # path -> [times opened, last opened]
        try:
            self.uses = json.loads(self.upath.read_text())
        except (OSError, ValueError):
            self.uses = {i["path"]: [1, i.get("at", 0)] for i in self.items if i.get("path")}

    def add(self, row):
        keep = {k: row.get(k) for k in ("kind", "path", "name", "page", "t", "chunk", "hitkind")}
        keep["at"] = time.time()
        self.items = [keep] + [i for i in self.items if i.get("path") != keep["path"]][:199]
        self.uses[keep["path"]] = [self.count(keep["path"]) + 1, keep["at"]]
        try:
            self.path.write_text(json.dumps(self.items))
            self.upath.write_text(json.dumps(self.uses))
        except OSError:
            pass

    def count(self, path):
        return self.uses.get(path, [0])[0]

    def frecency(self, path):
        """Times opened, halved for every two weeks since the last time."""
        n, at = self.uses.get(path, [0, 0])
        return n * 0.5 ** ((time.time() - at) / (14 * 86400))

    def top_apps(self, n):
        apps = [i for i in self.items if i.get("kind") == "app" and i.get("path") and os.path.exists(i["path"])]
        apps.sort(key=lambda i: -self.frecency(i["path"]))
        return [{"kind": "app", "name": i["name"], "path": i["path"]} for i in apps[:n]]


class Pins:
    """Favourites at the top of the empty window; cmd-P pins or unpins the selected app or file."""

    FILE = STATE / "pins.json"

    def __init__(self):
        try:
            self.items = json.loads(self.FILE.read_text())
        except (OSError, ValueError):
            self.items = []

    def has(self, path):
        return any(p["path"] == path for p in self.items)

    def toggle(self, kind, path, name):
        if self.has(path):
            self.items = [p for p in self.items if p["path"] != path]
        else:
            self.items.append({"kind": kind, "path": path, "name": name})
        try:
            self.FILE.parent.mkdir(exist_ok=True)
            self.FILE.write_text(json.dumps(self.items, ensure_ascii=False, indent=1))
        except OSError:
            pass
        return self.has(path)

    def rows(self):
        return [dict(p) for p in self.items if os.path.exists(p["path"])]


def used_apps(n=16):
    """The apps macOS saw you use in the last fortnight, by its own launch counts, recent ones first."""
    try:
        out = subprocess.run(["/usr/bin/mdfind", "-attr", "kMDItemUseCount", "-attr", "kMDItemLastUsedDate",
                              "kMDItemContentType == 'com.apple.application-bundle' && "
                              "kMDItemLastUsedDate >= $time.today(-14)"],
                             capture_output=True, text=True, timeout=6).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    apps = []
    for line in out.splitlines():
        m = re.match(r"(.+?\.app)\s+kMDItemUseCount = (\d+)\s+kMDItemLastUsedDate = (\S+ \S+ \S+)$", line)
        if not m or ".app/" in m[1] or "LocalSearch" in m[1]:
            continue
        age = (time.time() - datetime.strptime(m[3], "%Y-%m-%d %H:%M:%S %z").timestamp()) / 86400
        apps.append((math.log1p(int(m[2])) * 0.5 ** (age / 7), m[1]))
    fm = AppKit.NSFileManager.defaultManager()
    return [{"kind": "app", "name": fm.displayNameAtPath_(p).removesuffix(".app"), "path": p}
            for _, p in sorted(apps, reverse=True)[:n]]


def dock_apps():
    """The apps kept in the Dock, read from its preferences file: the suggestions before there is any history."""
    try:
        with open(HOME / "Library" / "Preferences" / "com.apple.dock.plist", "rb") as f:
            dock = plistlib.load(f)
    except (OSError, ValueError, plistlib.InvalidFileException):
        return []
    out = []
    for tile in dock.get("persistent-apps", []):
        data = tile.get("tile-data") or {}
        url = (data.get("file-data") or {}).get("_CFURLString", "")
        p = urllib.parse.unquote(urllib.parse.urlparse(url).path if url.startswith("file:") else url).rstrip("/")
        if p.endswith(".app") and os.path.exists(p):
            out.append({"kind": "app", "name": data.get("file-label") or os.path.basename(p)[:-4], "path": p})
    return out



# --- applications ----------------------------------------------------------------------------------------------------

APP_DIRS = ("/Applications/", "/System/Applications/", str(HOME / "Applications") + "/",
            "/System/Library/CoreServices/Applications/", "/System/Library/CoreServices/Finder.app")


def subsequence(q, text):
    it = iter(text)
    return all(c in it for c in q)


def index_built(cfg):
    """True once a full indexer run has finished (it records last_index)."""
    try:
        db = sqlite3.connect(f"file:{expand(cfg['index'])}?mode=ro", uri=True)
        try:
            return db.execute("SELECT 1 FROM meta WHERE key = 'last_index'").fetchone() is not None
        finally:
            db.close()
    except sqlite3.Error:
        return False


class Apps:
    """Installed applications from the Spotlight index, matched like Spotlight: name prefix, word starts, initials."""

    def __init__(self, history):
        self.apps = []
        self.history = history
        threading.Thread(target=self.load, daemon=True).start()

    def load(self):
        try:
            out = subprocess.run(["/usr/bin/mdfind", "kMDItemContentType == 'com.apple.application-bundle'"],
                                 capture_output=True, text=True, timeout=20).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            out = []
        fm = AppKit.NSFileManager.defaultManager()
        apps, seen = [], set()
        for p in sorted(out, key=len):
            if not p.startswith(APP_DIRS) or ".app/" in p:
                continue
            name = fm.displayNameAtPath_(p)
            name = name[:-4] if name.endswith(".app") else name
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            words = re.findall(r"[A-Za-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name))
            apps.append({"name": name, "path": p, "low": fold(name),
                         "initials": "".join(w[0] for w in words).lower(), "words": [fold(w) for w in words]})
        self.apps = apps

    def match(self, q, n=4):
        q = fold(q.strip())
        if not q:
            return []
        out = []
        for a in self.apps:
            if a["low"].startswith(q):
                s = 100 - len(a["low"]) * 0.1
            elif any(w.startswith(q) for w in a["words"]):
                s = 80
            elif len(q) >= 2 and a["initials"].startswith(q):
                s = 70
            elif len(q) >= 3 and q in a["low"]:
                s = 55
            elif len(q) >= 3 and any(w[:1] == q[0] and subsequence(q, "".join(a["words"][i:]))
                                     for i, w in enumerate(a["words"])):
                s = 40   # typed with letters left out: "wtsp" for WhatsApp, "pwrpt" for PowerPoint
            else:
                continue
            s += min(self.history.count(a["path"]), 20) * 1.5
            out.append((s, a))
        out.sort(key=lambda x: -x[0])
        return [dict(a, score=s) for s, a in out[:n]]


# --- global hotkey: Carbon RegisterEventHotKey, which needs no Accessibility permission ------------------------------

class _HotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
MODIFIERS = {"cmd": 0x100, "command": 0x100, "shift": 0x200, "option": 0x800, "alt": 0x800, "opt": 0x800,
             "ctrl": 0x1000, "control": 0x1000}
KEYCODES = {"space": 49, "return": 36, "tab": 48, "f": 3, "k": 40, "s": 1, "l": 37, "j": 38, "/": 44, ".": 47}
SYMBOLS = {"cmd": "⌘", "command": "⌘", "shift": "⇧", "option": "⌥", "alt": "⌥", "opt": "⌥", "ctrl": "⌃",
           "control": "⌃", "space": "Space", "return": "↵"}
SPOTLIGHT_KEY = "64"            # com.apple.symbolichotkeys: "Show Spotlight search"
CMD_SPACE = (32, 49, 1 << 20)   # its default: space, keycode 49, command


def fourcc(s):
    return int.from_bytes(s.encode(), "big")


def hotkey_label(spec):
    parts = [p.strip().lower() for p in spec.split("+")]
    return "".join(SYMBOLS.get(p, p.upper()) for p in parts[:-1]) + SYMBOLS.get(parts[-1], parts[-1].upper())


def is_cmd_space(spec):
    return spec.replace(" ", "").lower() in ("cmd+space", "command+space")


def spotlight_holds_cmd_space():
    """True while macOS's own Spotlight shortcut is still cmd-space (the default when the entry is absent)."""
    domain = "com.apple.symbolichotkeys"
    Foundation.CFPreferencesAppSynchronize(domain)
    keys = Foundation.CFPreferencesCopyAppValue("AppleSymbolicHotKeys", domain) or {}
    entry = keys.get(SPOTLIGHT_KEY)
    if entry is None:
        return True
    params = tuple(int(x) for x in (entry.get("value") or {}).get("parameters") or ())
    if bool(entry.get("enabled")) and params == CMD_SPACE:
        return True
    # Tahoe's "Show Apps" (entry 160) can sit on cmd-space too; then both windows would open at once.
    apps = keys.get("160") or {}
    params = tuple(int(x) for x in (apps.get("value") or {}).get("parameters") or ())
    return bool(apps.get("enabled")) and params == CMD_SPACE


class HotKey:
    """One global hotkey; rebind() swaps the combination."""

    def __init__(self, callback):
        self.carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        self.carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
        self.target = self.carbon.GetApplicationEventTarget()
        self._cb = _HANDLER(lambda call, event, data: (AppHelper.callAfter(callback), 0)[1])  # kept alive on self
        spec_t = _EventTypeSpec(fourcc("keyb"), 5)  # kEventClassKeyboard, kEventHotKeyPressed
        handler_ref = ctypes.c_void_p()
        self.carbon.InstallEventHandler.argtypes = [ctypes.c_void_p, _HANDLER, ctypes.c_uint32,
                                                    ctypes.POINTER(_EventTypeSpec), ctypes.c_void_p,
                                                    ctypes.POINTER(ctypes.c_void_p)]
        self.carbon.InstallEventHandler(self.target, self._cb, 1, ctypes.byref(spec_t), None,
                                        ctypes.byref(handler_ref))
        self.carbon.RegisterEventHotKey.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _HotKeyID, ctypes.c_void_p,
                                                    ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        self.carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
        self.ref, self.spec, self.label, self.ok = None, None, "", False

    def rebind(self, spec):
        if spec == self.spec and self.ok:
            return True
        if self.ref is not None:
            self.carbon.UnregisterEventHotKey(self.ref)
            self.ref = None
        parts = [p.strip().lower() for p in spec.split("+")]
        mods = sum(MODIFIERS[p] for p in parts[:-1])
        ref = ctypes.c_void_p()
        status = self.carbon.RegisterEventHotKey(KEYCODES[parts[-1]], mods, _HotKeyID(fourcc("lsrc"), 1),
                                                 self.target, 0, ctypes.byref(ref))
        self.ok = status == 0   # -9878: another app already holds this combination
        self.ref = ref if self.ok else None
        self.spec, self.label = spec, hotkey_label(spec)
        print(time.strftime("%H:%M:%S"), "hotkey", self.label,
              "registered" if self.ok else f"NOT registered ({status})", flush=True)
        return self.ok


# --- engine: one worker thread owns the Searcher and the embedder ------------------------------------------------------

class Engine:
    """Runs deep searches off the main thread. post(q) supersedes any query not yet started; results come back
    through on_results(gen, query, hits, info) on the main thread."""

    def __init__(self, cfg, on_results, on_status):
        self.cfg = cfg
        app = cfg.get("app", {})
        self.embed_cfg = copy.deepcopy(cfg)
        self.embed_cfg["model"]["port"] = app.get("port", 18081)
        self.idle_stop = app.get("idle_stop_min", 15) * 60
        self.on_results, self.on_status = on_results, on_status
        self.searcher = None
        self.info = None
        self.emb = None
        self.emb_starting = False
        self.last_use = time.time()
        self.loaded_stamp = 0
        self.q = queue.Queue()
        self.gen = 0
        self.k = K   # raised while a kind:/in: filter will discard most hits
        self.exts = None   # a kind filter, applied inside the search
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._idle, daemon=True).start()
        self.q.put(("load", None, None))

    def index_stamp(self):
        p = expand(self.cfg["index"])
        return max((os.stat(f).st_mtime for f in (p, Path(str(p) + "-wal")) if os.path.exists(f)), default=0)

    def post(self, query):
        with self.lock:
            self.gen += 1
            gen = self.gen
        self.q.put(("search", gen, query))
        return gen

    def cancel(self):
        with self.lock:
            self.gen += 1

    def warm(self):
        """Start the query embedder ahead of the first keystroke (called when the window opens)."""
        self.last_use = time.time()
        if self.searcher is not None:
            self._ensure_embedder()

    def reload_if_stale(self):
        if self.searcher is not None and self.index_stamp() > self.loaded_stamp + 30:
            self.q.put(("load", None, None))

    def _status(self, text):
        AppHelper.callAfter(self.on_status, text)

    def _load(self):
        self._status("Loading index…")
        stamp = self.index_stamp()
        s = Searcher(self.cfg, embedder=None)
        old, self.searcher, self.loaded_stamp = self.searcher, s, stamp
        if old is not None:
            old.db.close()
        files = s.db.execute("SELECT count(*), max(indexed_at) FROM files").fetchone()
        self.info = {"files": files[0], "updated": files[1] or 0, "text": len(s.matrix), "media": len(s.media)}
        self._status(None)

    def _ensure_embedder(self):
        """The embedder if it is up; otherwise start it in the background and return None (keywords for now)."""
        if self.emb is not None:
            return self.emb
        if not self.emb_starting:
            self.emb_starting = True

            def start():
                try:
                    self.emb = Embedder(self.embed_cfg, log=lambda m: None)
                except Exception as e:
                    dbg("embedder failed:", e)
                finally:
                    self.emb_starting = False
                if self.emb is not None:
                    self.q.put(("rerun", None, None))
            threading.Thread(target=start, daemon=True).start()
        return None

    def _idle(self):
        while True:
            time.sleep(60)
            if self.emb is not None and time.time() - self.last_use > self.idle_stop:
                emb, self.emb = self.emb, None
                emb.close()

    def close(self):
        if self.emb is not None:
            self.emb.close()

    def _loop(self):
        last = None
        while True:
            kind, gen, query = self.q.get()
            if kind == "load":
                try:
                    self._load()
                except Exception as e:
                    self._status(f"Could not read the index: {e}")
                    continue
                if not last:
                    continue
                kind, gen, query = "search", self.gen, last[1]
            if kind == "rerun":
                if not last or last[0] != self.gen:
                    continue
                gen, query = last
            elif gen != self.gen:  # a newer query is already queued
                continue
            last = (gen, query)
            if self.searcher is None:
                continue
            self.last_use = time.time()
            t = time.time()
            emb = self._ensure_embedder()
            zero = np.zeros(self.cfg["model"]["dim"], np.float32)
            try:
                if query.startswith("like:"):
                    hits = self.searcher.similar(query[5:], 40, self.exts) or []
                else:
                    hits = self.searcher.search(query, "hybrid", self.k, exts=self.exts,
                                                qvec=emb.query(query) if emb is not None else zero)
            except Exception as e:  # the server went away mid-query: drop it and answer on keywords
                dbg("search fell back to keywords:", e)
                self.emb, emb = None, None
                hits = self.searcher.search(query, "hybrid", self.k, qvec=zero, exts=self.exts)
            info = {"ms": (time.time() - t) * 1000, "keywords_only": emb is None}
            AppHelper.callAfter(self.on_results, gen, query, hits, info)


class CodeEngine:
    """code: queries against codesearch.py's index, off the main thread; meaning joins once the embedder is up."""

    def __init__(self, cfg, engine, on_results):
        self.cfg, self.engine, self.on_results = cfg, engine, on_results
        self.searcher, self.stamp, self.gen = None, 0, 0
        self.q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def post(self, query):
        self.gen += 1
        self.q.put((self.gen, query))

    def _loop(self):
        while True:
            gen, query = self.q.get()
            if gen != self.gen:
                continue
            path = codesearch.index_path(self.cfg)
            if not path.exists():
                AppHelper.callAfter(self.on_results, gen, query, [])
                continue
            stamp = max(os.stat(f).st_mtime for f in (path, Path(str(path) + "-wal")) if os.path.exists(f))
            try:
                if self.searcher is None or stamp > self.stamp:
                    old, self.searcher, self.stamp = self.searcher, codesearch.CodeSearcher(self.cfg), stamp
                    if old is not None:
                        old.db.close()
                emb = self.engine._ensure_embedder()
                qvec = None
                if emb is not None and len(self.searcher.matrix):
                    try:
                        qvec = emb.query(query)
                    except Exception:
                        qvec = None
                hits = self.searcher.search(query, qvec)
            except Exception as e:
                dbg("code search failed:", e)
                hits = []
            AppHelper.callAfter(self.on_results, gen, query, hits)


_JB = "{scheme}://open?file={path}&line={line}"
EDITORS = (("/Applications/Cursor.app", "cursor://file{path}:{line}"),
           ("/Applications/Visual Studio Code.app", "vscode://file{path}:{line}"),
           (str(HOME / "Applications/Visual Studio Code.app"), "vscode://file{path}:{line}"),
           ("/Applications/Windsurf.app", "windsurf://file{path}:{line}"),
           ("/Applications/Zed.app", "zed://file{path}:{line}"),
           ("/Applications/Sublime Text.app", "subl://open?url=file://{path}&line={line}"),
           ("/Applications/PyCharm.app", _JB.replace("{scheme}", "pycharm")),
           ("/Applications/PyCharm CE.app", _JB.replace("{scheme}", "pycharm")),
           ("/Applications/IntelliJ IDEA.app", _JB.replace("{scheme}", "idea")),
           ("/Applications/IntelliJ IDEA CE.app", _JB.replace("{scheme}", "idea")),
           ("/Applications/WebStorm.app", _JB.replace("{scheme}", "webstorm")),
           ("/Applications/Xcode.app", "xed"))
PREFERRED_EDITOR = [None]   # [app name] from config [app] editor, set by the Controller and the Settings window


def editors():
    """[(app path, url format)] of the installed editors, the chosen one first."""
    have = [(a, f) for a, f in EDITORS if os.path.exists(a)]
    want = PREFERRED_EDITOR[0]
    return sorted(have, key=lambda e: os.path.basename(e[0])[:-4] != want) if want else have


def editor_name():
    have = editors()
    return os.path.basename(have[0][0])[:-4] if have else None


def open_in_editor(path, line):
    ws = AppKit.NSWorkspace.sharedWorkspace()
    for app, fmt in editors():
        if fmt == "xed":
            subprocess.Popen(["/usr/bin/xed", "-l", str(line), path], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        else:
            ws.openURL_(NSURL.URLWithString_(fmt.format(path=urllib.parse.quote(path), line=line)))
        return
    ws.openURL_(NSURL.fileURLWithPath_(path))


def play_at(full, t):
    """A video or recording at the matching moment: in IINA when it is installed (as DigUp does), else QuickTime."""
    if os.path.exists(IINA_CLI):
        subprocess.Popen([IINA_CLI, "--no-stdin", full, f"--mpv-start={t:.1f}"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return
    if ext_of(full) not in QUICKTIME_EXT:
        AppKit.NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(full))
        return
    script = ('on run argv\ntell application "QuickTime Player"\nactivate\n'
              'set d to open POSIX file (item 1 of argv)\nset current time of d to (item 2 of argv as real)\n'
              'end tell\nend run')
    subprocess.Popen(["/usr/bin/osascript", "-e", script, full, str(float(t))],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def path_rows(q):
    """Alfred-style browsing: "~/Doc" lists what in ~ starts with Doc; a trailing / lists the folder."""
    full = os.path.expanduser(q)
    folder, stem = (full, "") if full.endswith("/") or q == "~" else os.path.split(full)
    if not os.path.isdir(folder):
        return []
    sl = stem.lower()
    try:
        entries = [e for e in os.scandir(folder) if e.name.lower().startswith(sl)
                   and (sl.startswith(".") or not e.name.startswith("."))]
    except OSError:
        return []
    entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
    rows = [{"kind": "header", "title": short_path(folder)}] if entries else []
    return rows + [{"kind": "file", "path": e.path, "name": e.name, "dir": e.is_dir(follow_symlinks=False)}
                   for e in entries[:200]]


# --- files by name, from macOS's Spotlight index ----------------------------------------------------------------------

class Names:
    """mdfind -name; the newest query wins. Results come back through on_names(gen, query, paths)."""

    SKIP = ("/Library/", "/.Trash/", "/node_modules/", "/.git/", "/__pycache__/", "/.venv/", "/site-packages/")

    def __init__(self, on_names):
        self.on_names = on_names
        self.limit = 12
        self.gen = 0
        self.proc = None
        self.lock = threading.Lock()

    def post(self, q, limit=12, tag=None):
        """tag: every file carrying that Finder tag instead, answered under the query "#tag"."""
        self.limit = limit
        with self.lock:
            self.gen += 1
            gen = self.gen
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        if len(q.strip()) < 2 and not tag:
            AppHelper.callAfter(self.on_names, gen, q, [])
            return
        threading.Thread(target=self._run, args=(gen, q, tag), daemon=True).start()

    def _run(self, gen, q, tag=None):
        if tag:
            tag = re.sub(r"[\'\"\\*]", "", tag)
            cmd = ["/usr/bin/mdfind", "-onlyin", str(HOME), f"kMDItemUserTags == '{tag}*'cd"]
            q = "#" + tag
        else:
            cmd = ["/usr/bin/mdfind", "-onlyin", str(HOME), "-name", q.strip()]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        with self.lock:
            if gen != self.gen:
                proc.kill()
                return
            self.proc = proc
        paths = []
        t0 = time.time()
        home = str(HOME)
        for line in proc.stdout:
            p = line.rstrip("\n")
            rel = p[len(home):]
            if "/Library/CloudStorage/" not in p and any(s in rel for s in self.SKIP):
                continue
            if "/." in rel or ".app/" in p or p.endswith(".app"):
                continue
            paths.append(p)
            if len(paths) >= 400 or time.time() - t0 > 1.5:
                break
        proc.kill()
        if gen != self.gen:
            return
        ql = fold(q.strip())

        def rank(p):
            name = fold(os.path.basename(p))
            try:
                mtime = os.stat(p).st_mtime
            except OSError:
                mtime = 0
            return (0 if name.startswith(ql) else 1, "/Library/CloudStorage/" in p, -mtime)
        paths.sort(key=rank)
        AppHelper.callAfter(self.on_names, gen, q, paths[:self.limit])


# --- thumbnails ---------------------------------------------------------------------------------------------------------

class Thumbs:
    """Photo thumbnails and video frames at the hit's time, made off the main thread and cached."""

    def __init__(self):
        self.cache, self.q = {}, queue.LifoQueue()
        self.ws = AppKit.NSWorkspace.sharedWorkspace()
        threading.Thread(target=self._loop, daemon=True).start()

    def icon(self, path):
        return self.ws.iconForFile_(os.path.realpath(path))

    def get(self, full, t, kind, done, px=THUMB_PX):
        key = (full, t, px)
        if key in self.cache:
            return self.cache[key]
        if ext_of(full) in AUDIO_EXT:
            return None
        if kind in ("frame", "image", "audio") or is_media(full):
            self.q.put((key, full, t, kind, px, done))
        return None

    def _loop(self):
        while True:
            key, full, t, kind, px, done = self.q.get()
            if len(self.cache) > 1500:   # oldest first: dicts keep insertion order
                for k in list(self.cache)[:400]:
                    self.cache.pop(k, None)
            if key not in self.cache:
                try:
                    video = kind in ("frame", "audio") or ext_of(full) in VIDEO_EXT
                    self.cache[key] = self.frame(full, t or 0, px) if video else self.photo(full, px)
                except Exception:
                    self.cache[key] = None
            if self.cache[key] is not None:
                AppHelper.callAfter(done)

    @staticmethod
    def photo(full, px):
        src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(full), None)
        if src is None:
            return None
        opts = {Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
                Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
                Quartz.kCGImageSourceThumbnailMaxPixelSize: px}
        cg = Quartz.CGImageSourceCreateThumbnailAtIndex(src, 0, opts)
        return AppKit.NSImage.alloc().initWithCGImage_size_(cg, (0, 0)) if cg is not None else None

    @staticmethod
    def frame(full, t, px):
        asset = AVFoundation.AVURLAsset.URLAssetWithURL_options_(NSURL.fileURLWithPath_(full), None)
        gen = AVFoundation.AVAssetImageGenerator.assetImageGeneratorWithAsset_(asset)
        gen.setAppliesPreferredTrackTransform_(True)
        gen.setMaximumSize_((px, px))
        tol = CoreMedia.CMTimeMakeWithSeconds(1, 600)
        gen.setRequestedTimeToleranceBefore_(tol)
        gen.setRequestedTimeToleranceAfter_(tol)
        cg = gen.copyCGImageAtTime_actualTime_error_(CoreMedia.CMTimeMakeWithSeconds(t, 600), None, None)[0]
        return AppKit.NSImage.alloc().initWithCGImage_size_(cg, (0, 0)) if cg is not None else None


# --- views --------------------------------------------------------------------------------------------------------------

C = AppKit.NSColor


def font(size, weight=AppKit.NSFontWeightRegular):
    return AppKit.NSFont.systemFontOfSize_weight_(size, weight)


def symbol(name, size=16, weight=AppKit.NSFontWeightRegular):
    img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
    cfg = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(size, weight)
    return img.imageWithSymbolConfiguration_(cfg) if img is not None else None


def label(size, weight=AppKit.NSFontWeightRegular, color=None, mode=AppKit.NSLineBreakByTruncatingTail):
    f = AppKit.NSTextField.labelWithString_("")
    f.setFont_(font(size, weight))
    f.setTextColor_(color or C.labelColor())
    f.setLineBreakMode_(mode)
    f.setUsesSingleLineMode_(True)
    f.cell().setTruncatesLastVisibleLine_(True)
    return f


def marked(text, size, color, bold_color, wrap=False, highlight=False, weight=AppKit.NSFontWeightRegular):
    """Text with the «matched» words in a heavier weight (and, for the preview, on the find-highlight colour)."""
    out = AppKit.NSMutableAttributedString.alloc().init()
    heavy = AppKit.NSFontWeightBold if weight >= AppKit.NSFontWeightMedium else AppKit.NSFontWeightSemibold
    base = {AppKit.NSFontAttributeName: font(size, weight), AppKit.NSForegroundColorAttributeName: color}
    strong = {AppKit.NSFontAttributeName: font(size, heavy),
              AppKit.NSForegroundColorAttributeName: bold_color}
    if highlight:
        strong[AppKit.NSBackgroundColorAttributeName] = C.findHighlightColor()
        strong[AppKit.NSForegroundColorAttributeName] = C.blackColor()
    for i, part in enumerate(re.split(r"[«»]", text)):
        if part:
            out.appendAttributedString_(AppKit.NSAttributedString.alloc().initWithString_attributes_(
                part, strong if i % 2 else base))
    para = AppKit.NSMutableParagraphStyle.alloc().init()
    para.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping if wrap else AppKit.NSLineBreakByTruncatingTail)
    if wrap:
        para.setLineSpacing_(3)
    out.addAttribute_value_range_(AppKit.NSParagraphStyleAttributeName, para, NSMakeRange(0, out.length()))
    return out


def query_patterns(query):
    """Regex alternatives for the query's codes and words, longest first."""
    pats = [re.escape(c) for c in code_tokens(query)]
    pats += [re.escape(w) + r"\w*" for w in terms(query) if len(w) >= 3]
    return sorted(set(pats), key=len, reverse=True)


def highlight_terms(text, query):
    """Wrap the query's words and codes in «» wherever they occur in text (case ignored)."""
    pats = query_patterns(query)
    if not pats:
        return text
    rx = re.compile(r"(?<!\w)(?:" + "|".join(pats) + r")", re.IGNORECASE)
    return rx.sub(lambda m: "«" + m.group(0) + "»", text.replace("«", "").replace("»", ""))


class RowView(AppKit.NSTableRowView):
    """Spotlight's selection: a rounded accent-colour bar with white text, even though the field keeps the focus."""

    def drawSelectionInRect_(self, rect):
        path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            Foundation.NSInsetRect(self.bounds(), 8, 0), 8, 8)
        C.controlAccentColor().setFill()
        path.fill()

    def isEmphasized(self):
        return True


class Cell(AppKit.NSTableCellView):
    """One result row. Items are one line (title, then a dim kind or folder); hits add the matching snippet below."""

    def initWithWidth_(self, width):
        self = objc.super(Cell, self).initWithFrame_(NSMakeRect(0, 0, width, ROW_H["hit"]))
        if self is None:
            return None
        self.setIdentifier_("cell")
        self.icon = AppKit.NSImageView.alloc().initWithFrame_(NSMakeRect(6, 7, 32, 32))
        self.icon.setImageScaling_(AppKit.NSImageScaleProportionallyUpOrDown)
        self.icon.setWantsLayer_(True)
        self.icon.layer().setMasksToBounds_(True)
        self.title = label(13)
        self.line2 = AppKit.NSTextField.labelWithString_("")
        self.line2.setUsesSingleLineMode_(True)
        self.badge = label(11)
        self.badge.setFont_(AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(11, AppKit.NSFontWeightMedium))
        self.badge.setAlignment_(AppKit.NSTextAlignmentRight)
        for v in (self.icon, self.title, self.line2, self.badge):
            self.addSubview_(v)
        self.width = width
        self.text2 = ""
        self.title_text = ""
        self.mode = "hit"
        self.tinted = False
        return self

    @objc.python_method
    def setup(self, mode, title, text2, badge, tinted=False, indent=0):
        self.mode = mode
        w, h = self.width, ROW_H[mode]
        bw = min(150, 10 + 7 * len(badge)) if badge else 0
        if mode == "hit":
            self.icon.setFrame_(NSMakeRect(6, (h - 32) / 2, 32, 32))
            self.title.setFrame_(NSMakeRect(46, h - 24, w - 52 - bw, 17))
            self.line2.setFrame_(NSMakeRect(46, 8, w - 52, 15))
            self.badge.setFrame_(NSMakeRect(w - 6 - bw, h - 23, bw, 15))
        else:
            self.icon.setFrame_(NSMakeRect(9 + indent, (h - 22) / 2, 22, 22))
            self.title.setFrame_(NSMakeRect(40 + indent, (h - 17) / 2, w - 46 - bw - indent, 17))
            self.badge.setFrame_(NSMakeRect(w - 6 - bw, (h - 15) / 2, bw, 15))
        self.line2.setHidden_(mode != "hit")
        self.icon.layer().setCornerRadius_(4 if mode == "hit" else 0)
        self.title_text = title
        self.badge.setStringValue_(badge)
        self.text2 = text2
        self.tinted = tinted
        self.icon.setContentTintColor_(None)

    def setBackgroundStyle_(self, style):
        objc.super(Cell, self).setBackgroundStyle_(style)
        self.paint(style == AppKit.NSBackgroundStyleEmphasized)

    @objc.python_method
    def paint(self, selected):
        on = C.whiteColor()
        fg = on if selected else C.labelColor()
        dim = on.colorWithAlphaComponent_(0.78) if selected else C.secondaryLabelColor()
        if self.mode == "hit":
            self.title.setAttributedStringValue_(
                marked(self.title_text, 13, fg, fg, weight=AppKit.NSFontWeightMedium))
            self.line2.setAttributedStringValue_(marked(self.text2, 11.5, dim, fg))
        else:
            t = marked(self.title_text, 13, fg, fg)
            if self.text2:
                t.appendAttributedString_(marked("  –  " + self.text2, 12, dim, dim))
                para = AppKit.NSMutableParagraphStyle.alloc().init()
                para.setLineBreakMode_(AppKit.NSLineBreakByTruncatingTail)
                t.addAttribute_value_range_(AppKit.NSParagraphStyleAttributeName, para, NSMakeRange(0, t.length()))
            self.title.setAttributedStringValue_(t)
        self.badge.setTextColor_(dim)
        if self.tinted:
            self.icon.setContentTintColor_(on if selected else C.secondaryLabelColor())


class Header(AppKit.NSTableCellView):
    def initWithWidth_(self, width):
        self = objc.super(Header, self).initWithFrame_(NSMakeRect(0, 0, width, ROW_H["header"]))
        if self is None:
            return None
        self.setIdentifier_("header")
        self.text = label(11, AppKit.NSFontWeightSemibold, C.secondaryLabelColor())
        self.text.setFrame_(NSMakeRect(10, 4, width - 20, 15))
        self.addSubview_(self.text)
        return self


class GridView(AppKit.NSView, protocols=[objc.protocolNamed("NSDraggingSource")]):
    """Photos and videos as a grid of thumbnails (the Photos & Video scope, or a photo/video kind filter)."""

    def initWithController_(self, ctl):
        self = objc.super(GridView, self).initWithFrame_(NSMakeRect(0, 0, LIST_W, BODY_H))
        if self is None:
            return None
        self.ctl = ctl
        self.tiles = []
        self.down_row = None
        return self

    def isFlipped(self):
        return True

    def acceptsFirstMouse_(self, event):
        return True

    @objc.python_method
    def relayout(self, picks):
        gap, margin = 6, 12
        _, width, body = self.ctl.dims()
        cols = self.ctl.grid_cols()
        side = (width - 2 * margin - (cols - 1) * gap) / cols
        self.tiles = []
        for i, row in enumerate(picks):
            c, rr = i % cols, i // cols
            self.tiles.append((row, NSMakeRect(margin + c * (side + gap), margin + rr * (side + gap), side, side)))
        rows = (len(picks) + cols - 1) // cols
        self.setFrameSize_((width, max(body, 2 * margin + rows * (side + gap))))
        self.setNeedsDisplay_(True)

    @objc.python_method
    def rect_for(self, row):
        return next((rect for r, rect in self.tiles if r == row), None)

    def drawRect_(self, dirty):
        sel = self.ctl.table.selectedRow()
        for row, rect in self.tiles:
            if not Foundation.NSIntersectsRect(rect, dirty) or row >= len(self.ctl.rows):
                continue
            r = self.ctl.rows[row]
            path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, 8, 8)
            img = self.ctl.grid_image(row)
            AppKit.NSGraphicsContext.saveGraphicsState()
            path.addClip()
            C.quaternaryLabelColor().setFill()
            path.fill()
            if img is not None:
                w, h = img.size()
                if w and h:
                    k = min(w, h)
                    src = NSMakeRect((w - k) / 2, (h - k) / 2, k, k)   # fill the square, as Photos does
                    img.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(
                        rect, src, AppKit.NSCompositingOperationSourceOver, 1.0, True, None)
            else:
                icon = self.ctl.thumbs.icon(r["path"])
                d = rect.size.width * 0.5
                icon.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(
                    NSMakeRect(rect.origin.x + d / 2, rect.origin.y + d / 2, d, d), Foundation.NSZeroRect,
                    AppKit.NSCompositingOperationSourceOver, 1.0, True, None)
            video = r.get("hitkind") in ("frame", "audio") or ext_of(r["path"]) in VIDEO_EXT
            if video:
                text = ("▶ " + clock(r["t"])) if r.get("t") is not None and r.get("hitkind") else "▶"
                attrs = {AppKit.NSFontAttributeName: AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(
                    10, AppKit.NSFontWeightSemibold), AppKit.NSForegroundColorAttributeName: C.whiteColor()}
                s_ = AppKit.NSAttributedString.alloc().initWithString_attributes_(text, attrs)
                sw = s_.size().width
                pill = NSMakeRect(rect.origin.x + 5, rect.origin.y + rect.size.height - 20, sw + 10, 15)
                C.blackColor().colorWithAlphaComponent_(0.55).setFill()
                AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(pill, 7.5, 7.5).fill()
                s_.drawAtPoint_((pill.origin.x + 5, pill.origin.y + 1))
            AppKit.NSGraphicsContext.restoreGraphicsState()
            if row == sel:
                ring = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    Foundation.NSInsetRect(rect, 1.5, 1.5), 7, 7)
                ring.setLineWidth_(3)
                C.controlAccentColor().setStroke()
                ring.stroke()

    @objc.python_method
    def row_at(self, event):
        p = self.convertPoint_fromView_(event.locationInWindow(), None)
        return next((row for row, rect in self.tiles if Foundation.NSPointInRect(p, rect)), None)

    def mouseDown_(self, event):
        row = self.row_at(event)
        self.down_row = row
        if row is None:
            return
        self.ctl.navigated = True
        self.ctl.select(row)
        if event.clickCount() == 2:
            self.ctl.activate(row)

    def mouseDragged_(self, event):
        row, self.down_row = self.down_row, None
        full = self.ctl.target(row) if row is not None else None
        if not full or not os.path.exists(full):
            return
        item = AppKit.NSDraggingItem.alloc().initWithPasteboardWriter_(NSURL.fileURLWithPath_(full))
        item.setDraggingFrame_contents_(self.rect_for(row), self.ctl.grid_image(row) or self.ctl.thumbs.icon(full))
        self.beginDraggingSessionWithItems_event_source_([item], event, self)

    @objc.typedSelector(b"Q@:@q")
    def draggingSession_sourceOperationMaskForDraggingContext_(self, session, context):
        return AppKit.NSDragOperationCopy


class Panel(AppKit.NSPanel):
    def canBecomeKeyWindow(self):
        return True

    def canBecomeMainWindow(self):
        return True

    def performKeyEquivalent_(self, event):
        if self.delegate().keyEquivalent(event):
            return True
        return objc.super(Panel, self).performKeyEquivalent_(event)

    def cancelOperation_(self, sender):
        self.delegate().hide()

    # Quick Look: the panel takes control while this window is in the responder chain.
    @objc.typedSelector(b"Z@:@")
    def acceptsPreviewPanelControl_(self, panel):
        return True

    def beginPreviewPanelControl_(self, panel):
        panel.setDataSource_(self.delegate())
        panel.setDelegate_(self.delegate())

    def endPreviewPanelControl_(self, panel):
        pass


# --- preview pane -------------------------------------------------------------------------------------------------------

class Preview:
    """The right-hand pane: shows the selected row where it matched."""

    def __init__(self, parent, frame, thumbs):
        self.thumbs = thumbs
        self.view = AppKit.NSView.alloc().initWithFrame_(frame)
        w, h = frame.size.width, frame.size.height
        line = AppKit.NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, 1, h))
        line.setBoxType_(AppKit.NSBoxSeparator)
        self.view.addSubview_(line)
        mf = NSMakeRect(18, META_H, w - 36, h - META_H - 16)

        def rounded(v):
            v.setWantsLayer_(True)
            v.layer().setCornerRadius_(8)
            v.layer().setMasksToBounds_(True)
            return v

        self.image = rounded(AppKit.NSImageView.alloc().initWithFrame_(mf))
        self.image.setImageScaling_(AppKit.NSImageScaleProportionallyDown)

        self.big = label(40, AppKit.NSFontWeightLight)
        self.big.setFont_(AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(44, AppKit.NSFontWeightLight))
        self.big.setAlignment_(AppKit.NSTextAlignmentCenter)
        self.big.setFrame_(NSMakeRect(18, META_H + (h - META_H - 16) / 2 - 28, w - 36, 56))
        self.big.setSelectable_(True)

        self.pdf = rounded(Quartz.PDFView.alloc().initWithFrame_(mf))
        self.pdf.setAutoScales_(True)
        self.pdf.setDisplayMode_(Quartz.kPDFDisplaySinglePageContinuous)
        self.pdf.setBackgroundColor_(C.clearColor())

        self.player = rounded(AVKit.AVPlayerView.alloc().initWithFrame_(mf))
        self.player.setControlsStyle_(AVKit.AVPlayerViewControlsStyleInline)

        self.text_scroll = AppKit.NSScrollView.alloc().initWithFrame_(mf)
        self.text_scroll.setDrawsBackground_(False)
        self.text_scroll.setHasVerticalScroller_(True)
        self.text_scroll.setAutohidesScrollers_(True)
        tv = AppKit.NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, w - 36, 10))
        tv.setEditable_(False)
        tv.setDrawsBackground_(False)
        tv.setTextContainerInset_((2, 4))
        tv.setVerticallyResizable_(True)
        tv.setAutoresizingMask_(AppKit.NSViewWidthSizable)
        tv.textContainer().setWidthTracksTextView_(True)
        self.text = tv
        self.text_scroll.setDocumentView_(tv)

        self.ql = Quartz.QLPreviewView.alloc().initWithFrame_style_(mf, 0)
        self.ql.setAutostarts_(False)

        self.title = label(15, AppKit.NSFontWeightSemibold, mode=AppKit.NSLineBreakByTruncatingMiddle)
        self.title.setAlignment_(AppKit.NSTextAlignmentCenter)
        self.title.setFrame_(NSMakeRect(18, META_H - 38, w - 36, 20))
        split = int(w * 0.36)
        self.meta = []
        for i in range(5):
            y = META_H - 64 - i * 18
            k = label(11.5, color=C.secondaryLabelColor())
            k.setAlignment_(AppKit.NSTextAlignmentRight)
            k.setFrame_(NSMakeRect(18, y, split - 18, 15))
            v = label(11.5, mode=AppKit.NSLineBreakByTruncatingMiddle)
            v.setFrame_(NSMakeRect(split + 8, y, w - split - 26, 15))
            self.meta.append((k, v))
        self.note = AppKit.NSTextField.wrappingLabelWithString_("")
        self.note.setFont_(font(11))
        self.note.setTextColor_(C.tertiaryLabelColor())
        self.note.setAlignment_(AppKit.NSTextAlignmentCenter)
        self.note.setFrame_(NSMakeRect(18, 6, w - 36, 28))
        self.note.setMaximumNumberOfLines_(2)
        self.media = (self.image, self.big, self.pdf, self.player, self.text_scroll, self.ql)
        self.texts = (self.title, self.note) + tuple(x for pair in self.meta for x in pair)
        for v in self.media + self.texts:
            v.setHidden_(True)
            self.view.addSubview_(v)
        # The expanded window (⌘O) widens and heightens the pane: the media stretch, the lines below keep their place.
        grow, wide = AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable, AppKit.NSViewWidthSizable
        line.setAutoresizingMask_(AppKit.NSViewHeightSizable)
        for v in (self.image, self.pdf, self.player, self.text_scroll, self.ql):
            v.setAutoresizingMask_(grow)
        self.big.setAutoresizingMask_(wide | AppKit.NSViewMinYMargin | AppKit.NSViewMaxYMargin)
        for v in (self.title, self.note) + tuple(v for _, v in self.meta):
            v.setAutoresizingMask_(wide)
        parent.addSubview_(self.view)
        self.key = None

    def stop(self):
        p = self.player.player()
        if p is not None:
            p.pause()

    def clear(self):
        self.stop()
        self.player.setPlayer_(None)
        self.key = None
        for v in self.media + self.texts:
            v.setHidden_(True)

    def show_only(self, *views):
        for v in self.media:
            v.setHidden_(v not in views)
        if self.player not in views and self.player.player() is not None:
            self.stop()
            self.player.setPlayer_(None)

    def info(self, title, pairs, note=""):
        """The title, then up to five Kind / Size / Modified style lines, then a quiet note."""
        self.title.setStringValue_(title)
        self.title.setHidden_(False)
        pairs = [(k, v) for k, v in pairs if v]
        for i, (k, v) in enumerate(self.meta):
            show = i < len(pairs)
            k.setHidden_(not show)
            v.setHidden_(not show)
            if show:
                k.setStringValue_(pairs[i][0])
                v.setStringValue_(str(pairs[i][1]))
        self.note.setStringValue_(note)
        self.note.setHidden_(not note)

    def show(self, row, query, db):
        key = (row.get("kind"), row.get("path"), row.get("name"), row.get("page"), row.get("t"), row.get("value"),
               query)
        if key == self.key:
            return
        self.key = key
        kind = row["kind"]
        self.big.setFont_(AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(44, AppKit.NSFontWeightLight))
        if kind == "calc" and row.get("label"):
            self.show_only(self.big)
            self.big.setFont_(AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(
                44 if len(row["value"]) < 14 else 20, AppKit.NSFontWeightLight))
            self.big.setStringValue_(row["value"])
            self.info(row["expr"], [("Kind", row["label"])], "Press Return to copy it" if row["copy"] else "")
            return
        if kind == "link":
            self.show_only(self.image)
            self.image.setImage_(symbol("arrow.up.right.square", 96, AppKit.NSFontWeightUltraLight))
            self.image.setContentTintColor_(C.tertiaryLabelColor())
            self.info(row["name"], [("Opens", row["url"][:80])], "Only opened when you choose it")
            return
        if kind == "snippet":
            self.text.textStorage().setAttributedString_(
                marked(highlight_terms(row["value"], query) if query else row["value"], 13, C.labelColor(),
                       C.labelColor(), wrap=True, highlight=True))
            self.show_only(self.text_scroll)
            self.text.scrollToBeginningOfDocument_(None)
            self.info(row["name"], [("Kind", "Snippet"), ("Saved", ago(row["item"].get("at", time.time())))],
                      "Return copies it · ⌘⌫ forgets it")
            return
        if kind == "calc":
            self.show_only(self.big)
            self.big.setStringValue_(row["value"])
            self.info("Conversion" if row.get("conv") else "Calculator",
                      [("Input", row["expr"]), ("Result", row["value"])], "Press Return to copy " + row["copy"])
            return
        if kind == "setting":
            self.show_only(self.image)
            icon = self.thumbs.icon(SETTINGS_APP)
            icon.setSize_((192, 192))
            self.image.setImage_(icon)
            self.info(row["name"], [("Kind", "System Settings page")], "Press Return to open it")
            return
        if kind == "define":
            body = definition_layout(definition_gist(row["text"]))
            self.text.textStorage().setAttributedString_(
                marked(f"«{row['name']}»\n\n{body}", 13, C.labelColor(), C.labelColor(), wrap=True))
            self.show_only(self.text_scroll)
            self.text.scrollToBeginningOfDocument_(None)
            self.info(row["name"], [("Source", "Dictionary")], "Press Return to open it in Dictionary")
            return
        if kind == "action":
            self.show_only(self.image)
            self.image.setImage_(symbol(row["sym"], 96, AppKit.NSFontWeightUltraLight))
            self.image.setContentTintColor_(C.secondaryLabelColor())
            self.info(row["name"], [("Kind", "Quick action"), ("Does", row["does"])], "Press Return to run it")
            return
        if kind == "clip":
            it = row["item"]
            pairs = [("Kind", {"text": "Text", "image": "Image", "files": "Files"}[it["type"]]),
                     ("Copied", ago(it["at"])), ("From", it.get("app"))]
            if it["type"] == "image":
                self.image.setContentTintColor_(None)
                self.image.setImage_(it["image"])
                self.show_only(self.image)
            else:
                body = "\n".join(it["paths"]) if it["type"] == "files" else it["text"][:20000]
                self.text.textStorage().setAttributedString_(marked(
                    highlight_terms(body, query) if query else body, 13, C.labelColor(), C.labelColor(),
                    wrap=True, highlight=True))
                self.show_only(self.text_scroll)
                self.text.scrollToBeginningOfDocument_(None)
                if it["type"] == "text":
                    pairs.append(("Length", f"{len(it['text']):,} characters"))
            self.info("Clipboard", pairs, "Return copies it back · kept in memory only, gone when you quit")
            return
        if kind == "code":
            lines = row["text"].splitlines()
            width = len(str(row["start"] + len(lines)))
            body = "\n".join(f"{row['start'] + i:>{width}}  {ln}" for i, ln in enumerate(lines))
            body = highlight_terms(body, query) if query else body
            at = body.find("«")
            st = marked(body, 11.5, C.labelColor(), C.labelColor(), wrap=True, highlight=True)
            mono = AppKit.NSFont.monospacedSystemFontOfSize_weight_(11, AppKit.NSFontWeightRegular)
            st.addAttribute_value_range_(AppKit.NSFontAttributeName, mono, NSMakeRange(0, st.length()))
            self.text.textStorage().setAttributedString_(st)
            self.show_only(self.text_scroll)
            self.text.scrollToBeginningOfDocument_(None)
            if at > 300:
                self.text.scrollRangeToVisible_(NSMakeRange(at, 1))
            ed = editor_name()
            self.info(os.path.basename(row["path"]), [("Kind", "Code"), ("Lines", f"{row['start']}–{row['end']}"),
                                                       ("In", row.get("fn")),
                                                       ("Where", short_path(os.path.dirname(row["path"])))],
                      f"Return opens it in {ed} at line {row['line']}" if ed else f"Line {row['line']}")
            return
        if kind == "bookmark":
            self.show_only(self.image)
            app = row["app"] if os.path.exists(row["app"]) else "/Applications/Safari.app"
            icon = self.thumbs.icon(app)
            icon.setSize_((192, 192))
            self.image.setContentTintColor_(None)
            self.image.setImage_(icon)
            self.info(row["name"], [("Kind", f"{row['browser']} bookmark"), ("Address", row["url"]),
                                    ("Folder", row.get("folder"))],
                      f"Return opens it in {row['browser']} · ⌘C copies the address")
            return
        if kind == "emoji":
            self.big.setFont_(AppKit.NSFont.systemFontOfSize_(96))
            self.big.setStringValue_(row["value"])
            self.show_only(self.big)
            self.info(row["name"], [("Kind", "Emoji"), ("Code", row["code"])], "Return copies it, ready to paste")
            return
        if kind == "web":
            self.show_only(self.image)
            self.image.setImage_(symbol("globe", 96, AppKit.NSFontWeightUltraLight))
            self.image.setContentTintColor_(C.tertiaryLabelColor())
            self.info(f"Search the web for “{query}”", [("Opens", "your default browser")],
                      "The only row that leaves your Mac, and only when you choose it")
            return
        self.image.setContentTintColor_(None)
        full = row["path"]
        try:
            st = os.stat(full)
        except OSError:
            self.show_only()
            self.info(os.path.basename(full), [("Where", short_path(os.path.dirname(full)))],
                      "This file is no longer there")
            return
        if kind == "app":
            self.show_only(self.image)
            icon = self.thumbs.icon(full)
            icon.setSize_((192, 192))
            self.image.setImage_(icon)
            bundle = AppKit.NSBundle.bundleWithPath_(full)
            version = bundle.objectForInfoDictionaryKey_("CFBundleShortVersionString") if bundle else None
            self.info(row["name"], [("Kind", "Application"), ("Version", version), ("Where", short_path(full))])
            return

        ext = ext_of(full)
        hk = row.get("hitkind")
        ws = AppKit.NSWorkspace.sharedWorkspace()
        uti = ws.typeOfFile_error_(full, None)[0]
        kind_text = (ws.localizedDescriptionForType_(uti) if uti else None) or ext.upper() or "File"
        facts = [("Kind", kind_text), ("Size", size_text(st.st_size)), ("Modified", day(st.st_mtime))]
        note = ""
        if row.get("versions"):
            n = len(row["versions"])
            note = f"{n} other cop{'ies' if n > 1 else 'y'}: " + ", ".join(
                short_path(HOME / v) for v in row["versions"][:3]) + ("…" if n > 3 else "")
        facts.append(("Where", short_path(os.path.dirname(full))))

        if hk in ("frame", "audio") or ext in VIDEO_EXT | AUDIO_EXT:
            t = float(row.get("t") or 0)
            player = AVFoundation.AVPlayer.playerWithURL_(NSURL.fileURLWithPath_(full))
            player.seekToTime_toleranceBefore_toleranceAfter_(CoreMedia.CMTimeMakeWithSeconds(t, 600),
                                                             CoreMedia.kCMTimeZero, CoreMedia.kCMTimeZero)
            self.stop()
            self.player.setPlayer_(player)
            self.show_only(self.player)
            if hk == "frame":
                facts.insert(1, ("Frame", clock(t)))
            elif hk == "audio":
                facts.insert(1, ("Heard" if ext in AUDIO_EXT else "Sound", "from " + clock(t)))
            if ext in AUDIO_EXT and os.path.exists(IINA_CLI):
                note = note or "Return plays it in IINA from this moment"
            self.info(os.path.basename(full), facts, note)
            return
        if hk == "image" or ext in IMAGE_EXT:
            self.show_only(self.image)
            self.image.setImage_(self.thumbs.cache.get((full, row.get("t"), THUMB_PX)) or self.thumbs.icon(full))
            key_now = self.key

            def loaded():
                img = self.thumbs.cache.get((full, None, 1400))
                if self.key == key_now and img is not None:
                    self.image.setImage_(img)
            img = self.thumbs.get(full, None, "image", loaded, px=1400)
            if img is not None:
                self.image.setImage_(img)
            self.info(os.path.basename(full), facts, note)
            return
        if ext == "pdf":
            doc = Quartz.PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(full))
            if doc is not None and doc.pageCount() > 0:
                n = max(0, min((row.get("page") or 1) - 1, doc.pageCount() - 1))
                page = doc.pageAtIndex_(n)
                self.pdf.setDocument_(doc)
                self.pdf.goToPage_(page)
                found = []
                text = page.string() or ""
                pats = query_patterns(query)
                if pats and text:
                    rx = re.compile(r"(?<!\w)(?:" + "|".join(pats) + r")", re.IGNORECASE)
                    for m in list(rx.finditer(text))[:40]:
                        sel = page.selectionForRange_(NSMakeRange(m.start(), m.end() - m.start()))
                        if sel is not None:
                            sel.setColor_(C.findHighlightColor())
                            found.append(sel)
                self.pdf.setHighlightedSelections_(found or None)
                box = page.boundsForBox_(Quartz.kPDFDisplayBoxMediaBox)
                lo, hi = box.origin.y + box.size.height * 0.08, box.origin.y + box.size.height * 0.92

                def in_body(sel):   # running headers and footers repeat the title words on every page
                    y = sel.boundsForPage_(page).origin.y
                    return lo < y < hi
                body = [s for s in found if in_body(s)]
                if body:
                    self.pdf.goToSelection_(body[0])
                elif found:
                    self.pdf.goToSelection_(found[0])
                self.show_only(self.pdf)
                if row.get("page"):
                    facts.insert(1, ("Page", f"{row['page']} of {doc.pageCount()}"))
                self.info(os.path.basename(full), facts, note)
                return
        if row.get("chunk") and db is not None:
            r = db.execute("SELECT text FROM chunks WHERE id = ?", (row["chunk"],)).fetchone()
            if r:
                body = highlight_terms(re.sub(r"[ \t]+", " ", r[0]).strip(), query)
                plain_at = body.find("«")
                self.text.textStorage().setAttributedString_(
                    marked(body, 13, C.labelColor(), C.labelColor(), wrap=True, highlight=True))
                self.show_only(self.text_scroll)
                self.text.scrollToBeginningOfDocument_(None)
                if plain_at > 200:
                    self.text.scrollRangeToVisible_(NSMakeRange(plain_at, 1))
                if row.get("page"):
                    facts.insert(1, ("Page", row["page"]))
                self.info(os.path.basename(full), facts, note)
                return
        self.ql.setPreviewItem_(NSURL.fileURLWithPath_(full))
        self.show_only(self.ql)
        self.info(os.path.basename(full), facts, note)


# --- controller ---------------------------------------------------------------------------------------------------------

# --- settings window -----------------------------------------------------------------------------------------------------

def toml_set(text, section, key, value):
    """config.toml with `key = value` set inside [section] ("" is the top level), keeping every comment and line
    around it. value is already TOML."""
    lines = text.split("\n")
    cur, start, at = "", 0, None
    for i, ln in enumerate(lines):
        m = re.match(r"\s*\[([^\]]+)\]\s*$", ln)
        if m:
            if cur == section and at is None:
                break
            cur = m[1].strip()
            if cur == section:
                start = i + 1
            continue
        if cur == section and re.match(rf"\s*{re.escape(key)}\s*=", ln):
            at = i
            break
    if at is not None:
        tail = re.search(r"(\]|\"|\d)(\s+#.*)$", lines[at])
        lines[at] = f"{key} = {value}" + (tail[2] if tail else "")
    elif section == "" or any(re.match(rf"\s*\[{re.escape(section)}\]\s*$", ln) for ln in lines):
        lines.insert(start, f"{key} = {value}")
    else:
        lines += ["", f"[{section}]", f"{key} = {value}"]
    return "\n".join(lines)


def tilde(p):
    home = str(HOME)
    return "~" + p[len(home):] if p == home or p.startswith(home + "/") else p


class FolderList(NSObject):
    """One folder list of the Settings window: the table's data source."""

    def initWithItems_(self, items):
        self = objc.super(FolderList, self).init()
        if self is None:
            return None
        self.items = list(items)
        return self

    def numberOfRowsInTableView_(self, table):
        return len(self.items)

    def tableView_objectValueForTableColumn_row_(self, table, col, row):
        return self.items[row]


class Settings(NSObject):
    """Folders searched, media folders, code folders and the code editor, written back into config.toml."""

    def initWithController_(self, ctl):
        self = objc.super(Settings, self).init()
        if self is None:
            return None
        self.ctl = ctl
        cfg = ctl.cfg
        W, H = 620, 790
        style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable)
        win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), style, AppKit.NSBackingStoreBuffered, False)
        win.setTitle_("localsearch Settings")
        win.setReleasedWhenClosed_(False)
        self.win = win
        view = win.contentView()
        self.lists = []
        y = H - 20
        sections = (("Search inside these folders", "Documents, sheets, PDFs, slides, photos, video and audio",
                     [tilde(str(expand(r))) for r in cfg["roots"]]),
                    ("Photos, video and audio also from", "Media only: documents here are left out",
                     [tilde(str(expand(r))) for r in cfg.get("media", {}).get("roots", [])]),
                    ("Code search (code:)", "Code, notebooks and READMEs; .gitignore is respected",
                     [tilde(str(r)) for r in codesearch.code_roots(cfg)] or list(codesearch.DEFAULT_ROOTS)),
                    ("Never search these folders", "Left out of every search, and dropped from the index",
                     [tilde(str(expand(u))) for u in cfg["skip"]["under"]]))
        for i, (title, sub, items) in enumerate(sections):
            head = AppKit.NSTextField.labelWithString_(title)
            head.setFont_(AppKit.NSFont.boldSystemFontOfSize_(13))
            head.setFrame_(NSMakeRect(20, y - 18, W - 40, 18))
            view.addSubview_(head)
            note = AppKit.NSTextField.labelWithString_(sub)
            note.setFont_(AppKit.NSFont.systemFontOfSize_(11))
            note.setTextColor_(C.secondaryLabelColor())
            note.setFrame_(NSMakeRect(20, y - 34, W - 40, 15))
            view.addSubview_(note)
            box_h = 82
            scroll = AppKit.NSScrollView.alloc().initWithFrame_(NSMakeRect(20, y - 40 - box_h, W - 40, box_h))
            scroll.setBorderType_(AppKit.NSBezelBorder)
            scroll.setHasVerticalScroller_(True)
            t = AppKit.NSTableView.alloc().initWithFrame_(scroll.bounds())
            col = AppKit.NSTableColumn.alloc().initWithIdentifier_("p")
            col.setWidth_(W - 60)
            col.setEditable_(False)
            t.addTableColumn_(col)
            t.setHeaderView_(None)
            t.setUsesAlternatingRowBackgroundColors_(True)
            t.setAllowsMultipleSelection_(True)
            src = FolderList.alloc().initWithItems_(items)
            t.setDataSource_(src)
            scroll.setDocumentView_(t)
            view.addSubview_(scroll)
            seg = AppKit.NSSegmentedControl.segmentedControlWithImages_trackingMode_action_(
                [symbol("plus", 12), symbol("minus", 12)], AppKit.NSSegmentSwitchTrackingMomentary, "edit:")
            seg.setTarget_(self)
            seg.setTag_(i)
            seg.setFrame_(NSMakeRect(20, y - 40 - box_h - 26, 64, 22))
            view.addSubview_(seg)
            self.lists.append((t, src))
            y -= 40 + box_h + 26 + 16
        lab = AppKit.NSTextField.labelWithString_("Skip file types")
        lab.setFrame_(NSMakeRect(20, y - 22, 110, 18))
        view.addSubview_(lab)
        self.types = AppKit.NSTextField.alloc().initWithFrame_(NSMakeRect(130, y - 24, 260, 22))
        self.types.setPlaceholderString_("e.g. csv, txt")
        self.types.setStringValue_(", ".join(cfg["skip"].get("types", [])))
        view.addSubview_(self.types)
        y -= 36
        lab = AppKit.NSTextField.labelWithString_("Open code in")
        lab.setFrame_(NSMakeRect(20, y - 22, 110, 18))
        view.addSubview_(lab)
        self.editor = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(130, y - 26, 260, 26), False)
        names = [os.path.basename(a)[:-4] for a, _ in EDITORS if os.path.exists(a)]
        names = list(dict.fromkeys(names))
        self.editor.addItemsWithTitles_(names or ["No code editor found"])
        if PREFERRED_EDITOR[0] in names:
            self.editor.selectItemWithTitle_(PREFERRED_EDITOR[0])
        self.editor.setEnabled_(bool(names))
        view.addSubview_(self.editor)
        self.msg = AppKit.NSTextField.labelWithString_("Saving updates the index in the background.")
        self.msg.setFont_(AppKit.NSFont.systemFontOfSize_(11))
        self.msg.setTextColor_(C.secondaryLabelColor())
        self.msg.setFrame_(NSMakeRect(20, 22, W - 220, 15))
        view.addSubview_(self.msg)
        save = AppKit.NSButton.buttonWithTitle_target_action_("Save", self, "save:")
        save.setKeyEquivalent_("\r")
        save.setFrame_(NSMakeRect(W - 110, 14, 90, 30))
        view.addSubview_(save)
        cancel = AppKit.NSButton.buttonWithTitle_target_action_("Cancel", self, "cancel:")
        cancel.setKeyEquivalent_("\x1b")
        cancel.setFrame_(NSMakeRect(W - 206, 14, 90, 30))
        view.addSubview_(cancel)
        win.center()
        return self

    def show(self):
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self.win.makeKeyAndOrderFront_(None)
        self.win.orderFrontRegardless()

    def edit_(self, sender):
        t, src = self.lists[sender.tag()]
        if sender.selectedSegment() == 1:   # remove the selected folders
            rows = t.selectedRowIndexes()
            src.items = [p for i, p in enumerate(src.items) if not rows.containsIndex_(i)]
            t.reloadData()
            return
        panel = AppKit.NSOpenPanel.openPanel()
        panel.setCanChooseDirectories_(True)
        panel.setCanChooseFiles_(False)
        panel.setAllowsMultipleSelection_(True)
        panel.setPrompt_("Add")
        panel.setDirectoryURL_(NSURL.fileURLWithPath_(str(HOME)))
        if panel.runModal() == AppKit.NSModalResponseOK:
            for url in panel.URLs():
                p = tilde(url.path())
                if p not in src.items:
                    src.items.append(p)
            t.reloadData()

    def cancel_(self, sender):
        self.win.orderOut_(None)

    def save_(self, sender):
        docs, media, code, never = (src.items for _, src in self.lists)
        types = [t.strip().lower().lstrip(".") for t in re.split(r"[,\s]+", self.types.stringValue()) if t.strip()]
        if not docs:
            self.msg.setStringValue_("Keep at least one folder to search.")
            return
        path = HERE / "config.toml"
        text = path.read_text()
        arr = lambda xs: "[" + ", ".join(json.dumps(x) for x in xs) + "]"   # noqa: E731
        new = toml_set(text, "", "roots", arr(docs))
        new = toml_set(new, "media", "roots", arr(media))
        new = toml_set(new, "code", "roots", arr(code))
        new = toml_set(new, "skip", "under", arr(never))
        new = toml_set(new, "skip", "types", arr(types))
        if self.editor.isEnabled():
            new = toml_set(new, "app", "editor", json.dumps(self.editor.titleOfSelectedItem()))
        import tomllib
        try:
            tomllib.loads(new)
        except tomllib.TOMLDecodeError as e:
            self.msg.setStringValue_(f"Not saved: {e}")
            return
        (STATE / "config.toml.bak").write_text(text)
        path.write_text(new)
        self.ctl.reloadConfig()
        self.msg.setStringValue_("Saved. The index is updating in the background.")
        self.win.orderOut_(None)


class Controller(NSObject):
    def init(self):
        self = objc.super(Controller, self).init()
        if self is None:
            return None
        self.cfg = load_config()
        app = self.cfg.get("app", {})
        self.want_key = app.get("hotkey", "cmd+space")
        self.fallback_key = app.get("fallback_hotkey", "ctrl+option+space")
        self.history = History()
        self.apps = Apps(self.history)
        self.rows, self.hits, self.names = [], [], []
        self.query, self.scope = "", 0
        self.clean, self.exts, self.places = "", set(), []
        self.recents = []
        self.fresh = []
        self.used_apps = []
        self.pins = Pins()
        self.hits_for, self.names_for = None, None
        self.info = {}
        self.status_text = None
        self.debounce = None
        self.ql_opened = 0
        self.db = None
        self.thumbs = Thumbs()
        self.engine = Engine(self.cfg, self.onHits, self.setStatus)
        self.namer = Names(self.onNames)
        self.coder = CodeEngine(self.cfg, self.engine, self.onCode)
        self.code_hits, self.code_for = [], None
        self.clipboard = Clipboard()
        self.shortcuts = Shortcuts()
        self.bookmarks = Bookmarks()
        self.emoji = Emoji()
        self.snippets = Snippets()
        self.scratch = None
        self.timers, self.timer_ids = {}, 0
        self.armed = (None, 0.0)
        self.expanded = False
        self.settings_win = None
        PREFERRED_EDITOR[0] = app.get("editor")
        self.like_path = None
        self.last_menu = None
        self.mode = None
        self.navigated = False
        self.grid_mode = False
        self.awake_proc = None
        self.watcher = None
        if not DEMO:
            try:
                from watch import Watcher
                self.watcher = Watcher(self.cfg, self.onIndexUpdated,
                                       log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
                AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(15.0, True,
                                                                             lambda t: self.watcher.tick())
            except Exception as e:   # the window works without it; the nightly refresh still catches up
                dbg("watcher failed:", e)
        self.build()
        # Opening LocalSearch.app again (Finder, Launchpad, Spotlight) asks this copy to show its window.
        Foundation.NSDistributedNotificationCenter.defaultCenter().addObserver_selector_name_object_suspensionBehavior_(
            self, "remoteShow:", "com.kritin.localsearch.show", None,
            Foundation.NSNotificationSuspensionBehaviorDeliverImmediately)
        self.hotkey = HotKey(self.toggle)
        self.bindHotkey()
        AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(10.0, True, lambda t: self.bindHotkey())
        self.menu()
        return self

    @objc.python_method
    def bindHotkey(self):
        """cmd-space once Spotlight has let go of it; the fallback until then."""
        want = self.want_key
        if is_cmd_space(want) and spotlight_holds_cmd_space():
            want = self.fallback_key
        if want != self.hotkey.spec or not self.hotkey.ok:
            if not self.hotkey.rebind(want) and want != self.fallback_key:
                self.hotkey.rebind(self.fallback_key)
            self.refreshFooter()

    # build ---------------------------------------------------------------------------------------------------------
    @objc.python_method
    def build(self):
        style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
        h = FIELD_H + FOOT_H
        self.panel = Panel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIDTH, h), style, AppKit.NSBackingStoreBuffered, False)
        p = self.panel
        p.setLevel_(AppKit.NSModalPanelWindowLevel)
        p.setOpaque_(False)
        p.setBackgroundColor_(C.clearColor())
        p.setHasShadow_(True)
        p.setHidesOnDeactivate_(False)
        p.setMovableByWindowBackground_(True)
        p.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                 | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
                                 | AppKit.NSWindowCollectionBehaviorTransient)
        p.setDelegate_(self)

        frame = NSMakeRect(0, 0, WIDTH, h)
        if hasattr(AppKit, "NSGlassEffectView"):   # macOS 26+: the same glass Spotlight is made of
            bg = AppKit.NSGlassEffectView.alloc().initWithFrame_(frame)
            bg.setCornerRadius_(RADIUS)
            box = AppKit.NSView.alloc().initWithFrame_(frame)
            bg.setContentView_(box)
        else:
            bg = AppKit.NSVisualEffectView.alloc().initWithFrame_(frame)
            bg.setMaterial_(AppKit.NSVisualEffectMaterialPopover)
            bg.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
            bg.setState_(AppKit.NSVisualEffectStateActive)
            bg.setWantsLayer_(True)
            bg.layer().setCornerRadius_(RADIUS)
            bg.layer().setMasksToBounds_(True)
            box = bg
        p.setContentView_(bg)
        self.bg, self.box = bg, box

        self.glass = AppKit.NSImageView.imageViewWithImage_(symbol("magnifyingglass", 19, AppKit.NSFontWeightMedium))
        self.glass.setContentTintColor_(C.secondaryLabelColor())
        box.addSubview_(self.glass)

        f = AppKit.NSTextField.alloc().initWithFrame_(NSMakeRect(52, 0, WIDTH - 220, 30))
        f.setBezeled_(False)
        f.setDrawsBackground_(False)
        f.setFocusRingType_(AppKit.NSFocusRingTypeNone)
        f.setFont_(font(22))
        f.setPlaceholderString_("Search")
        f.cell().setScrollable_(True)
        f.cell().setUsesSingleLineMode_(True)
        f.setDelegate_(self)
        self.field = f
        box.addSubview_(f)

        # Spotlight's scope buttons at the end of the bar; Tab cycles them too.
        self.scope_buttons = []
        for scope, name, tip in ((3, "square.grid.2x2", "Applications"), (1, "doc.text", "Documents"),
                                 (2, "photo.on.rectangle", "Photos & Video"), (4, "doc.on.clipboard", "Clipboard")):
            b = AppKit.NSButton.buttonWithImage_target_action_(symbol(name, 14, AppKit.NSFontWeightMedium), self,
                                                               "scopeClicked:")
            b.setBordered_(False)
            b.setImagePosition_(AppKit.NSImageOnly)
            b.setTag_(scope)
            b.setRefusesFirstResponder_(True)
            b.setToolTip_(f"{tip} only (Tab)")
            b.setWantsLayer_(True)
            b.layer().setCornerRadius_(14)
            box.addSubview_(b)
            self.scope_buttons.append(b)
        self.paintScopes()

        self.spin = AppKit.NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(0, 0, 16, 16))
        self.spin.setStyle_(AppKit.NSProgressIndicatorStyleSpinning)
        self.spin.setControlSize_(AppKit.NSControlSizeSmall)
        self.spin.setDisplayedWhenStopped_(False)
        box.addSubview_(self.spin)

        self.sep = AppKit.NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, WIDTH, 1))
        self.sep.setBoxType_(AppKit.NSBoxSeparator)
        box.addSubview_(self.sep)

        self.scroll = AppKit.NSScrollView.alloc().initWithFrame_(NSMakeRect(0, FOOT_H, LIST_W, BODY_H))
        self.scroll.setDrawsBackground_(False)
        self.scroll.setHasVerticalScroller_(True)
        self.scroll.setAutohidesScrollers_(True)
        t = AppKit.NSTableView.alloc().initWithFrame_(self.scroll.bounds())
        t.setStyle_(AppKit.NSTableViewStyleInset)
        t.setHeaderView_(None)
        t.setBackgroundColor_(C.clearColor())
        t.setIntercellSpacing_((0, 1))
        t.setFocusRingType_(AppKit.NSFocusRingTypeNone)
        t.setRefusesFirstResponder_(True)
        col = AppKit.NSTableColumn.alloc().initWithIdentifier_("c")
        col.setWidth_(LIST_W - 22)
        t.addTableColumn_(col)
        t.setDataSource_(self)
        t.setDelegate_(self)
        t.setTarget_(self)
        t.setDoubleAction_("openClicked:")
        t.setDraggingSourceOperationMask_forLocal_(AppKit.NSDragOperationCopy, False)
        self.table = t
        self.scroll.setDocumentView_(t)
        box.addSubview_(self.scroll)
        self.grid_scroll = AppKit.NSScrollView.alloc().initWithFrame_(NSMakeRect(0, FOOT_H, LIST_W, BODY_H))
        self.grid_scroll.setDrawsBackground_(False)
        self.grid_scroll.setHasVerticalScroller_(True)
        self.grid_scroll.setAutohidesScrollers_(True)
        self.grid = GridView.alloc().initWithController_(self)
        self.grid_scroll.setDocumentView_(self.grid)
        box.addSubview_(self.grid_scroll)

        self.preview = Preview(box, NSMakeRect(LIST_W, FOOT_H, WIDTH - LIST_W, BODY_H), self.thumbs)

        self.foot_left = label(11, color=C.secondaryLabelColor())
        self.foot_left.setFrame_(NSMakeRect(26, 9, 400, 14))
        self.foot_right = label(11, color=C.tertiaryLabelColor())
        self.foot_right.setAlignment_(AppKit.NSTextAlignmentRight)
        self.foot_right.setFrame_(NSMakeRect(WIDTH - 26 - 420, 9, 420, 14))
        box.addSubview_(self.foot_left)
        box.addSubview_(self.foot_right)

        # The ⌘1-9 numbers appear only while ⌘ is held, as in Spotlight's menu-bar cousins.
        self.cmd_down = False

        def flags(event):
            down = bool(event.modifierFlags() & AppKit.NSEventModifierFlagCommand)
            if down != self.cmd_down:
                self.cmd_down = down
                self.redrawRows()
            return event
        self.flag_monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            AppKit.NSEventMaskFlagsChanged, flags)

        # Space opens Quick Look once you have moved into the results (before that it types a space).
        mods = (AppKit.NSEventModifierFlagCommand | AppKit.NSEventModifierFlagControl
                | AppKit.NSEventModifierFlagOption)

        def keys(event):
            if (event.keyCode() == 49 and self.navigated and not event.modifierFlags() & mods
                    and (self.ql_visible() or (self.panel.isKeyWindow()
                                               and self.target(self.table.selectedRow())))):
                self.quicklook()
                return None
            return event
        self.key_monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(AppKit.NSEventMaskKeyDown,
                                                                                        keys)
        self.layout(False)

    @objc.python_method
    def dims(self):
        """(window width, list or grid width, body height): Spotlight's size, or the expanded window's."""
        if self.expanded:
            return BIG[0], BIG[1] if self.grid_mode else LIST_W, BIG[2]
        return WIDTH, LIST_W, BODY_H

    @objc.python_method
    def grid_cols(self):
        return BIG_GRID_COLS if self.expanded else GRID_COLS

    @objc.python_method
    def layout(self, body):
        """The bar alone while there is nothing to show; the list, preview and footer grow down beneath it."""
        W, LW, BH = self.dims()
        h = FIELD_H + ((BH + FOOT_H) if body else 0)
        p = self.panel
        fr = p.frame()
        top = fr.origin.y + fr.size.height
        if fr.size.height != h or fr.size.width != W:
            x = fr.origin.x + (fr.size.width - W) / 2   # widen about the middle
            p.setFrame_display_(NSMakeRect(x, top - h, W, h), True)
        self.bg.setFrame_(NSMakeRect(0, 0, W, h))
        if self.box is not self.bg:
            self.box.setFrame_(NSMakeRect(0, 0, W, h))
        self.scroll.setFrame_(NSMakeRect(0, FOOT_H, LIST_W, BH))
        self.grid_scroll.setFrame_(NSMakeRect(0, FOOT_H, LW, BH))
        self.preview.view.setFrame_(NSMakeRect(LW, FOOT_H, W - LW, BH))
        self.foot_right.setFrame_(NSMakeRect(W - 26 - 520, 9, 520, 14))
        y0 = h - FIELD_H
        self.glass.setFrame_(NSMakeRect(22, y0 + (FIELD_H - 22) / 2, 22, 22))
        x = W - 16
        for b in reversed(self.scope_buttons):
            x -= 28
            b.setFrame_(NSMakeRect(x, y0 + (FIELD_H - 28) / 2, 28, 28))
            x -= 4
        self.spin.setFrame_(NSMakeRect(x - 22, y0 + (FIELD_H - 16) / 2, 16, 16))
        self.field.setFrame_(NSMakeRect(54, y0 + (FIELD_H - 30) / 2, x - 30 - 54, 30))
        self.sep.setFrame_(NSMakeRect(0, y0, W, 1))
        for v in (self.sep, self.preview.view, self.foot_left, self.foot_right):
            v.setHidden_(not body)
        self.scroll.setHidden_(not body or self.grid_mode)
        self.grid_scroll.setHidden_(not body or not self.grid_mode)

    @objc.python_method
    def paintScopes(self):
        for b in self.scope_buttons:
            on = b.tag() == self.scope
            b.layer().setBackgroundColor_((C.controlAccentColor() if on else C.clearColor()).CGColor())
            b.setContentTintColor_(C.whiteColor() if on else C.secondaryLabelColor())
        self.field.setPlaceholderString_({"Clipboard": "Search clipboard history", "Apps": "Search applications",
                                          "Documents": "Search documents",
                                          "Photos & Video": "Search photos and video"}.get(SCOPES[self.scope],
                                                                                           "Search"))

    @objc.python_method
    def redrawRows(self):
        rows = self.table.rowsInRange_(self.table.visibleRect())
        for i in range(rows.location, rows.location + rows.length):
            if i < len(self.rows) and self.rows[i]["kind"] != "header":
                self.table.reloadDataForRowIndexes_columnIndexes_(AppKit.NSIndexSet.indexSetWithIndex_(i),
                                                                 AppKit.NSIndexSet.indexSetWithIndex_(0))

    @objc.python_method
    def menu(self):
        bar = AppKit.NSStatusBar.systemStatusBar()
        self.item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_("doc.text.magnifyingglass",
                                                                                 "localsearch")
        img.setTemplate_(True)
        self.item.button().setImage_(img)
        m = AppKit.NSMenu.alloc().init()
        m.setDelegate_(self)
        m.setAutoenablesItems_(False)
        self.search_item = m.addItemWithTitle_action_keyEquivalent_("Search", "show:", "")
        self.search_item.setTarget_(self)
        m.addItem_(AppKit.NSMenuItem.separatorItem())
        self.state_item = m.addItemWithTitle_action_keyEquivalent_("", None, "")
        self.state_item.setEnabled_(False)
        self.progress_item = m.addItemWithTitle_action_keyEquivalent_("", None, "")
        self.progress_item.setEnabled_(False)
        self.watch_item = m.addItemWithTitle_action_keyEquivalent_("", None, "")
        self.watch_item.setEnabled_(False)
        self.pause_item = m.addItemWithTitle_action_keyEquivalent_("Pause indexing", "pauseIndexing:", "")
        self.pause_item.setTarget_(self)
        self.refresh_item = m.addItemWithTitle_action_keyEquivalent_("Update index now", "refreshIndex:", "")
        self.refresh_item.setTarget_(self)
        m.addItem_(AppKit.NSMenuItem.separatorItem())
        self.hotkey_item = m.addItemWithTitle_action_keyEquivalent_("", "hotkeyHelp:", "")
        self.hotkey_item.setTarget_(self)
        m.addItemWithTitle_action_keyEquivalent_("Settings…", "settings:", ",").setTarget_(self)
        m.addItemWithTitle_action_keyEquivalent_("Edit config.toml…", "editConfig:", "").setTarget_(self)
        m.addItem_(AppKit.NSMenuItem.separatorItem())
        m.addItemWithTitle_action_keyEquivalent_("Quit localsearch", "quit:", "q").setTarget_(self)
        self.item.setMenu_(m)

    # menu ----------------------------------------------------------------------------------------------------------
    @objc.python_method
    def indexer_pids(self):
        out = subprocess.run(["/usr/bin/pgrep", "-f", r"python indexer\.py"], capture_output=True, text=True).stdout
        return [int(x) for x in out.split()]

    @objc.python_method
    def indexer_paused(self, pids):
        out = subprocess.run(["/bin/ps", "-o", "stat=", "-p", ",".join(map(str, pids))],
                             capture_output=True, text=True).stdout
        return any(s.strip().startswith("T") for s in out.splitlines())

    @objc.python_method
    def indexer_progress(self):
        """The last progress line of the running refresh, in plain words."""
        try:
            with open(HERE / "logs/refresh.out", "rb") as f:
                f.seek(max(0, os.path.getsize(f.name) - 4000))
                lines = f.read().decode(errors="replace").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            if "DONE exit" in line:
                return None
            if "battery" in line and "power" in line:
                return "waiting for mains power"
            m = re.search(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)", line)
            if m and ("media" in line or "embed" in line or "extract" in line):
                stage = "photos and video" if "media" in line else "text" if "embed" in line else "reading files"
                return f"{stage}, {m[1]} of {m[2]}"
        return None

    def menuWillOpen_(self, menu):
        info = self.engine.info
        self.state_item.setTitle_(f"{info['files']:,} files · updated {ago(info['updated'])}" if info
                                  else "Index loading…")
        self.search_item.setTitle_(f"Search      {self.hotkey.label}")
        pids = self.indexer_pids()
        paused = bool(pids) and self.indexer_paused(pids)
        prog = self.indexer_progress() if pids else None
        if pids:
            self.progress_item.setTitle_(("Indexing paused" if paused else "Indexing") + (f": {prog}" if prog else ""))
        else:
            self.progress_item.setTitle_("Indexer idle")
        self.watch_item.setTitle_(self.watcher.status() if self.watcher else "Not watching folders")
        self.pause_item.setHidden_(not pids)
        self.pause_item.setTitle_("Resume indexing" if paused else "Pause indexing")
        self.refresh_item.setHidden_(bool(pids))
        free = not spotlight_holds_cmd_space()
        on_cmd_space = free and self.hotkey.label == "⌘Space"
        self.hotkey_item.setTitle_("⌘Space opens this search" if on_cmd_space
                                   else "Use ⌘Space instead of Spotlight…")
        self.hotkey_item.setEnabled_(not on_cmd_space)

    def pauseIndexing_(self, sender):
        pids = self.indexer_pids()
        if not pids:
            return
        sig = signal.SIGCONT if self.indexer_paused(pids) else signal.SIGSTOP
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    def refreshIndex_(self, sender):
        if not self.indexer_pids():
            subprocess.Popen(["/bin/zsh", str(HERE / "tools/refresh.sh")], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def hotkeyHelp_(self, sender):
        # Opens Keyboard Shortcuts in System Settings; moving Spotlight off cmd-space is the user's change to make.
        AppKit.NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(
            "x-apple.systempreferences:com.apple.Keyboard-Settings.extension?Shortcuts"))

    def remoteShow_(self, note):
        self.show()

    def settings_(self, sender):
        self.hide()
        if self.settings_win is None:
            self.settings_win = Settings.alloc().initWithController_(self)
        self.settings_win.show()

    def editConfig_(self, sender):
        AppKit.NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(HERE / "config.toml")))

    @objc.python_method
    def reloadConfig(self):
        """After the Settings window saves: the new folders are watched and indexed, the editor takes effect."""
        self.cfg = load_config()
        PREFERRED_EDITOR[0] = self.cfg.get("app", {}).get("editor")
        self.settings_win = None
        if self.watcher is not None:
            self.watcher.restart(self.cfg)
        self.refreshIndex_(None)

    def show_(self, sender):
        self.show()

    def quit_(self, sender):
        AppKit.NSApp.terminate_(None)

    # show / hide ---------------------------------------------------------------------------------------------------
    @objc.python_method
    def toggle(self):
        if self.panel.isVisible() and self.panel.isKeyWindow():
            self.hide()
        else:
            self.show()

    @objc.python_method
    def show(self):
        self.engine.reload_if_stale()
        self.engine.warm()
        self.shortcuts.refresh()
        self.bookmarks.refresh()
        self.navigated = False
        if not self.panel.isVisible():
            screen = AppKit.NSScreen.mainScreen() or AppKit.NSScreen.screens()[0]
            vf = screen.visibleFrame()
            fr = self.panel.frame()
            x = vf.origin.x + (vf.size.width - self.dims()[0]) / 2
            top = vf.origin.y + vf.size.height * 0.82
            self.panel.setFrameOrigin_((x, top - fr.size.height))
            if not self.query:
                self.rebuild()
            self.panel.setAlphaValue_(0.0)
        if not self.clean:
            self.loadRecents()
        # A non-activating panel takes key focus without activating the app, the way Spotlight's does; macOS 14+
        # no longer lets a background app activate itself.
        self.panel.orderFrontRegardless()
        self.panel.makeKeyWindow()

        def fade(ctx):
            ctx.setDuration_(0.12)
            self.panel.animator().setAlphaValue_(1.0)
        AppKit.NSAnimationContext.runAnimationGroup_completionHandler_(fade, None)
        self.panel.makeFirstResponder_(self.field)
        ed = self.field.currentEditor()
        if ed:
            ed.selectAll_(None)
        self.refreshFooter()
        AppHelper.callLater(0.15, lambda: dbg("show: key", self.panel.isKeyWindow()))

    @objc.python_method
    def hide(self):
        self.cmd_down = False
        self.preview.stop()
        if self.ql_visible():
            Quartz.QLPreviewPanel.sharedPreviewPanel().orderOut_(None)
        self.panel.orderOut_(None)

    def windowDidResignKey_(self, note):
        if not self.ql_visible() and time.time() - self.ql_opened > 1.0:
            dbg("resign key")
            self.preview.stop()
            self.panel.orderOut_(None)

    @staticmethod
    def ql_visible():
        return (Quartz.QLPreviewPanel.sharedPreviewPanelExists()
                and Quartz.QLPreviewPanel.sharedPreviewPanel().isVisible())

    # typing --------------------------------------------------------------------------------------------------------
    def controlTextDidChange_(self, note):
        self.query = self.field.stringValue().strip()
        self.mode, rest = prefix_mode(self.query)
        if self.mode == "like" and not (self.like_path and rest.strip() == os.path.basename(self.like_path)):
            self.mode, rest = None, self.query
        self.clean, self.exts, self.places = (parse_filters(rest) if self.mode not in ("code", "like", "path", "emoji",
                                                                                     "snip")
                                              else (rest.strip(), set(), []))
        self.navigated = False
        self.rebuild()   # apps and the calculator answer now; the slower layers follow
        if self.debounce:
            self.debounce.invalidate()
        self.debounce = AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            DEBOUNCE_S, self, "fire:", None, False)

    def fire_(self, timer):
        self.debounce = None
        q = self.clean
        dbg("query", repr(self.query))
        if self.mode == "code":
            if q:
                self.coder.post(q)
                self.spin.startAnimation_(None)
            return
        if self.mode in ("clip", "path", "emoji", "snip") or SCOPES[self.scope] == "Clipboard":
            return
        if self.mode == "name":
            self.engine.cancel()
            self.namer.post(q, limit=60)
            return
        if self.mode == "like":
            self.engine.k, self.engine.exts = 40, None
            self.engine.post("like:" + os.path.relpath(self.like_path, HOME))
            self.spin.startAnimation_(None)
            return
        if not q:
            self.engine.cancel()
            tag = tag_filter(self.places)
            self.namer.post("", limit=60, tag=tag)
            self.spin.stopAnimation_(None)
            self.hits, self.names, self.hits_for, self.names_for = [], [], None, None
            self.rebuild()
            return
        if SCOPES[self.scope] != "Apps" and not answers(q):
            self.engine.k = 90 if (self.exts or self.places or self.scope) else K
            self.engine.exts = set(self.exts) if self.exts else (
                IMAGE_EXT | VIDEO_EXT if SCOPES[self.scope] == "Photos & Video" else None)
            self.engine.post(q)
            self.spin.startAnimation_(None)
            self.namer.post(q)

    def scopeClicked_(self, sender):
        self.setScope(0 if self.scope == sender.tag() else sender.tag())

    def nextScope_(self, sender):
        self.setScope((self.scope + 1) % len(SCOPES))

    @objc.python_method
    def setScope(self, scope):
        self.scope = scope
        self.paintScopes()
        if self.clean and SCOPES[self.scope] not in ("Apps", "Clipboard") and self.mode is None:
            # The scope changes what the engine keeps, so search again.
            if True:
                self.engine.exts = set(self.exts) if self.exts else (
                    IMAGE_EXT | VIDEO_EXT if SCOPES[self.scope] == "Photos & Video" else None)
                self.engine.post(self.clean)
            if self.names_for != self.clean:
                self.namer.post(self.clean)
        self.rebuild()

    def control_textView_doCommandBySelector_(self, control, view, sel):
        sel = sel.decode() if isinstance(sel, bytes) else str(sel)
        if sel == "moveDown:":
            self.step(self.grid_cols() if self.grid_mode and self.navigated else 1)
        elif sel == "moveUp:":
            self.step(-self.grid_cols() if self.grid_mode else -1)
        elif sel in ("moveLeft:", "moveRight:") and self.grid_mode and self.navigated:
            self.step(-1 if sel == "moveLeft:" else 1)
        elif sel in ("scrollPageDown:", "moveToEndOfDocument:"):
            self.step(6)
        elif sel in ("scrollPageUp:", "moveToBeginningOfDocument:"):
            self.step(-6)
        elif sel == "insertTab:" and self.mode == "path":
            self.completePath()
        elif sel == "insertTab:":
            self.nextScope_(None)
        elif sel == "insertBacktab:":
            self.setScope((self.scope - 1) % len(SCOPES))
        elif sel == "insertNewline:":
            self.activate(self.table.selectedRow())
        elif sel in ("cancelOperation:", "complete:"):   # a text field turns escape into complete:
            if self.field.stringValue():
                self.field.setStringValue_("")
                self.controlTextDidChange_(None)
            else:
                self.hide()
        else:
            return False
        return True

    @objc.python_method
    def keyEquivalent(self, event):
        flags = event.modifierFlags() & AppKit.NSEventModifierFlagDeviceIndependentFlagsMask
        if not flags & AppKit.NSEventModifierFlagCommand:
            return False
        shift = flags & AppKit.NSEventModifierFlagShift
        ch = (event.charactersIgnoringModifiers() or "").lower()
        row = self.table.selectedRow()
        if ch in ("\r", "r") and not shift:
            self.reveal(row)
        elif ch == "y":
            self.quicklook()
        elif ch == "c" and shift:
            self.copy(row, path=True)
        elif ch == "c":
            ed = self.field.currentEditor()
            if ed and ed.selectedRange().length > 0:
                return False   # ordinary text copy
            self.copy(row, path=False)
        elif ch.isdigit() and ch != "0" and not shift:
            picks = self.picks()
            n = int(ch)
            if n <= len(picks):
                self.activate(picks[n - 1])
        elif ch == "w":
            self.hide()
        elif ch == "k" and not shift:
            self.actionMenu(row)
        elif ch == "o" and not shift:
            self.toggleExpanded()
        elif ch == ",":
            self.settings_(None)
        elif ch == "l" and not shift:
            self.findSimilar(row)
        elif ch == "p" and not shift:
            self.togglePin(row)
        elif ch in ("\x7f", "\x08") and 0 <= row < len(self.rows) and self.rows[row]["kind"] == "clip":
            self.clipboard.forget(self.rows[row]["item"])
            self.rebuild()
        elif ch in ("\x7f", "\x08") and 0 <= row < len(self.rows) and self.rows[row]["kind"] == "snippet":
            self.snippets.forget(self.rows[row]["item"])
            self.rebuild()
        else:
            return False
        return True

    @objc.python_method
    def completePath(self):
        """Tab in a path: take the selected entry into the field, a folder with its / so its contents list."""
        row = self.table.selectedRow()
        if not 0 <= row < len(self.rows) or not self.rows[row].get("path"):
            return
        r = self.rows[row]
        p = r["path"]
        home = str(HOME)
        text = ("~" + p[len(home):]) if p == home or p.startswith(home + "/") else p
        self.field.setStringValue_(text + ("/" if r.get("dir") else ""))
        self.controlTextDidChange_(None)

    @objc.python_method
    def toggleExpanded(self):
        self.expanded = not self.expanded
        self.rebuild()
        self.layout(bool(self.rows))
        if self.grid_mode:
            self.grid.relayout(self.picks())
        self.preview.key = None
        self.updatePreview()

    @objc.python_method
    def findSimilar(self, row):
        full = self.target(row)
        if not full or not os.path.exists(full) or not full.startswith(str(HOME) + "/"):
            return
        self.like_path = full
        self.field.setStringValue_("like:" + os.path.basename(full))
        self.controlTextDidChange_(None)

    @objc.python_method
    def actionMenu(self, row):
        """Every action for the selected row, as Raycast's and Alfred's action panels do."""
        if not 0 <= row < len(self.rows) or self.rows[row]["kind"] == "header":
            return
        r = self.rows[row]
        full = self.target(row)
        m = AppKit.NSMenu.alloc().initWithTitle_("Actions")
        m.setAutoenablesItems_(False)

        def add(menu, title, key, keq="", mods=AppKit.NSEventModifierFlagCommand, sym=None):
            it = menu.addItemWithTitle_action_keyEquivalent_(title, "menuAction:", keq)
            it.setTarget_(self)
            it.setRepresentedObject_(key)
            it.setKeyEquivalentModifierMask_(mods if keq else 0)
            if sym:
                it.setImage_(symbol(sym, 13))
            return it
        add(m, "Copy Back to Clipboard" if r["kind"] == "clip" else "Open", "open", "\r", 0, "arrow.up.forward.app")
        if full and os.path.exists(full):
            ws = AppKit.NSWorkspace.sharedWorkspace()
            url = NSURL.fileURLWithPath_(full)
            default = ws.URLForApplicationToOpenURL_(url)
            apps = [a for a in (ws.URLsForApplicationsToOpenURL_(url) or []) if a != default][:10]
            if apps:
                sub = AppKit.NSMenu.alloc().initWithTitle_("Open With")
                sub.setAutoenablesItems_(False)
                for a in apps:
                    it = add(sub, os.path.basename(a.path())[:-4], "with:" + a.path())
                    icon = ws.iconForFile_(a.path())
                    icon.setSize_((16, 16))
                    it.setImage_(icon)
                holder = m.addItemWithTitle_action_keyEquivalent_("Open With", None, "")
                holder.setImage_(symbol("square.on.square", 13))
                m.setSubmenu_forItem_(sub, holder)
            m.addItem_(AppKit.NSMenuItem.separatorItem())
            add(m, "Show in Finder", "reveal", "\r", sym="folder")
            add(m, "Quick Look", "ql", "y", sym="eye")
            if r["kind"] in ("hit", "file", "recent") and full.startswith(str(HOME) + "/"):
                add(m, "Find Similar Files", "similar", "l", sym="sparkle.magnifyingglass")
            m.addItem_(AppKit.NSMenuItem.separatorItem())
            add(m, "Copy File", "copy", "c", sym="doc.on.doc")
            add(m, "Copy Path", "path", "c", AppKit.NSEventModifierFlagCommand | AppKit.NSEventModifierFlagShift,
                sym="link")
            add(m, "Copy Name", "name", sym="textformat")
            add(m, "Unpin from Home" if self.pins.has(full) else "Pin to Home", "pin", "p", sym="pin")
            add(m, "Open Folder in Ghostty" if os.path.exists("/Applications/Ghostty.app") else "Open Folder in Terminal", "terminal", sym="terminal")
        if r["kind"] == "clip":
            m.addItem_(AppKit.NSMenuItem.separatorItem())
            add(m, "Forget This Item", "forget", "\x08", sym="trash")
            add(m, "Clear Clipboard History", "clearclip", sym="clear")
        if self.grid_mode and self.grid.rect_for(row) is not None:
            rect = self.grid.rect_for(row)
            view, point = self.grid, (rect.origin.x + 12, rect.origin.y + rect.size.height - 8)
        else:
            rect = self.table.rectOfRow_(row)
            view, point = self.table, (rect.origin.x + 40, rect.origin.y + rect.size.height - 4)
        self.last_menu = m
        m.popUpMenuPositioningItem_atLocation_inView_(None, point, view)

    def menuAction_(self, sender):
        key = sender.representedObject()
        row = self.table.selectedRow()
        full = self.target(row)
        if key == "open":
            self.activate(row)
        elif key == "pin":
            self.togglePin(row)
        elif key.startswith("with:") and full:
            self.hide()
            AppKit.NSWorkspace.sharedWorkspace().openURLs_withApplicationAtURL_configuration_completionHandler_(
                [NSURL.fileURLWithPath_(full)], NSURL.fileURLWithPath_(key[5:]),
                AppKit.NSWorkspaceOpenConfiguration.configuration(), None)
        elif key == "reveal":
            self.reveal(row)
        elif key == "ql":
            self.quicklook()
        elif key == "similar":
            self.findSimilar(row)
        elif key == "copy":
            self.copy(row, path=False)
        elif key == "path":
            self.copy(row, path=True)
        elif key == "name" and full:
            self.copy_text(os.path.basename(full))
            self.foot_left.setStringValue_("Name copied")
        elif key == "terminal" and full:
            self.hide()
            folder = full if os.path.isdir(full) and not full.endswith(".app") else os.path.dirname(full)
            term = "Ghostty" if os.path.exists("/Applications/Ghostty.app") else "Terminal"
            subprocess.Popen(["/usr/bin/open", "-a", term, folder])
        elif key == "forget" and 0 <= row < len(self.rows) and self.rows[row]["kind"] == "clip":
            self.clipboard.forget(self.rows[row]["item"])
            self.rebuild()
        elif key == "clearclip":
            self.clipboard.forget()
            self.rebuild()

    @objc.python_method
    def picks(self):
        return [i for i, r in enumerate(self.rows) if r["kind"] != "header"]

    @objc.python_method
    def step(self, d):
        picks = self.picks()
        if not picks:
            return
        self.navigated = True
        cur = self.table.selectedRow()
        if cur not in picks:
            self.select(picks[0])
            return
        i = picks.index(cur) + d
        self.select(picks[max(0, min(i, len(picks) - 1))])

    @objc.python_method
    def select(self, row):
        self.table.selectRowIndexes_byExtendingSelection_(AppKit.NSIndexSet.indexSetWithIndex_(row), False)
        self.table.scrollRowToVisible_(row)
        if row <= 2:
            self.table.scrollRowToVisible_(0)   # keep the section header above the first row in view
        if self.grid_mode:
            rect = self.grid.rect_for(row)
            if rect is not None:
                self.grid.scrollRectToVisible_(Foundation.NSInsetRect(rect, 0, -12))
            self.grid.setNeedsDisplay_(True)

    # results -------------------------------------------------------------------------------------------------------
    @objc.python_method
    def onHits(self, gen, query, hits, info):
        if gen != self.engine.gen:
            return
        self.spin.stopAnimation_(None)
        self.hits, self.hits_for, self.info = hits, self.clean if query.startswith("like:") else query, info
        self.rebuild()

    @objc.python_method
    def onNames(self, gen, query, paths):
        if gen != self.namer.gen:
            return
        self.names, self.names_for = paths, query
        self.rebuild()

    @objc.python_method
    def onCode(self, gen, query, hits):
        if gen != self.coder.gen:
            return
        self.spin.stopAnimation_(None)
        self.code_hits, self.code_for = hits, query
        self.rebuild()

    @objc.python_method
    def onIndexUpdated(self):
        self.engine.q.put(("load", None, None))

    @objc.python_method
    def grid_image(self, row):
        r = self.rows[row]
        full = r["path"]
        hk = r.get("hitkind") or ("frame" if ext_of(full) in VIDEO_EXT else "image")
        t = r.get("t") if r.get("hitkind") in ("frame", "audio") else (1.0 if hk == "frame" else None)
        return self.thumbs.get(full, t, hk, lambda: self.grid.setNeedsDisplay_(True), px=GRID_PX)

    @objc.python_method
    def setStatus(self, text):
        self.status_text = text
        self.refreshFooter()

    @objc.python_method
    def hit_row(self, h):
        return {"kind": "hit", "path": str(HOME / h["path"]), "name": os.path.basename(h["path"]),
                "page": h.get("page"), "t": h.get("t"), "hitkind": h.get("kind"), "snippet": h["snippet"],
                "versions": h["versions"], "chunk": h.get("chunk")}

    @objc.python_method
    def hit_rows(self, h):
        """A hit, then its other matching moments when it is a video."""
        r = self.hit_row(h)
        return [r] + [{**r, "t": m["t"], "hitkind": m["kind"], "sub": True, "versions": [], "snippet": ""}
                      for m in h.get("moments") or []]

    @objc.python_method
    def loadRecents(self):
        """What you opened lately, from macOS's last-used dates and from this window's own history."""
        history = [(h["path"], h.get("at", 0)) for h in self.history.items if h.get("kind") != "app" and h.get("path")]
        cfg = self.cfg

        def run():
            used = dict(history)
            try:
                out = subprocess.run(["/usr/bin/mdfind", "-onlyin", str(HOME), "-attr", "kMDItemLastUsedDate",
                                      "kMDItemLastUsedDate >= $time.today(-21)"],
                                     capture_output=True, text=True, timeout=6).stdout
            except (OSError, subprocess.SubprocessError):
                out = ""
            home = str(HOME)
            for line in out.splitlines():
                m = re.match(r"(.+?)\s+kMDItemLastUsedDate = (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4})$", line)
                if not m:
                    continue
                p, rel = m[1], m[1][len(home):]
                if "/." in rel or ".app" in p or ("/Library/" in rel and "/Library/CloudStorage/" not in p):
                    continue
                if ext_of(p) in RECENT_SKIP_EXT or any(d in rel for d in RECENT_SKIP_DIRS):
                    continue
                t = datetime.strptime(m[2], "%Y-%m-%d %H:%M:%S %z").timestamp()
                used[p] = max(t, used.get(p, 0))
            rows = []
            for p, t in sorted(used.items(), key=lambda x: -x[1]):
                if os.path.isfile(p):
                    rows.append({"kind": "file", "path": p, "name": os.path.basename(p), "used": t, "recent": True})
                if len(rows) >= 40:
                    break
            # Files that arrived in the indexed folders in the last three days: downloads, AirDrops, exports.
            fresh, per_dir = [], {}
            try:
                db = sqlite3.connect(f"file:{expand(cfg['index'])}?mode=ro", uri=True, timeout=2)
                found = db.execute("SELECT path, mtime FROM files WHERE status = 'ok' AND mtime >= ? "
                                   "ORDER BY mtime DESC LIMIT 200", (time.time() - 3 * 86400,)).fetchall()
                db.close()
            except sqlite3.Error:
                found = []
            for rel, mt in found:
                p, name = str(HOME / rel), os.path.basename(rel)
                d = os.path.dirname(rel)
                if (name.startswith(("~$", ".")) or p in used or ext_of(p) in RECENT_SKIP_EXT
                        or per_dir.get(d, 0) >= 2 or not os.path.isfile(p)):
                    continue
                per_dir[d] = per_dir.get(d, 0) + 1
                fresh.append({"kind": "file", "path": p, "name": name, "added": mt})
                if len(fresh) >= 12:
                    break
            AppHelper.callAfter(self.onRecents, rows, fresh, used_apps())
        threading.Thread(target=run, daemon=True).start()

    @objc.python_method
    def onRecents(self, rows, fresh=(), apps=()):
        changed = ([r["path"] for r in rows] != [r["path"] for r in self.recents]
                   or [r["path"] for r in fresh] != [r["path"] for r in self.fresh]
                   or [a["path"] for a in apps] != [a["path"] for a in self.used_apps])
        self.recents, self.fresh, self.used_apps = rows, list(fresh), list(apps)
        if changed and not self.clean:
            self.rebuild()

    @objc.python_method
    def home_rows(self, scope, exts, places):
        """The empty window: what you pinned, the apps you use most, what you had open, what just arrived."""
        filtered = bool(exts or places)
        everything = scope == "All" and not filtered

        def fits(r):
            if r["kind"] == "app":
                return scope in ("All", "Apps") and not filtered
            p = r["path"]
            if scope == "Apps" or not passes(p, exts, places):
                return False
            return not is_media(p) if scope == "Documents" else is_media(p) if scope == "Photos & Video" else True
        rows, shown = [], set()

        def section(title, items, n):
            items = [r for r in items if fits(r) and r["path"] not in shown][:n]
            if items:
                rows.append({"kind": "header", "title": title})
                rows.extend(items)
                shown.update(r["path"] for r in items)
        section("Favourites", self.pins.rows(), 12)
        if scope in ("All", "Apps") and not filtered:
            n = 4 if scope == "All" else 14
            apps, seen = [], set()
            for a in self.history.top_apps(n) + self.used_apps + dock_apps():
                if a["path"] not in seen and "LocalSearch" not in a["path"]:
                    seen.add(a["path"])
                    apps.append(a)
            section("Suggested", apps, n)
        section("Jump Back In", self.recents, 5 if everything else 10)
        section("New in Your Folders", self.fresh, 4 if everything else 8)
        if everything:
            quick = [{"kind": "action", "name": t, "act": a, "sym": s, "does": d}
                     for t, _, s, a, d in ACTIONS if a in ("scratchpad", "clipboard", "lock")]
            rows.append({"kind": "header", "title": "Quick Actions"})
            rows.extend(sorted(quick, key=lambda r: ("scratchpad", "clipboard", "lock").index(r["act"])))
        return rows

    @objc.python_method
    def rows_for(self):
        """Sections in a fixed order, so rows do not jump about as the slower layers arrive."""
        q, exts, places = self.clean, self.exts, self.places
        filtered = bool(exts or places)
        scope = SCOPES[self.scope]
        rows = []
        if scope == "Clipboard" or self.mode == "clip":
            items = self.clipboard.find(q)
            if items:
                rows.append({"kind": "header", "title": "Clipboard History"})
                rows += [{"kind": "clip", "item": it, "name": tidy(it["text"])[:200], "value": it["at"],
                          "path": it["paths"][0] if it["type"] == "files" else None} for it in items]
            return rows
        if self.mode == "path":
            return path_rows(self.query)
        if self.mode == "emoji":
            found = self.emoji.match(q)
            return ([{"kind": "header", "title": "Emoji"}] + found) if found else []
        if self.mode == "snip":
            found = self.snippets.find(q)
            return ([{"kind": "header", "title": "Snippets"}] + found) if found else []
        if self.mode == "code":
            fresh = self.code_for == q or (self.code_for and q.startswith(self.code_for))
            hits = self.code_hits if fresh and q else []
            if hits:
                rows.append({"kind": "header", "title": "Code"})
                rows += [{"kind": "code", "path": str(HOME / h["path"]), "name": os.path.basename(h["path"]),
                          "line": h["line"], "start": h["start"], "end": h["end"], "fn": h["name"],
                          "text": h["text"], "page": h["line"]} for h in hits]
            return rows
        if self.mode == "like":
            if self.hits_for == q:
                rows.append({"kind": "header", "title": f"Like {q}"})
                for h in self.hits:
                    rows += self.hit_rows(h)
            return rows
        if self.mode == "name":
            fresh = self.names_for == q or (self.names_for and q.startswith(self.names_for))
            names = [p for p in (self.names if fresh and q else []) if passes(p, exts, places)]
            if scope == "Documents":
                names = [p for p in names if not is_media(p)]
            elif scope == "Photos & Video":
                names = [p for p in names if is_media(p)]
            if names:
                rows.append({"kind": "header", "title": "File Names"})
                rows += [{"kind": "file", "path": p, "name": os.path.basename(p)} for p in names]
            return rows
        if not q and tag_filter(places):
            tag = tag_filter(places)
            tagged = [p for p in self.names if passes(p, exts, places)] if self.names_for == "#" + tag else []
            if tagged:
                rows.append({"kind": "header", "title": f"Tagged {tag.title()}"})
                rows += [{"kind": "file", "path": p, "name": os.path.basename(p)} for p in tagged]
            return rows
        if not q:
            return self.home_rows(scope, exts, places)

        calcs = [] if filtered or scope != "All" else [
            {"kind": "calc", "value": v, "expr": e, "copy": c, "conv": not calculate(q)} for v, e, c in answers(q)]
        apps = self.apps.match(q) if scope in ("All", "Apps") and not filtered else []
        sets = match_settings(q) if scope in ("All", "Apps") and not filtered else []
        acts = (match_actions(q, self.awake_proc is not None and self.awake_proc.poll() is None,
                              self.shortcuts.names, self.timers)
                if scope in ("All", "Apps") and not filtered else [])
        marks = self.bookmarks.match(q) if scope == "All" and not filtered and not calcs else []
        facts = [] if filtered or scope != "All" or calcs else [
            {"kind": "calc", "value": v, "expr": e, "copy": c, "label": "Answer"} for v, e, c in instant(q)]
        links = [] if filtered or scope != "All" else quicklink_rows(q, self.cfg.get("quicklinks"))
        # While a new deep search runs, keep showing the last one if the query only grew (no flicker as you type).
        fresh = self.hits_for == q or (self.hits_for and q.startswith(self.hits_for))
        hits = list(self.hits) if fresh and scope != "Apps" and not calcs else []
        codes = [fold(c) for c in code_tokens(q)]
        if codes:   # a code typed exactly: when files contain it, the near-misses below it are noise
            exact = [h for h in hits if all(c in fold(h["snippet"].replace("«", "").replace("»", "")) for c in codes)]
            hits = exact or hits
        elif hits and not filtered:   # meaning scores trail off into the unrelated; stop well below the best match
            cut = max(4.0, 0.55 * hits[0]["score"])
            hits = [h for i, h in enumerate(hits) if h["score"] >= cut or i < 3]
        hits = [h for h in hits if passes(str(HOME / h["path"]), exts, places)]
        if scope == "Documents":
            hits = [h for h in hits if not h.get("kind") and not is_media(h["path"])]
        elif scope == "Photos & Video":
            hits = [h for h in hits if h.get("kind") or is_media(h["path"])]
        shown = {str(HOME / h["path"]) for h in hits}
        names = []
        if scope != "Apps" and not calcs and (self.names_for == q or (self.names_for and q.startswith(self.names_for))):
            names = [p for p in self.names if p not in shown and passes(p, exts, places)]
        if scope == "Documents":
            names = [p for p in names if not is_media(p)]
        elif scope == "Photos & Video":
            names = [p for p in names if is_media(p)]
        text = define(q) if scope == "All" and not filtered and not calcs else None

        if facts:
            rows += [{"kind": "header", "title": "Answer"}] + facts
        if links:
            rows += [{"kind": "header", "title": "Quicklink"}] + links
        if calcs:
            rows.append({"kind": "header", "title": "Calculator" if calculate(q) else "Conversion"})
            rows += calcs
        elif facts or links:
            pass
        else:
            top = None
            if apps and apps[0]["score"] >= 95 - min(len(q), 4) * 5:
                top = [{"kind": "app", **apps.pop(0)}]
            elif acts and acts[0]["score"] >= 90 and len(q) >= 3:
                top = [acts.pop(0)]
            elif sets and sets[0]["score"] >= 95:
                top = [sets.pop(0)]
            elif marks and marks[0]["score"] >= 100 and len(q) >= 3 and not hits:
                top = [marks.pop(0)]
            elif hits:
                top = self.hit_rows(hits.pop(0))
            elif names and len(q) >= 3:
                top = [{"kind": "file", "path": names[0], "name": os.path.basename(names[0])}]
                names = names[1:]
            if top:
                rows += [{"kind": "header", "title": "Top Hit"}] + top
        if apps:
            rows.append({"kind": "header", "title": "Applications"})
            rows += [{"kind": "app", **a} for a in apps[:3]]
        if acts:
            rows.append({"kind": "header", "title": "Actions"})
            rows += acts
        if sets:
            rows.append({"kind": "header", "title": "System Settings"})
            rows += sets
        if marks:
            rows.append({"kind": "header", "title": "Bookmarks"})
            rows += marks
        docs = [h for h in hits if not h.get("kind")]
        sounds = [h for h in hits if h.get("kind") and ext_of(h["path"]) in AUDIO_EXT]
        media = [h for h in hits if h.get("kind") and ext_of(h["path"]) not in AUDIO_EXT]
        if docs:
            rows.append({"kind": "header", "title": "Documents"})
            rows += [self.hit_row(h) for h in docs[:14]]
        if media:
            rows.append({"kind": "header", "title": "Photos & Video"})
            for h in media[:40 if self.wants_grid() else 10]:
                rows += self.hit_rows(h)
        if sounds:
            rows.append({"kind": "header", "title": "Audio"})
            for h in sounds[:8]:
                rows += self.hit_rows(h)
        if text:
            rows += [{"kind": "header", "title": "Definition"}, {"kind": "define", "name": q, "text": text}]
        if names:
            rows.append({"kind": "header", "title": "Names"})
            rows += [{"kind": "file", "path": p, "name": os.path.basename(p)}
                     for p in names[:24 if self.wants_grid() else 8]]
        if scope == "All" and not filtered:
            rows += [{"kind": "header", "title": "Web"}, {"kind": "web"}]
        return rows

    @objc.python_method
    def rebuild(self):
        prev = self.table.selectedRow()
        prev_key = self.row_key(self.rows[prev]) if 0 <= prev < len(self.rows) else None
        picks = self.picks()
        prev_was_first = bool(picks) and prev == picks[0]
        self.rows = self.rows_for()
        self.table.reloadData()
        picks = self.picks()
        self.grid_mode = self.wants_grid() and bool(picks) and all(
            self.rows[i].get("path") and is_media(self.rows[i]["path"]) for i in picks)
        if self.grid_mode:
            self.grid.relayout(picks)
        self.layout(bool(self.rows))
        if picks:
            keep = None
            if prev_key and not prev_was_first:   # keep a row the user moved to; the top row follows the best hit
                keep = next((i for i in picks if self.row_key(self.rows[i]) == prev_key), None)
            self.select(keep if keep is not None else picks[0])
            self.updatePreview()
        else:
            self.preview.clear()
        self.refreshFooter()

    @objc.python_method
    def wants_grid(self):
        if self.mode == "like":
            return bool(self.like_path) and is_media(self.like_path)
        return self.mode in (None, "name") and (SCOPES[self.scope] == "Photos & Video" or bool(
            self.exts and self.exts <= IMAGE_EXT | VIDEO_EXT))

    @staticmethod
    def row_key(r):
        return (r["kind"], r.get("path"), r.get("name"), r.get("page"), r.get("t"), r.get("value"))

    @objc.python_method
    def refreshFooter(self):
        if getattr(self, "hotkey", None) is None:
            return
        hints = "↵ Open    ⌘↵ Show in Finder    ⌘C Copy    ␣ Quick Look    ⇥ Filter"
        if self.grid_mode:
            hints = "↵ Open    ←→↑↓ Move    ␣ Quick Look    ⌘K Actions    Drag to share"
        elif not self.clean and self.rows:
            hints = "↵ Open    ⌘P Pin to Home    ⌘K Actions    Try code:, kind:pdf, in:Documents"
        if self.expanded and self.rows:
            hints = hints.replace("⇥ Filter", "⌘O Smaller") if "⇥ Filter" in hints else hints + "    ⌘O Smaller"
        if self.mode == "path":
            hints = "⇥ Complete    ↵ Open    ⌘↵ Show in Finder    ␣ Quick Look"
        elif self.mode == "emoji":
            hints = "↵ Copy    ⌘1-9 Copy that one"
        elif self.mode == "code":
            ed = editor_name()
            hints = f"↵ Open in {ed} at the line    ⌘↵ Show in Finder" if ed else "↵ Open    ⌘↵ Show in Finder"
        elif self.mode == "clip" or SCOPES[self.scope] == "Clipboard":
            hints = "↵ Copy back    ⌘⌫ Forget    ⌘K Actions"
        elif self.rows and not self.grid_mode and self.clean:
            hints = "↵ Open    ⌘↵ Show in Finder    ␣ Quick Look    ⌘K Actions    ⇥ Filter"
        waiting = is_cmd_space(self.want_key) and self.hotkey.label != "⌘Space"
        if self.status_text:
            left = self.status_text
        elif not self.clean and self.mode is None and SCOPES[self.scope] != "Clipboard":
            info = self.engine.info
            left = f"{info['files']:,} files indexed · updated {ago(info['updated'])}" if info else "Loading index…"
            if not self.rows:
                hints = (f"{self.hotkey.label} now · ⌘Space once Spotlight's shortcut is moved" if waiting
                         else "Meaning, exact codes, scans, photos and video")
        elif self.mode == "path":
            n = sum(1 for r in self.rows if r["kind"] == "file")
            left = f"{n} items" if n else "Nothing there by that name"
        elif self.mode == "emoji":
            n = sum(1 for r in self.rows if r["kind"] == "emoji")
            left = f"{n} emoji" if n else "No emoji by that name" if self.clean else "Type what the emoji shows"
        elif self.mode in ("code", "name") and not self.clean:
            left = "Type what the code does or a name in it" if self.mode == "code" else "Type part of a file name"
        elif self.mode == "code":
            n = sum(1 for r in self.rows if r["kind"] == "code")
            left = (f"{n} places in your code" if n else "Nothing in your code" if self.code_for == self.clean
                    else "Searching code…")
        elif self.mode == "like":
            n = sum(1 for r in self.rows if r["kind"] == "hit" and not r.get("sub"))
            left = (f"{n} files like it" if n else "Not in the index yet, so nothing to compare"
                    if self.hits_for == self.clean else "Looking for files like it…")
        elif self.mode == "name":
            n = sum(1 for r in self.rows if r["kind"] == "file")
            left = f"{n} file names" if n else "No file names match" if self.names_for == self.clean else "Searching…"
        elif self.mode == "clip" or SCOPES[self.scope] == "Clipboard":
            n = sum(1 for r in self.rows if r["kind"] == "clip")
            left = f"{n} copied since localsearch started" if n else "Nothing copied yet"
        else:
            n = sum(1 for r in self.rows if r["kind"] == "hit")
            same = self.hits_for == self.clean
            ms = self.info.get("ms", 0) if same else 0
            if any(r["kind"] == "calc" for r in self.rows):
                left = "↵ copies the result"
            elif n:
                left = f"{n} found inside your files" + (f" · {ms:.0f} ms" if ms else "")
            elif same:
                left = "Nothing inside your files. Try fewer words, or a code exactly as written."
            else:
                left = "Searching…"
            if same and self.info.get("keywords_only"):
                left += " · words only while meaning starts"
            if self.exts or self.places:
                kinds = ("videos" if self.exts <= VIDEO_EXT else "photos" if self.exts <= IMAGE_EXT
                         else "photos and videos" if self.exts <= IMAGE_EXT | VIDEO_EXT
                         else ", ".join(sorted(self.exts)[:3])) if self.exts else ""
                span = next((p for p in self.places if isinstance(p, tuple) and p[0] != "tag"), None)
                when = f"changed {datetime.fromtimestamp(span[0]):%-d %b %Y} on" if span else ""
                tag = tag_filter(self.places)
                label = " ".join(x for x in (f"Only {kinds}" if kinds else "", f"tagged {tag}" if tag else "", when)
                                 if x)
                left = (label[0].upper() + label[1:] + " · " if label else "Filtered · ") + left
        self.foot_left.setStringValue_(left)
        self.foot_right.setStringValue_(hints)

    # table ---------------------------------------------------------------------------------------------------------
    def numberOfRowsInTableView_(self, table):
        return len(self.rows)

    def tableView_heightOfRow_(self, table, row):
        r = self.rows[row]
        if r["kind"] == "header":
            return ROW_H["header"]
        return ROW_H["hit"] if (r["kind"] == "hit" and not r.get("recent") and not r.get("sub")) or r["kind"] == "code" \
            else ROW_H["item"]

    def tableView_rowViewForRow_(self, table, row):
        return RowView.alloc().init()

    @objc.typedSelector(b"@@:@q")
    def tableView_pasteboardWriterForRow_(self, table, row):
        full = self.target(row)
        return NSURL.fileURLWithPath_(full) if full and os.path.exists(full) else None

    def tableView_shouldSelectRow_(self, table, row):
        return self.rows[row]["kind"] != "header"

    def tableView_viewForTableColumn_row_(self, table, col, row):
        r = self.rows[row]
        if r["kind"] == "header":
            v = table.makeViewWithIdentifier_owner_("header", self) or Header.alloc().initWithWidth_(col.width())
            v.text.setStringValue_(r["title"])
            return v
        cell = table.makeViewWithIdentifier_owner_("cell", self) or Cell.alloc().initWithWidth_(col.width())
        kind = r["kind"]
        picks = self.picks()
        n = picks.index(row) + 1 if row in picks else 0
        shortcut = f"⌘{n}" if n <= 9 and self.cmd_down else ""
        if kind == "calc" and r.get("label"):
            cell.setup("item", r["value"], r["expr"], shortcut, tinted=True)
            img = symbol("info.circle.fill", 19)
        elif kind == "link":
            cell.setup("item", r["name"], r["site"], shortcut, tinted=True)
            img = symbol("arrow.up.right.square.fill", 19)
        elif kind == "snippet":
            cell.setup("item", mark_words(r["name"], self.clean), tidy(r["value"])[:120], shortcut, tinted=True)
            img = symbol("text.quote", 17)
        elif kind == "calc":
            cell.setup("item", f"= {r['value']}", r["expr"], shortcut, tinted=True)
            img = symbol("equal.circle.fill", 20)
        elif kind == "web":
            cell.setup("item", f"Search the web for “{self.query}”", "", shortcut, tinted=True)
            img = symbol("safari", 20)
        elif kind == "bookmark":
            host = urllib.parse.urlsplit(r["url"]).netloc.removeprefix("www.")
            cell.setup("item", mark_words(r["name"], self.clean), f"{host} · {r['browser']}", shortcut)
            img = self.thumbs.icon(r["app"]) if os.path.exists(r["app"]) else symbol("bookmark", 17)
        elif kind == "emoji":
            cell.setup("item", mark_words(r["name"], self.clean), "Emoji", shortcut)
            img = emoji_image(r["value"])
        elif kind == "app":
            cell.setup("item", mark_words(r["name"], self.clean),
                       "Application" if self.clean or not self.history.count(r["path"])
                       else f"Opened {self.history.count(r['path'])} times" if self.history.count(r["path"]) > 1
                       else "Opened once", shortcut)
            img = self.thumbs.icon(r["path"])
        elif kind == "setting":
            cell.setup("item", mark_words(r["name"], self.clean), "System Settings", shortcut)
            img = self.thumbs.icon(SETTINGS_APP)
        elif kind == "define":
            cell.setup("item", r["name"], definition_gist(r["text"])[:200], shortcut, tinted=True)
            img = symbol("character.book.closed", 19)
        elif kind == "action":
            cell.setup("item", r["name"], r.get("label", "Action"), shortcut, tinted=r["act"] != "quit")
            img = self.thumbs.icon(r["app_path"]) if r.get("app_path") else symbol(r["sym"], 17)
        elif kind == "clip":
            it = r["item"]
            cell.setup("item", r["name"] or " ", ago(it["at"]) + (f" · {it['app']}" if it.get("app") else ""),
                       shortcut, tinted=it["type"] == "text")
            img = (it["image"] if it["type"] == "image" else self.thumbs.icon(it["paths"][0])
                   if it["type"] == "files" else symbol("text.alignleft", 16))
        elif kind == "code":
            line = next((ln.strip() for ln in r["text"].splitlines()
                         if any(w in fold(ln) for w in fold(self.clean).split())), r["text"].splitlines()[0].strip())
            folder = short_path(os.path.dirname(r["path"]))
            cell.setup("hit", mark_words(r["name"], self.clean),
                       mark_words(line[:200], self.clean) + f"  ·  {folder}", f"{r['line']}" + (
                           f"  {shortcut}" if shortcut else ""))
            img = self.thumbs.icon(r["path"])
        else:
            full = r["path"]
            folder = short_path(os.path.dirname(full))
            hk = r.get("hitkind")
            badge = []
            if hk in ("frame", "audio") and not r.get("sub"):
                badge.append(clock(r.get("t")))
            if r.get("page"):
                badge.append(f"p.{r['page']}")
            if r.get("versions"):
                badge.append(f"+{len(r['versions'])}")
            if shortcut:
                badge.append(shortcut)
            if r.get("sub"):
                cell.setup("item", f"Also at {clock(r.get('t'))}",
                           "same recording" if ext_of(full) in AUDIO_EXT else "soundtrack" if hk == "audio"
                           else "same video", "  ".join(badge), indent=38)
            elif kind == "hit" and not r.get("recent"):
                if hk == "frame":
                    text2 = f"Video frame at {clock(r.get('t'))} · {folder}"
                elif hk == "audio":
                    what = "Heard" if ext_of(full) in AUDIO_EXT else "Soundtrack"
                    text2 = f"{what} from {clock(r.get('t'))} · {folder}"
                elif hk == "image":
                    text2 = f"Photo · {folder}"
                else:
                    text2 = tidy(r.get("snippet") or "")[:300] or folder
                cell.setup("hit", mark_words(r["name"], self.clean), text2, "  ".join(badge))
            elif r.get("used"):
                cell.setup("item", r["name"], ago(r["used"]), "  ".join(badge))
            elif r.get("added"):
                cell.setup("item", r["name"], f"Added {ago(r['added'])} · {folder}", "  ".join(badge))
            elif self.mode == "path":
                cell.setup("item", mark_words(r["name"], os.path.basename(self.query)),
                           "Folder" if r.get("dir") else folder, "  ".join(badge))
            else:
                cell.setup("item", mark_words(r["name"], self.clean), folder, "  ".join(badge))
            if r.get("versions"):
                cell.setToolTip_("Also at:\n" + "\n".join(short_path(HOME / v) for v in r["versions"][:8]))
            else:
                cell.setToolTip_(None)
            img = self.thumbs.get(full, r.get("t"), hk, lambda i=row, p=full: self.thumbReady(i, p))
            img = img or self.thumbs.icon(full)
        cell.icon.setImage_(img)
        cell.paint(table.selectedRow() == row and self.panel.isKeyWindow())
        return cell

    @objc.python_method
    def thumbReady(self, row, path):
        if row < len(self.rows) and self.rows[row].get("path") == path:
            self.table.reloadDataForRowIndexes_columnIndexes_(AppKit.NSIndexSet.indexSetWithIndex_(row),
                                                             AppKit.NSIndexSet.indexSetWithIndex_(0))

    def tableViewSelectionDidChange_(self, note):
        self.updatePreview()
        if self.grid_mode:
            self.grid.setNeedsDisplay_(True)
        if self.ql_visible():
            Quartz.QLPreviewPanel.sharedPreviewPanel().reloadData()

    @objc.python_method
    def updatePreview(self):
        row = self.table.selectedRow()
        if not 0 <= row < len(self.rows):
            self.preview.clear()
            return
        if self.db is None:
            try:
                self.db = sqlite3.connect(f"file:{expand(self.cfg['index'])}?mode=ro", uri=True)
            except sqlite3.Error:
                self.db = None
        try:
            self.preview.show(self.rows[row], self.clean, self.db)
        except Exception as e:  # a preview must never take the window down
            dbg("preview failed:", e)
            self.preview.clear()

    # actions -------------------------------------------------------------------------------------------------------
    def openClicked_(self, sender):
        self.activate(self.table.clickedRow())

    @objc.python_method
    def activate(self, row):
        if not 0 <= row < len(self.rows):
            return
        r = self.rows[row]
        kind = r["kind"]
        if kind == "header":
            return
        if kind == "calc":
            self.copy_text(r["copy"])
            self.foot_left.setStringValue_(f"Copied {r['copy']}")
            return
        if kind == "clip":
            self.clipboard.restore(r["item"])
            self.hide()
            return
        if kind == "action":
            self.run_action(r)
            return
        if kind in ("emoji", "snippet"):
            self.copy_text(r["value"])
            self.foot_left.setStringValue_("Copied")
            self.hide()
            return
        self.hide()
        ws = AppKit.NSWorkspace.sharedWorkspace()
        if kind == "link":
            ws.openURL_(NSURL.URLWithString_(r["url"]))
            return
        if kind == "bookmark":
            if os.path.exists(r["app"]):
                subprocess.Popen(["/usr/bin/open", "-a", r["app"], r["url"]], start_new_session=True)
            else:
                ws.openURL_(NSURL.URLWithString_(r["url"]))
            return
        if kind == "code":
            open_in_editor(r["path"], r["line"])
            return
        if kind == "web":
            ws.openURL_(NSURL.URLWithString_("https://www.google.com/search?q=" + urllib.parse.quote(self.query)))
            return
        if kind == "setting":
            ws.openURL_(NSURL.URLWithString_("x-apple.systempreferences:" + r["pane"]))
            return
        if kind == "define":
            ws.openURL_(NSURL.URLWithString_("dict://" + urllib.parse.quote(r["name"])))
            return
        self.history.add(r)
        full = r["path"]
        if kind == "app":
            ws.openApplicationAtURL_configuration_completionHandler_(
                NSURL.fileURLWithPath_(full), AppKit.NSWorkspaceOpenConfiguration.configuration(), None)
            return
        if r.get("hitkind") in ("frame", "audio") and r.get("t"):
            play_at(full, float(r["t"]))
            return
        if ext_of(full) == "pdf" and self.clean:
            self.open_pdf_find(full, self.find_text(r))
            return
        ws.openURL_(NSURL.fileURLWithPath_(full))

    @objc.python_method
    def find_text(self, r):
        """What Preview should search for: the words the snippet matched, else the code typed, else the query."""
        m = re.search(r"«([^»]{2,60})»", r.get("snippet") or "")
        if m:
            return m[1]
        codes = code_tokens(self.clean)
        return codes[0] if codes else self.clean

    @objc.python_method
    def open_pdf_find(self, full, text):
        """Open the PDF in its usual app with text in the app's find (what Spotlight sends: odoc with keyAESearchText)."""
        ws = AppKit.NSWorkspace.sharedWorkspace()
        url = NSURL.fileURLWithPath_(full)
        app = ws.URLForApplicationToOpenURL_(url)
        D = Foundation.NSAppleEventDescriptor
        ev = D.appleEventWithEventClass_eventID_targetDescriptor_returnID_transactionID_(
            fourcc("aevt"), fourcc("odoc"), D.nullDescriptor(), -1, 0)
        files = D.listDescriptor()
        files.insertDescriptor_atIndex_(D.descriptorWithFileURL_(url), 0)
        ev.setParamDescriptor_forKeyword_(files, fourcc("----"))
        ev.setParamDescriptor_forKeyword_(D.descriptorWithString_(text), fourcc("stxt"))
        conf = AppKit.NSWorkspaceOpenConfiguration.configuration()
        conf.setAppleEvent_(ev)
        if app is None:
            ws.openURL_(url)
            return
        ws.openURLs_withApplicationAtURL_configuration_completionHandler_([url], app, conf, None)

    @objc.python_method
    def run_action(self, r):
        act, arg = r["act"], r.get("arg")

        def osa(script, *args):
            subprocess.Popen(["/usr/bin/osascript", "-e", script, *map(str, args)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        if act == "awake" and self.awake_proc is not None and self.awake_proc.poll() is None:
            self.awake_proc.terminate()
            self.awake_proc = None
            self.foot_left.setStringValue_("Your Mac can sleep again")
            self.rebuild()
            return
        if act == "awake":
            self.awake_proc = subprocess.Popen(["/usr/bin/caffeinate", "-d", "-i", "-t", "3600"],
                                               start_new_session=True)
            self.foot_left.setStringValue_("Staying awake for an hour · search awake again to stop")
            return
        if act in CONFIRM and (self.armed[0] != act or time.time() - self.armed[1] > 6):
            self.armed = (act, time.time())
            self.foot_left.setStringValue_(f"Press Return again to {CONFIRM[act]}")
            return
        self.armed = (None, 0.0)
        if act == "timer":
            secs, label = arg
            self.timer_ids += 1
            tid = self.timer_ids
            self.timers[tid] = (time.time() + secs, label)

            def ring():
                if self.timers.pop(tid, None) is None:
                    return   # cancelled
                sound = AppKit.NSSound.soundNamed_("Glass")
                if sound is not None:
                    sound.play()
                osa('on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv) sound name '
                    '"Glass"\nend run', "Timer done", label or f"{span_text(secs)} is up")
            AppHelper.callLater(secs, ring)
            self.foot_left.setStringValue_(f"Timer set for {span_text(secs)}" + (f": {label}" if label else ""))
            self.hide()
            return
        if act == "canceltimer":
            self.timers.pop(arg, None)
            self.foot_left.setStringValue_("Timer cancelled")
            self.rebuild()
            return
        if act == "settings":
            self.hide()
            self.settings_(None)
            return
        if act == "clipboard":
            self.field.setStringValue_("")
            self.controlTextDidChange_(None)
            self.setScope(SCOPES.index("Clipboard"))
            return
        if act == "scratchpad":
            self.hide()
            if self.scratch is None:
                self.scratch = Scratchpad.alloc().init()
            self.scratch.show()
            return
        if act == "savesnippet":
            text = AppKit.NSPasteboard.generalPasteboard().stringForType_(AppKit.NSPasteboardTypeString)
            if text and text.strip():
                self.foot_left.setStringValue_(f"Saved snippet “{self.snippets.add(text)}”")
            else:
                self.foot_left.setStringValue_("Copy some text first, then save it as a snippet")
            return
        if act == "kill":
            try:
                os.kill(arg, signal.SIGKILL)
                self.foot_left.setStringValue_("Force quit")
            except OSError as e:
                self.foot_left.setStringValue_(f"Could not: {e.strerror}")
            self.rebuild()
            return
        if act == "clearclip":
            self.clipboard.forget()
            self.foot_left.setStringValue_("Clipboard history cleared")
            self.rebuild()
            return
        if act == "reindex":
            self.refreshIndex_(None)
            if self.watcher is not None:
                self.watcher.state["last_update"] = time.time()
            self.foot_left.setStringValue_("Updating the index in the background")
            return
        self.hide()
        if act == "screenshot":
            subprocess.Popen(["/usr/bin/open", "-a", "Screenshot"])
        elif act == "note":
            osa('on run argv\ntell application "Notes"\nactivate\nif (item 1 of argv) is not "" then make new '
                'note with properties {body:(item 1 of argv)}\nend tell\nend run', arg or "")
        elif act == "reminder":
            osa('on run argv\ntell application "Reminders"\nactivate\nif (item 1 of argv) is not "" then make new '
                'reminder with properties {name:(item 1 of argv)}\nend tell\nend run', arg or "")
        elif act == "displaysleep":
            AppHelper.callLater(0.4, lambda: subprocess.Popen(["/usr/bin/pmset", "displaysleepnow"]))
        elif act == "dark":
            osa('tell application "System Events" to tell appearance preferences to set dark mode to not dark mode')
        elif act == "eject":
            osa('tell application "Finder" to eject (every disk whose ejectable is true)')
        elif act == "lock":
            def lock():
                login = ctypes.CDLL("/System/Library/PrivateFrameworks/login.framework/Versions/Current/login")
                login.SACLockScreenImmediate()
            AppHelper.callLater(0.3, lock)
        elif act == "powersleep":
            AppHelper.callLater(0.4, lambda: subprocess.Popen(["/usr/bin/pmset", "sleepnow"]))
        elif act in ("restart", "shutdown", "logout"):
            verb = {"restart": "restart", "shutdown": "shut down", "logout": "log out"}[act]
            osa(f'tell application "System Events" to {verb}')
        elif act == "shortcut":
            subprocess.Popen(["/usr/bin/shortcuts", "run", arg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        elif act == "quit":
            app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(arg)
            if app is not None:
                app.terminate()

    @objc.python_method
    def togglePin(self, row):
        if not 0 <= row < len(self.rows):
            return
        r = self.rows[row]
        full = self.target(row)
        if r["kind"] not in ("app", "file", "hit", "code") or not full or not os.path.exists(full):
            self.foot_left.setStringValue_("Only apps and files can be pinned")
            return
        on = self.pins.toggle("app" if r["kind"] == "app" else "file", full,
                              r["name"] if r["kind"] == "app" else os.path.basename(full))
        if not self.clean:
            self.rebuild()
        self.foot_left.setStringValue_(f"Pinned {r['name']} to Home" if on else f"Unpinned {r['name']}")

    @objc.python_method
    def target(self, row):
        if 0 <= row < len(self.rows) and self.rows[row].get("path"):
            return self.rows[row]["path"]
        return None

    @objc.python_method
    def reveal(self, row):
        full = self.target(row)
        if full:
            self.hide()
            AppKit.NSWorkspace.sharedWorkspace().activateFileViewerSelectingURLs_([NSURL.fileURLWithPath_(full)])

    @objc.python_method
    def copy_text(self, text):
        pb = AppKit.NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, AppKit.NSPasteboardTypeString)

    @objc.python_method
    def copy(self, row, path):
        r = self.rows[row] if 0 <= row < len(self.rows) else None
        if r and r["kind"] == "calc":
            self.copy_text(r["copy"])
            self.foot_left.setStringValue_(f"Copied {r['copy']}")
            return
        if r and r["kind"] == "clip":
            self.clipboard.restore(r["item"])
            self.foot_left.setStringValue_("Copied back to the clipboard")
            return
        if r and r["kind"] == "code":
            self.copy_text(r["text"])
            self.foot_left.setStringValue_("Code copied")
            return
        if r and r["kind"] == "define":
            self.copy_text(definition_gist(r["text"]))
            self.foot_left.setStringValue_("Definition copied")
            return
        if r and r["kind"] == "bookmark":
            self.copy_text(r["url"])
            self.foot_left.setStringValue_("Address copied")
            return
        if r and r["kind"] in ("emoji", "snippet"):
            self.copy_text(r["value"])
            self.foot_left.setStringValue_(f"Copied {r['value']}")
            return
        full = self.target(row)
        if not full:
            return
        if path:
            self.copy_text(full)
            self.foot_left.setStringValue_("Path copied")
        else:
            pb = AppKit.NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.writeObjects_([NSURL.fileURLWithPath_(full)])
            self.foot_left.setStringValue_(f"Copied {os.path.basename(full)}, paste it into Finder, Mail or WhatsApp")

    @objc.python_method
    def quicklook(self):
        if self.target(self.table.selectedRow()) is None:
            return
        ql = Quartz.QLPreviewPanel.sharedPreviewPanel()
        if ql.isVisible():
            ql.orderOut_(None)
        else:
            self.preview.stop()
            self.ql_opened = time.time()
            ql.makeKeyAndOrderFront_(None)

    # QLPreviewPanelDataSource
    @objc.typedSelector(b"q@:@")
    def numberOfPreviewItemsInPreviewPanel_(self, panel):
        return 1 if self.target(self.table.selectedRow()) else 0

    @objc.typedSelector(b"@@:@q")
    def previewPanel_previewItemAtIndex_(self, panel, i):
        return NSURL.fileURLWithPath_(self.target(self.table.selectedRow()))

    @objc.typedSelector(b"Z@:@@")
    def previewPanel_handleEvent_(self, panel, event):
        if event.type() == AppKit.NSEventTypeKeyDown and event.keyCode() in (123, 124, 125, 126):  # arrows step
            code = event.keyCode()
            if code in (123, 124) and not self.grid_mode:
                return False
            cols = self.grid_cols() if self.grid_mode else 1
            d = {123: -1, 124: 1, 125: cols, 126: -cols}
            self.step(d[code])
            panel.reloadData()
            return True
        return False


# --- app ----------------------------------------------------------------------------------------------------------------

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, note):
        self.ctl = Controller.alloc().init()
        if DEMO:
            threading.Thread(target=self.demo, daemon=True).start()
        elif not index_built(self.ctl.cfg):
            self.ctl.refreshIndex_(None)   # a fresh install: build the first index now, not at night

    def applicationWillTerminate_(self, note):
        self.ctl.engine.close()

    @objc.python_method
    def demo(self):
        """--demo QUERY [--shot PNG] [--down N]: show, type, wait for meaning-ranked results, save the window."""
        query, shot, down, scope = DEMO
        while self.ctl.engine.info is None:
            time.sleep(0.2)
        AppHelper.callAfter(self.ctl.show)
        time.sleep(0.6)
        if scope:
            AppHelper.callAfter(self.ctl.setScope, scope)
        if "--menu" in sys.argv:   # open the cmd-K menu after the results, capture the screen, close it
            def pop():
                def snap(t):
                    subprocess.run(["/usr/sbin/screencapture", "-x", shot.replace(".png", "_menu.png")])
                    self.ctl.last_menu.cancelTracking()
                AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(
                    AppKit.NSTimer.timerWithTimeInterval_repeats_block_(1.0, False, snap), AppKit.NSRunLoopCommonModes)
                self.ctl.actionMenu(self.ctl.table.selectedRow())
            self.after_results = pop
        if query.startswith("like:") and "/" in query:   # --demo like:/full/path finds files like that one
            self.ctl.like_path, query = os.path.expanduser(query[5:]), "like:" + os.path.basename(query[5:])
        if query:
            AppHelper.callAfter(self.ctl.field.setStringValue_, query)
            AppHelper.callAfter(self.ctl.controlTextDidChange_, None)
            for _ in range(300):
                time.sleep(0.2)
                info = self.ctl.info
                if not self.ctl.clean or answers(self.ctl.clean) or (self.ctl.hits_for == self.ctl.clean and info
                                               and not info.get("keywords_only")) or (
                        self.ctl.mode == "code" and self.ctl.code_for == self.ctl.clean) or (
                        self.ctl.mode in ("name", "clip")) or (self.ctl.mode == "like" and self.ctl.hits_for == self.ctl.clean):
                    break
            time.sleep(1.8)   # file names (mdfind)
            for _ in range(down):
                AppHelper.callAfter(self.ctl.step, 1)
            time.sleep(2.5)   # thumbnails, previews
        else:
            time.sleep(2.5)   # recent files (mdfind)
        AppHelper.callAfter(self.capture, shot)

    @objc.python_method
    def capture(self, shot):
        c = self.ctl
        print("DEMO", c.foot_left.stringValue(), "| grid" if c.grid_mode else "", flush=True)
        for r in c.rows[:30]:
            print("  ", r["kind"], r.get("title") or r.get("name") or r.get("value") or "", r.get("page") or "",
                  (f"@{r['t']:.0f}s" if r.get("t") is not None and r.get("hitkind") in ("frame", "audio") else ""),
                  "(moment)" if r.get("sub") else "", flush=True)
        try:
            if shot:
                cg = Quartz.CGWindowListCreateImage(Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow,
                                                    c.panel.windowNumber(), Quartz.kCGWindowImageBoundsIgnoreFraming)
                rep = AppKit.NSBitmapImageRep.alloc().initWithCGImage_(cg)
                rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(
                    shot, True)
        except Exception as e:
            print("DEMO capture failed:", e, flush=True)
        if getattr(self, "after_results", None):
            self.after_results()
        AppKit.NSApp.terminate_(None)


def single_instance():
    """Hold an exclusive lock for the life of the process; a second copy exits instead of fighting for the hotkey."""
    import fcntl
    (HERE / "logs").mkdir(exist_ok=True)
    f = open(HERE / "logs/app.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    return f


DEMO = None


def main():
    global DEMO
    a = sys.argv
    if "--demo" in a:
        DEMO = (a[a.index("--demo") + 1], a[a.index("--shot") + 1] if "--shot" in a else None,
                int(a[a.index("--down") + 1]) if "--down" in a else 0,
                int(a[a.index("--scope") + 1]) if "--scope" in a else 0)
    lock = single_instance() if not DEMO else True
    if lock is None:
        Foundation.NSDistributedNotificationCenter.defaultCenter().postNotificationName_object_userInfo_deliverImmediately_(
            "com.kritin.localsearch.show", None, None, True)
        print("localsearch is already running; asked it to show its window")
        return
    Foundation.NSProcessInfo.processInfo().setProcessName_("LocalSearch")
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    signal.signal(signal.SIGTERM, lambda *a: AppHelper.callAfter(app.terminate_, None))
    # Python runs signal handlers only between bytecodes; an idle Cocoa run loop runs none, so tick once a second.
    AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1.0, True, lambda timer: None)
    AppHelper.runEventLoop(installInterrupt=True)


if __name__ == "__main__":
    main()
