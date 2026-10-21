"""Photos and video for the index: thumbnails, camera EXIF, video frames per shot, soundtrack windows.

Sources are only read. Thumbnails, frames and audio windows are written to a temporary folder and removed after use.
"""

import json
import re
import subprocess
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
RAW_EXT = {"dng", "cr2", "cr3", "nef", "arw", "raf", "orf", "rw2"}   # camera RAW: ImageIO decodes them
IMAGE_EXT = {"jpg", "jpeg", "png", "heic", "webp", "gif", "tif", "tiff", "bmp"} | RAW_EXT
VIDEO_EXT = {"mp4", "mov", "m4v"}
AUDIO_EXT = {"mp3", "m4a", "wav", "aac", "flac", "aiff", "aif", "opus", "ogg", "caf"}   # 30 s windows, as soundtracks
THUMB_PX = 448          # long side sent to the server; it resizes to the encoder's 224 itself (448 tested a bit better)
MIN_PX = 64             # icons and spacers are skipped
SCENE = 0.3             # ffmpeg scene-change score for a new shot
FRAME_EVERY_S = 10      # at least one frame this often
FRAME_MIN_GAP_S = 1     # and no two frames closer than this
MAX_FRAMES = 600
AUDIO_WINDOW_S = 30
SCREENSHOT_NAME = re.compile(r"^(screenshot|screen shot|captura de tela|bildschirmfoto)", re.I)
MACHINE_NAME = re.compile(r"^(?:[0-9a-f]{8}-[0-9a-f-]{27}\S*|[0-9a-f]{16,}|[a-z]{0,8}[_-]?\d{2,}(?:[_-][0-9a-z]+)*"
                          r"|\d+_[0-9a-f-]{20,}|img_\d+|dsc_?\d+|gopr\d+|tile\S*|\S*_\d{3,})\.\w+$", re.I)

_quartz = None
_quartz_lock = threading.Lock()
QUARTZ_NAMES = ["CGImageSourceCreateWithURL", "CGImageSourceGetCount", "CGImageSourceCopyPropertiesAtIndex",
                "CGImageSourceCreateThumbnailAtIndex", "CGImageDestinationCreateWithData", "CGImageDestinationAddImage",
                "CGImageDestinationFinalize", "kCGImageSourceCreateThumbnailFromImageAlways",
                "kCGImageSourceCreateThumbnailWithTransform", "kCGImageSourceThumbnailMaxPixelSize",
                "kCGImageDestinationLossyCompressionQuality"]


def _q():
    """Quartz names resolved once, under a lock: pyobjc's lazy lookup is not safe across threads."""
    global _quartz
    with _quartz_lock:
        if _quartz is None:
            import Quartz
            from Foundation import NSURL, NSMutableData
            _quartz = SimpleNamespace(NSURL=NSURL, NSMutableData=NSMutableData,
                                      **{n: getattr(Quartz, n) for n in QUARTZ_NAMES})
    return _quartz


def _source(path):
    q = _q()
    src = q.CGImageSourceCreateWithURL(q.NSURL.fileURLWithPath_(str(path)), None)
    if src is None or q.CGImageSourceGetCount(src) < 1:
        raise OSError("unreadable image (permission or format)")
    return src


def image_info(path):
    """(width, height, camera make or None) from the file header, without decoding the image."""
    props = _q().CGImageSourceCopyPropertiesAtIndex(_source(path), 0, None) or {}
    tiff = props.get("{TIFF}") or {}
    return int(props.get("PixelWidth") or 0), int(props.get("PixelHeight") or 0), tiff.get("Make")


def thumbnail(path, px=THUMB_PX):
    """JPEG bytes of the image scaled so its long side is px, EXIF orientation applied."""
    q = _q()
    opts = {q.kCGImageSourceCreateThumbnailFromImageAlways: True,
            q.kCGImageSourceCreateThumbnailWithTransform: True,
            q.kCGImageSourceThumbnailMaxPixelSize: px}
    img = q.CGImageSourceCreateThumbnailAtIndex(_source(path), 0, opts)
    if img is None:
        raise OSError("could not decode image")
    data = q.NSMutableData.data()
    dest = q.CGImageDestinationCreateWithData(data, "public.jpeg", 1, None)
    q.CGImageDestinationAddImage(dest, img, {q.kCGImageDestinationLossyCompressionQuality: 0.85})
    if not q.CGImageDestinationFinalize(dest):
        raise OSError("could not encode thumbnail")
    return bytes(data)


def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json",
                          str(path)], capture_output=True, text=True, timeout=120).stdout
    d = json.loads(out or "{}")
    kinds = {s.get("codec_type") for s in d.get("streams", [])}
    return float(d.get("format", {}).get("duration") or 0), "video" in kinds, "audio" in kinds


def video_frames(path, tmp):
    """[(seconds, jpeg path)]: one frame per shot (scene change), at least one every 10 s, none closer than 1 s."""
    sel = (f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,{FRAME_EVERY_S})"
           f"+gt(scene\\,{SCENE})*gte(t-prev_selected_t\\,{FRAME_MIN_GAP_S})")
    vf = f"select='{sel}',showinfo,scale={THUMB_PX}:{THUMB_PX}:force_original_aspect_ratio=decrease"
    r = subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-i", str(path), "-map", "0:v:0", "-vf", vf, "-fps_mode",
                        "vfr", "-frames:v", str(MAX_FRAMES), "-q:v", "4", str(Path(tmp) / "f%05d.jpg")],
                       capture_output=True, text=True, timeout=3600)
    times = [float(m) for m in re.findall(r"pts_time:\s*([0-9.]+)", r.stderr)]
    files = sorted(Path(tmp).glob("f*.jpg"))
    return list(zip(times, files))


def audio_windows(path, tmp, duration):
    """[(start, end, wav path)] of 30 s mono 16 kHz windows; silent windows are dropped."""
    subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                    "-f", "segment", "-segment_time", str(AUDIO_WINDOW_S), str(Path(tmp) / "a%04d.wav")],
                   capture_output=True, timeout=3600)
    out = []
    for i, f in enumerate(sorted(Path(tmp).glob("a*.wav"))):
        start = i * AUDIO_WINDOW_S
        with wave.open(str(f)) as w:
            n = w.getnframes()
            samples = np.frombuffer(w.readframes(n), dtype=np.int16)
        if n < 16000 * 2 or not len(samples) or np.abs(samples).max() < 100:   # under 2 s, or silence
            continue
        out.append((start, min(start + AUDIO_WINDOW_S, duration or start + AUDIO_WINDOW_S), f))
    return out


def is_dataset(names, sample_paths, exempt):
    """DigUp's rule, made photo-aware: a folder of 100+ images, mostly machine-named, where fewer than 1 in 5 of a
    sample carries a camera make in EXIF and most of the sample is PNG/WebP or square (generated tiles and chips).
    Camera rolls keep their EXIF; phone exports (001_<uuid>.JPG, EXIF stripped) are non-square JPEG/HEIC."""
    if exempt or len(names) < 100:
        return False
    if sum(bool(MACHINE_NAME.match(n)) for n in names) < 0.8 * len(names):
        return False
    makes = generated = 0
    for p in sample_paths:
        try:
            w, h, make = image_info(p)
        except OSError:
            continue
        makes += make is not None
        generated += p.rsplit(".", 1)[-1].lower() in ("png", "webp") or w == h
    return makes < 0.2 * len(sample_paths) and generated >= 0.5 * len(sample_paths)


def screenshots(roots):
    """Paths Spotlight marks as screen captures (kMDItemIsScreenCapture), under the given folders."""
    out = set()
    for root in roots:
        r = subprocess.run(["/usr/bin/mdfind", "-onlyin", str(root), "kMDItemIsScreenCapture == 1"],
                           capture_output=True, text=True, timeout=120)
        out.update(line for line in r.stdout.splitlines() if line)
    return out
