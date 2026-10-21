"""OCR for image-only PDF pages and screenshots. Text stays in the index; source files are only read.

Two engines, compared in tools/ocr_compare.py:
    vision     Apple Vision VNRecognizeTextRequest, accurate mode, pt-BR + en-US, language correction on
    tesseract  Tesseract 5 (eng+por, tessdata in ~/localsearch/tessdata)
"""

import os
import re
import subprocess
import tempfile
import threading
from types import SimpleNamespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDFTOPPM = "/opt/homebrew/bin/pdftoppm"
PDFTOTEXT = "/opt/homebrew/bin/pdftotext"
PDFIMAGES = "/opt/homebrew/bin/pdfimages"
TESSERACT = "/opt/homebrew/bin/tesseract"
DPI = 300
MAX_PX = 3508   # long side of A4 at 300 dpi: oversized pages (poster-size scans at 72 dpi) are fitted into this box


def image_only_pages(pdf, min_chars=25):
    """(page count, [1-based pages with fewer than min_chars letters/digits but an image of at least 200 x 200])."""
    text = subprocess.run([PDFTOTEXT, "-layout", str(pdf), "-"], capture_output=True,
                          timeout=120).stdout.decode("utf-8", "ignore")
    listing = subprocess.run([PDFIMAGES, "-list", str(pdf)], capture_output=True, text=True, timeout=120).stdout
    imgs = set()
    for line in listing.splitlines()[2:]:
        f = line.split()
        if len(f) > 4 and f[0].isdigit() and f[2] == "image" and int(f[3]) * int(f[4]) >= 200 * 200:
            imgs.add(int(f[0]))
    pages = text.split("\f")[:-1] or [text]
    return len(pages), [i for i, p in enumerate(pages, 1) if len(re.findall(r"\w", p)) < min_chars and i in imgs]


def render_page(pdf, page, out_dir, dpi=DPI, max_px=MAX_PX):
    """Render one PDF page to PNG in out_dir (never next to the PDF); returns the PNG path. 300 dpi, but no side
    longer than max_px: a 48 x 67 cm page holding a 72 dpi scan would otherwise become a 45-megapixel upscale that
    Vision needs minutes for."""
    stem = Path(out_dir) / f"p{page}"
    size = [float(x) for x in re.findall(r"Page\s+\d+\s+size:\s+([\d.]+) x ([\d.]+)", subprocess.run(
        ["/opt/homebrew/bin/pdfinfo", "-f", str(page), "-l", str(page), str(pdf)], capture_output=True, text=True,
        timeout=60).stdout)[0]] or [0, 0]
    scale = ["-r", str(dpi)] if max(size) * dpi / 72 <= max_px else ["-scale-to", str(max_px)]
    subprocess.run([PDFTOPPM, *scale, "-f", str(page), "-l", str(page), "-singlefile", "-png", str(pdf),
                    str(stem)], check=True, capture_output=True, timeout=300)
    return str(stem) + ".png"


def tesseract(image, langs="eng+por"):
    env = dict(os.environ, TESSDATA_PREFIX=str(HERE / "tessdata"))
    r = subprocess.run([TESSERACT, image, "stdout", "-l", langs, "--psm", "3"], capture_output=True, env=env,
                       timeout=300)
    return r.stdout.decode("utf-8", "ignore")


_vision = None
_vision_lock = threading.Lock()


def vision(image, langs=("pt-BR", "en-US")):
    """Apple Vision, accurate mode. Lines are joined top to bottom as Vision returns them."""
    global _vision
    with _vision_lock:  # pyobjc resolves names lazily, which is not safe across threads: do it once
        if _vision is None:
            import Quartz
            import Vision
            from Foundation import NSURL
            _vision = (SimpleNamespace(**{n: getattr(Quartz, n) for n in (
                "CGImageSourceCreateWithURL", "CGImageSourceCreateImageAtIndex")}),
                SimpleNamespace(**{n: getattr(Vision, n) for n in (
                    "VNRecognizeTextRequest", "VNRequestTextRecognitionLevelAccurate", "VNImageRequestHandler")}),
                NSURL)
    Quartz, Vision, NSURL = _vision
    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(image)), None)
    if src is None:
        return ""
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return ""
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(list(langs))
    req.setUsesLanguageCorrection_(True)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        return ""
    lines = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if cand:
            lines.append(str(cand[0].string()))
    return "\n".join(lines)


ENGINES = {"vision": vision, "tesseract": tesseract}


def ocr_pdf_pages(pdf, pages, engine="vision"):
    """{page: text} for the given 1-based pages, rendered at 300 dpi into a temp folder that is removed after."""
    fn = ENGINES[engine]
    out = {}
    with tempfile.TemporaryDirectory(prefix="localsearch-ocr-") as tmp:
        for p in pages:
            png = render_page(pdf, p, tmp)
            out[p] = fn(png)
            os.unlink(png)
    return out
