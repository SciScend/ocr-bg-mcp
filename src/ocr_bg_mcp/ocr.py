"""OCR pipeline for photographed or scanned Bulgarian documents.

Per image (and per page of a scanned PDF):
  1. Preprocess with OpenCV: grayscale, upscale small photos, denoise, deskew,
     adaptive binarization (phone photos have uneven light).
  2. Tesseract (-l bul by default), TSV output for word-level confidence.
  3. One Markdown document per source, with YAML front matter: source, engine,
     date, mean/min confidence and a low_confidence flag, so a client can send
     only the weak pages to a vision model.

The functions here know nothing about MCP; server.py exposes them as tools.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = ".pdf"
SOURCE_EXTS = IMAGE_EXTS | {PDF_EXT}
PDF_DPI = 300
TARGET_MIN_SIDE = 1800  # upscale small photos so Tesseract sees ~300 dpi glyphs
DEFAULT_THRESHOLD = 72.0


class OcrError(RuntimeError):
    """The environment or the input makes OCR impossible."""


@dataclass
class Page:
    text: str
    confidences: list[float]

    @property
    def mean(self) -> float:
        return round(float(np.mean(self.confidences)), 1) if self.confidences else 0.0


@dataclass
class OcrResult:
    source: Path
    pages: list[Page]
    lang: str
    engine: str
    models: str
    preprocessed: bool
    threshold: float
    date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    @property
    def is_pdf(self) -> bool:
        return self.source.suffix.lower() == PDF_EXT

    @property
    def words(self) -> int:
        return sum(len(p.confidences) for p in self.pages)

    @property
    def confidence_mean(self) -> float:
        allc = [c for p in self.pages for c in p.confidences]
        return round(float(np.mean(allc)), 1) if allc else 0.0

    @property
    def confidence_min(self) -> float:
        allc = [c for p in self.pages for c in p.confidences]
        return round(float(np.min(allc)), 1) if allc else 0.0

    @property
    def low_confidence_pages(self) -> list[int]:
        return [n for n, p in enumerate(self.pages, 1)
                if not p.confidences or p.mean < self.threshold]

    @property
    def low_confidence(self) -> bool:
        return bool(self.low_confidence_pages) or self.confidence_mean < self.threshold

    def to_markdown(self) -> str:
        fm = [
            "---",
            f"source: {self.source.name}",
            f"source_path: {self.source.resolve()}",
            f"extracted_by: {self.engine}",
            f"ocr_date: {self.date}",
            f"ocr_lang: {self.lang}",
            f"ocr_models: {self.models}",
            f"preprocessed: {str(self.preprocessed).lower()}",
        ]
        if self.is_pdf:
            fm.append(f"pages: {len(self.pages)}")
        fm += [
            f"words: {self.words}",
            f"confidence_mean: {self.confidence_mean}",
            f"confidence_min: {self.confidence_min}",
            f"low_confidence: {str(self.low_confidence).lower()}",
        ]
        if self.is_pdf:
            fm.append(f"low_confidence_pages: [{', '.join(map(str, self.low_confidence_pages))}]")
        fm += ["---", ""]
        if self.is_pdf:
            body = "\n".join(f"<!-- page {n} -->\n\n{p.text}"
                             for n, p in enumerate(self.pages, 1))
        else:
            body = self.pages[0].text
        return "\n".join(fm) + body


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

def check_tesseract(lang: str, tessdata: Path | None) -> None:
    if shutil.which("tesseract") is None:
        raise OcrError("tesseract not found: apt-get install tesseract-ocr tesseract-ocr-bul")
    cmd = ["tesseract", "--list-langs"]
    if tessdata is not None:
        cmd += ["--tessdata-dir", str(tessdata)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    have = set((out.stdout + out.stderr).split())
    missing = [l for l in lang.split("+") if l not in have]
    if missing:
        raise OcrError(f"tesseract language data missing: {', '.join(missing)}")


def pick_tessdata(best_dir: Path | None, lang: str) -> Path | None:
    """Use BEST_DIR (tessdata_best models) when it covers every language in LANG,
    otherwise the system models (None)."""
    if best_dir is None:
        return None
    if all((best_dir / f"{l}.traineddata").is_file() for l in lang.split("+")):
        return best_dir
    return None


def tesseract_version() -> str:
    """'tesseract 5.3.4', the engine named in extracted_by."""
    out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    lines = (out.stdout or out.stderr).splitlines()
    return lines[0].strip() if lines else "tesseract"


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def _deskew_angle(binary: np.ndarray) -> float:
    """Skew angle in degrees, from the minimum-area rectangle around the ink."""
    coords = np.column_stack(np.where(binary < 255))
    if coords.shape[0] < 100:
        return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    # Photographed pages are rarely tilted more than ~15 degrees; a larger
    # estimate usually means the heuristic latched onto the layout, not skew.
    return angle if abs(angle) <= 15 else 0.0


def preprocess(img: np.ndarray) -> np.ndarray:
    """Grayscale -> upscale if small -> denoise -> deskew -> adaptive threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    min_side = min(gray.shape)
    if min_side < TARGET_MIN_SIDE:
        scale = TARGET_MIN_SIDE / min_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, h=10)

    _, rough = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    angle = _deskew_angle(rough)
    if abs(angle) > 0.3:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, blockSize=41, C=15)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #

def clean_text(text: str) -> str:
    """NFC-normalize, strip trailing spaces, keep at most two blank lines."""
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    blanks = 0
    for ln in (ln.rstrip() for ln in text.splitlines()):
        blanks = blanks + 1 if not ln else 0
        if blanks <= 2:
            out.append(ln)
    return "\n".join(out).strip() + "\n"


def run_tesseract(image: Path, base: Path, lang: str, psm: int,
                  tessdata: Path | None) -> Page:
    cmd = ["tesseract", str(image), str(base), "-l", lang, "--psm", str(psm)]
    if tessdata is not None:
        cmd += ["--tessdata-dir", str(tessdata)]
    subprocess.run(cmd + ["txt", "tsv"], check=True, capture_output=True)

    text = base.with_suffix(".txt").read_text(encoding="utf-8")
    confidences: list[float] = []
    for line in base.with_suffix(".tsv").read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 12 and cols[11].strip():
            conf = float(cols[10])
            if conf >= 0:  # -1 marks structural rows, not words
                confidences.append(conf)
    return Page(clean_text(text), confidences)


def render_pdf(pdf: Path, workdir: Path, dpi: int = PDF_DPI,
               first: int | None = None, last: int | None = None) -> list[Path]:
    """Render PDF pages to PNG with pdftoppm, in page order."""
    if shutil.which("pdftoppm") is None:
        raise OcrError("pdftoppm not found: apt-get install poppler-utils")
    cmd = ["pdftoppm", "-r", str(dpi), "-png"]
    if first is not None:
        cmd += ["-f", str(first)]
    if last is not None:
        cmd += ["-l", str(last)]
    subprocess.run(cmd + [str(pdf), str(workdir / "page")], check=True, capture_output=True)
    pngs = sorted(workdir.glob("page-*.png"), key=lambda f: int(f.stem.rsplit("-", 1)[1]))
    if not pngs:
        raise OcrError(f"pdftoppm produced no pages for {pdf.name}")
    return pngs


def pdf_page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise OcrError(f"cannot read the page count of {pdf.name}")


def ocr_source(src: Path, *, lang: str = "bul", psm: int = 6, do_preprocess: bool = True,
               threshold: float = DEFAULT_THRESHOLD, best_tessdata: Path | None = None,
               max_pages: int = 50) -> OcrResult:
    """OCR one image or scanned PDF and return the result; writes nothing."""
    if src.suffix.lower() not in SOURCE_EXTS:
        raise OcrError(f"not an image or PDF: {src.name}")
    tessdata = pick_tessdata(best_tessdata, lang)
    check_tesseract(lang, tessdata)

    is_pdf = src.suffix.lower() == PDF_EXT
    if is_pdf and pdf_page_count(src) > max_pages:
        raise OcrError(f"{src.name} has more than {max_pages} pages")

    with tempfile.TemporaryDirectory(prefix="ocr-bg-") as tmp:
        workdir = Path(tmp)
        images = render_pdf(src, workdir) if is_pdf else [src]
        pages = []
        for n, image in enumerate(images, 1):
            ocr_input = image
            if do_preprocess:
                img = cv2.imread(str(image))
                if img is None:
                    raise OcrError(f"cannot read image {image.name}")
                ocr_input = workdir / f"prep-{n}.png"
                cv2.imwrite(str(ocr_input), preprocess(img))
            pages.append(run_tesseract(ocr_input, workdir / f"ocr-{n}", lang, psm, tessdata))

    return OcrResult(source=src, pages=pages, lang=lang, engine=tesseract_version(),
                     models=str(tessdata) if tessdata else "system",
                     preprocessed=do_preprocess, threshold=threshold)


def front_matter(md: str) -> dict[str, str]:
    """The flat key: value pairs of a Markdown file's YAML front matter."""
    lines = md.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fm: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep:
            fm[key.strip()] = value.strip()
    return fm
