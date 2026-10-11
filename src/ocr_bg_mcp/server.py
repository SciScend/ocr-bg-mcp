"""MCP server: OCR for Bulgarian documents, as tools an AI agent can call.

The point is token economy. A photo of a page costs a vision model well over a
thousand tokens; its OCR text costs a few hundred, and once written as
Foo.md next to Foo.jpg it is never paid for again. The agent reads text, and
looks at an image only for the pages Tesseract was unsure about.

Tools:
  ocr_document  - one image or scanned PDF -> Markdown with confidence metadata
  ocr_folder    - every image and PDF in a folder; returns a summary, not the text
  render_page   - one page as an image, for the vision fallback on weak pages

Security: the server reads and writes only under the folders listed in
OCR_BG_ROOTS (separated by the OS path separator; default: the working
directory). Symlinks are resolved before the check.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

import anyio.to_thread
import cv2
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from . import ocr

INSTRUCTIONS = """\
OCR for photographed or scanned Bulgarian documents (also English with lang=bul+eng).
Prefer these tools over reading document images with vision: text is far cheaper.
Each source gets a Markdown file next to it (Foo.jpg -> Foo.md) and is not OCRed again
unless force=true. When a result has low_confidence pages, check only those pages with
render_page; numbers with legal or financial weight (sums, IDs, IBANs) are worth that check
even at high confidence. Tesseract confuses и/н, п/л, З/3 and О/0 in Cyrillic."""

mcp = MCPServer("ocr-bg", instructions=INSTRUCTIONS, version="0.1.0")


def _roots() -> list[Path]:
    raw = os.environ.get("OCR_BG_ROOTS", "")
    roots = [Path(r).expanduser().resolve() for r in raw.split(os.pathsep) if r.strip()]
    return roots or [Path.cwd().resolve()]


def _best_tessdata() -> Path | None:
    raw = os.environ.get("OCR_BG_TESSDATA")
    return Path(raw).expanduser() if raw else None


def _allowed(path: str) -> Path:
    """Resolve PATH and refuse anything outside the configured roots."""
    p = Path(path).expanduser().resolve()
    if not any(p == r or p.is_relative_to(r) for r in _roots()):
        raise ToolError(f"{path} is outside the allowed folders (OCR_BG_ROOTS)")
    if not p.exists():
        raise ToolError(f"{path} does not exist")
    return p


class DocumentResult(BaseModel):
    source: str
    markdown_path: str | None = Field(description="The .md written next to the source, if saved")
    cached: bool = Field(description="True when an earlier .md was returned instead of a new OCR run")
    pages: int
    words: int | None
    confidence_mean: float | None
    low_confidence: bool
    low_confidence_pages: list[int]
    markdown: str = Field(description="Front matter plus text; PDF pages start with <!-- page N -->")


class FolderItem(BaseModel):
    source: str
    markdown_path: str | None
    status: str = Field(description="ocr | cached | error")
    confidence_mean: float | None = None
    low_confidence: bool = False
    low_confidence_pages: list[int] = []
    error: str | None = None


class FolderResult(BaseModel):
    folder: str
    files: int
    low_confidence: int
    errors: int
    items: list[FolderItem]


def _int_list(value: str) -> list[int]:
    inner = value.strip().strip("[]")
    return [int(x) for x in inner.split(",") if x.strip()]


def _from_existing(src: Path, md_path: Path) -> DocumentResult:
    md = md_path.read_text(encoding="utf-8")
    fm = ocr.front_matter(md)
    return DocumentResult(
        source=str(src), markdown_path=str(md_path), cached=True,
        pages=int(fm.get("pages", 1)),
        words=int(fm["words"]) if "words" in fm else None,
        confidence_mean=float(fm["confidence_mean"]) if "confidence_mean" in fm else None,
        low_confidence=fm.get("low_confidence") == "true",
        low_confidence_pages=_int_list(fm.get("low_confidence_pages", "")),
        markdown=md)


def _ocr_one(src: Path, lang: str, psm: int, preprocess: bool, threshold: float,
             save: bool, force: bool) -> DocumentResult:
    md_path = src.with_suffix(".md")
    if md_path.exists() and not force:
        return _from_existing(src, md_path)
    try:
        res = ocr.ocr_source(src, lang=lang, psm=psm, do_preprocess=preprocess,
                             threshold=threshold, best_tessdata=_best_tessdata())
    except ocr.OcrError as exc:
        raise ToolError(str(exc)) from exc
    md = res.to_markdown()
    if save:
        md_path.write_text(md, encoding="utf-8")
    return DocumentResult(
        source=str(src), markdown_path=str(md_path) if save else None, cached=False,
        pages=len(res.pages), words=res.words, confidence_mean=res.confidence_mean,
        low_confidence=res.low_confidence, low_confidence_pages=res.low_confidence_pages,
        markdown=md)


Lang = Annotated[str, Field(description="Tesseract languages, e.g. bul or bul+eng")]
Psm = Annotated[int, Field(ge=0, le=13, description="Page segmentation: 6 = one text block "
                           "(photographed pages), 3 or 4 = multi-column layouts")]
Preprocess = Annotated[bool, Field(description="Deskew, denoise and binarize; turn off for clean scans")]
Threshold = Annotated[float, Field(ge=0, le=100, description="Mean confidence below which a page is flagged")]


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def ocr_document(
    path: Annotated[str, Field(description="Image (.jpg .png .tif .webp ...) or scanned PDF")],
    lang: Lang = "bul",
    psm: Psm = 6,
    preprocess: Preprocess = True,
    threshold: Threshold = ocr.DEFAULT_THRESHOLD,
    save: Annotated[bool, Field(description="Write Foo.md next to the source")] = True,
    force: Annotated[bool, Field(description="OCR again even if Foo.md exists")] = False,
) -> DocumentResult:
    """OCR one photographed or scanned document and return its text as Markdown,
    with confidence per page. Returns the existing .md instead when there is one."""
    src = _allowed(path)
    if not src.is_file():
        raise ToolError(f"{path} is not a file; use ocr_folder for folders")
    return _ocr_one(src, lang, psm, preprocess, threshold, save, force)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
async def ocr_folder(
    path: Annotated[str, Field(description="Folder with document photos or scanned PDFs")],
    ctx: Context,
    recursive: bool = False,
    lang: Lang = "bul",
    psm: Psm = 6,
    preprocess: Preprocess = True,
    threshold: Threshold = ocr.DEFAULT_THRESHOLD,
    force: Annotated[bool, Field(description="OCR again sources whose .md exists")] = False,
) -> FolderResult:
    """OCR every image and scanned PDF in a folder, writing Foo.md next to each.
    Returns a summary without the text, so a large folder stays cheap; read the
    .md files you need afterwards, or grep them."""
    folder = _allowed(path)
    if not folder.is_dir():
        raise ToolError(f"{path} is not a folder")
    it = folder.rglob("*") if recursive else folder.glob("*")
    files = sorted(f for f in it if f.is_file() and f.suffix.lower() in ocr.SOURCE_EXTS)

    items: list[FolderItem] = []
    for i, src in enumerate(files):
        await ctx.report_progress(i, len(files), src.name)
        try:
            r = await anyio.to_thread.run_sync(
                _ocr_one, src, lang, psm, preprocess, threshold, True, force)
            items.append(FolderItem(
                source=str(src), markdown_path=r.markdown_path,
                status="cached" if r.cached else "ocr", confidence_mean=r.confidence_mean,
                low_confidence=r.low_confidence, low_confidence_pages=r.low_confidence_pages))
        except Exception as exc:  # keep the batch going
            items.append(FolderItem(source=str(src), markdown_path=None, status="error",
                                    low_confidence=True, error=str(exc)))
    await ctx.report_progress(len(files), len(files), "done")

    return FolderResult(folder=str(folder), files=len(files),
                        low_confidence=sum(1 for x in items if x.low_confidence),
                        errors=sum(1 for x in items if x.status == "error"),
                        items=items)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def render_page(
    path: Annotated[str, Field(description="The image or scanned PDF")],
    page: Annotated[int, Field(ge=1, description="Page number, for PDFs")] = 1,
    max_side: Annotated[int, Field(ge=400, le=3000, description="Longest side in pixels")] = 1600,
) -> Image:
    """Return one page as a PNG image, so a vision model can check a page that
    OCR flagged as low confidence. Use it only for those pages."""
    src = _allowed(path)
    if src.suffix.lower() not in ocr.SOURCE_EXTS:
        raise ToolError(f"not an image or PDF: {src.name}")
    with tempfile.TemporaryDirectory(prefix="ocr-bg-") as tmp:
        if src.suffix.lower() == ocr.PDF_EXT:
            try:
                if page > ocr.pdf_page_count(src):
                    raise ToolError(f"{src.name} has no page {page}")
                image_path = ocr.render_pdf(src, Path(tmp), dpi=150, first=page, last=page)[0]
            except ocr.OcrError as exc:
                raise ToolError(str(exc)) from exc
        else:
            if page != 1:
                raise ToolError("an image has only page 1")
            image_path = src
        img = cv2.imread(str(image_path))
        if img is None:
            raise ToolError(f"cannot read image {src.name}")
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ToolError("PNG encoding failed")
    return Image(data=buf.tobytes(), format="png")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
