"""End-to-end tests: a real MCP client talks to the server in-process, and
Tesseract reads documents rendered from known Bulgarian text."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp import Client
from PIL import Image, ImageDraw, ImageFont

from ocr_bg_mcp.server import mcp

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LINES = [
    "УДОСТОВЕРЕНИЕ",
    "Настоящото удостоверение се издава",
    "в уверение на това, че документът",
    "е приет от общината в Сандански.",
]


def page(lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (1700, 1100), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 44)
    for i, line in enumerate(lines):
        draw.text((120, 140 + i * 90), line, fill="black", font=font)
    return img


@pytest.fixture
def docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OCR_BG_ROOTS", str(tmp_path))
    page(LINES).save(tmp_path / "certificate.png")
    first, blank = page(LINES), page([])
    first.save(tmp_path / "scan.pdf", save_all=True, append_images=[blank], resolution=150)
    return tmp_path


def call(tool: str, args: dict):
    async def go():
        async with Client(mcp) as client:
            return await client.call_tool(tool, args)
    return asyncio.run(go())


def test_ocr_document_reads_bulgarian_and_saves_markdown(docs: Path):
    res = call("ocr_document", {"path": str(docs / "certificate.png")})
    assert not res.is_error
    out = res.structured_content
    assert out["cached"] is False
    assert out["low_confidence"] is False
    assert "удостоверение" in out["markdown"].lower()
    assert "Сандански" in out["markdown"]
    md = (docs / "certificate.md").read_text(encoding="utf-8")
    assert md.startswith("---\nsource: certificate.png\n")


def test_second_call_returns_the_existing_markdown(docs: Path):
    call("ocr_document", {"path": str(docs / "certificate.png")})
    out = call("ocr_document", {"path": str(docs / "certificate.png")}).structured_content
    assert out["cached"] is True


def test_pdf_flags_the_blank_page(docs: Path):
    out = call("ocr_document", {"path": str(docs / "scan.pdf"), "save": False}).structured_content
    assert out["pages"] == 2
    assert out["low_confidence_pages"] == [2]
    assert "<!-- page 2 -->" in out["markdown"]
    assert not (docs / "scan.md").exists()


def test_folder_returns_a_summary_without_text(docs: Path):
    out = call("ocr_folder", {"path": str(docs)}).structured_content
    assert out["files"] == 2
    assert out["errors"] == 0
    assert "markdown" not in out["items"][0]
    assert (docs / "certificate.md").exists() and (docs / "scan.md").exists()


def test_render_page_returns_an_image(docs: Path):
    res = call("render_page", {"path": str(docs / "scan.pdf"), "page": 2})
    assert not res.is_error
    assert res.content[0].type == "image"
    assert res.content[0].mime_type == "image/png"


def test_paths_outside_the_roots_are_refused(docs: Path, tmp_path_factory: pytest.TempPathFactory):
    outside = tmp_path_factory.mktemp("outside") / "x.png"
    page(LINES).save(outside)
    res = call("ocr_document", {"path": str(outside)})
    assert res.is_error
    assert "outside the allowed folders" in res.content[0].text


def test_symlink_out_of_the_roots_is_refused(docs: Path, tmp_path_factory: pytest.TempPathFactory):
    outside = tmp_path_factory.mktemp("secret") / "x.png"
    page(LINES).save(outside)
    (docs / "link.png").symlink_to(outside)
    assert call("ocr_document", {"path": str(docs / "link.png")}).is_error
