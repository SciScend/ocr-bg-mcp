# ocr-bg-mcp

An [MCP](https://modelcontextprotocol.io/) server that gives an AI agent OCR for
photographed and scanned Bulgarian documents: certificates, declarations,
invoices, contracts, letters from institutions.

**Why.** A photo of a page costs a vision model well over a thousand tokens, every
time it is read. Its OCR text costs a few hundred, and once it is written as
`Foo.md` next to `Foo.jpg` it is never paid for again. The agent works with text
and looks at an image only for the pages Tesseract was unsure about.

## Tools

| Tool | What it does |
|---|---|
| `ocr_document` | One image or scanned PDF -> Markdown with YAML front matter: engine, date, mean and minimum word confidence, `low_confidence`, and for a PDF the numbers of the weak pages. Writes `Foo.md` next to the source; a later call returns that file instead of running OCR again (`force` redoes it). |
| `ocr_folder` | Every image and PDF in a folder, with progress notifications. Returns a summary without the text, so a folder of 200 scans stays cheap; the agent then greps or reads only the `.md` files it needs. |
| `render_page` | One page as a PNG, downscaled, for the vision fallback on a page flagged as low confidence. |

Both OCR tools return structured output (a JSON schema the client can rely on),
and every tool carries MCP annotations (`readOnlyHint`, `idempotentHint`).

## Pipeline

1. **Preprocessing** with OpenCV for phone photos: grayscale, upscale small images
   to about 300 dpi, denoise, deskew, adaptive binarization against uneven light.
   Off with `preprocess: false` for clean scanner output.
2. **Tesseract** with `-l bul` (or `bul+eng` for mixed text such as IBANs and
   e-mail addresses), TSV output for per-word confidence. Scanned PDFs are
   rendered at 300 dpi with `pdftoppm`.
3. **Markdown** with the confidence numbers in the front matter, so the client
   decides what to trust. PDF pages are separated by `<!-- page N -->`.

## Security

The server reads and writes only under the folders in `OCR_BG_ROOTS`
(separated by `:` on Linux and macOS; default: the working directory). Paths
are resolved, symlinks included, before the check, so a link that points
outside a root is refused. PDFs over 50 pages are refused. Nothing leaves the
machine: no API calls, no telemetry.

## Install

System packages (Debian/Ubuntu):

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-bul poppler-utils
```

Add it to Claude Code:

```bash
claude mcp add ocr-bg \
  -e OCR_BG_ROOTS="$HOME/Documents" \
  -- uvx --from git+https://github.com/SciScend/ocr-bg-mcp ocr-bg-mcp
```

Or in any MCP client's JSON config:

```json
{
  "mcpServers": {
    "ocr-bg": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/SciScend/ocr-bg-mcp", "ocr-bg-mcp"],
      "env": { "OCR_BG_ROOTS": "/home/me/Documents" }
    }
  }
}
```

Optional: `OCR_BG_TESSDATA` points at a folder with the more accurate
[tessdata_best](https://github.com/tesseract-ocr/tessdata_best) models. They are
used when that folder has every requested language; otherwise the system models.

## Develop

```bash
uv sync
uv run pytest
```

The tests start a real MCP client against the server in-process and OCR
documents rendered from known Bulgarian text: an image, a two-page PDF with a
blank page (which must be flagged), a folder, and paths and symlinks outside the
allowed roots.

## Known limits

Handwriting: Tesseract cannot read it; use vision. Tables keep their text but
lose their structure. Deskew handles rotation, not perspective: a page shot at
an angle should be shot again from above. Cyrillic confusions to watch even at
high confidence: и/н, п/л, З/3, О/0.

---

Built with Claude Opus 5.5 via Claude Code. Published by [SciScend](https://sciscend.com/).
