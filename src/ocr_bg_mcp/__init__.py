"""ocr-bg-mcp: OCR for Bulgarian documents as an MCP server."""


def main() -> None:
    from .server import main as run

    run()
