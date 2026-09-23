from pathlib import Path


def scan_line_offsets(path: Path, input_format: str) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            if input_format == "single-json" and stripped in {b"[", b"]"}:
                continue
            if input_format == "single-json" and stripped.startswith(b"]"):
                break
            offsets.append(offset)
    return offsets
