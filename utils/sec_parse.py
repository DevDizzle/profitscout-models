import re, itertools

HEAD = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$", re.MULTILINE)

def parse_sections(md: str) -> dict[str, str]:
    """Return {'Heading': 'body…', …} for a markdown earnings-call."""
    pieces = HEAD.split(md)
    # pieces = ["intro text", "Heading", "body", "Heading2", "body2", …]
    intro, chunks = pieces[0], pieces[1:]
    return {
        head.strip(): body.strip()
        for head, body in itertools.batched(chunks, 2)
    }
