import re, logging

# Match either markdown headings (# Title) or bold headings (**Title**)
HEAD = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*)(.+?)(?:\*\*)?\s*$", re.MULTILINE
)

def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = []
        try:
            for _ in range(n):
                batch.append(next(it))
            yield batch
        except StopIteration:
            if batch:
                yield batch
            break

def parse_sections(md: str) -> dict[str, str]:
    """Return {'Heading': 'body…', …} for markdown or bold-section earnings-call."""
    pieces = HEAD.split(md)
    intro, chunks = pieces[0], pieces[1:]
    if not chunks:
        logging.warning("[parse_sections] No section headings found, returning whole doc as 'All Text'")
        return {"All Text": md.strip()}
    return {
        head.strip(): body.strip()
        for head, body in batched(chunks, 2)
    }
