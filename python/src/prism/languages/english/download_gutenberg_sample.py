"""Download a dialect-balanced sample of public-domain Project Gutenberg texts.

English-language Project Gutenberg is naturally a mix of British and American
authors, so a broad English sample already covers both dialects (the Gutenberg
adapter applies no spelling normalization). This fetches the official catalog,
filters to monolingual English text works, takes a deterministic sample spread
across the catalog, and downloads each book's plain-text cache file into a
directory that ``prepare_silver_corpus --source gutenberg-eng`` consumes.

Best-effort: books without a ``pg<id>.txt`` cache file are skipped, so the
sampler oversamples to compensate. Keep ``--book-count`` modest and the default
request delay to stay friendly to the Gutenberg servers; already-downloaded
files are skipped so the command is resumable.
"""

import argparse
import csv
import io
import time
import urllib.error
import urllib.request
from pathlib import Path


CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_TEXT_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


def _fetch(url: str, *, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Prism-Gutenberg-silver/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def english_text_book_ids(catalog_bytes: bytes) -> list[int]:
    """Return the ids of every monolingual English text work in the catalog."""

    book_ids: list[int] = []
    reader = csv.DictReader(io.StringIO(catalog_bytes.decode("utf-8", errors="replace")))
    for row in reader:
        if row.get("Type") == "Text" and row.get("Language") == "en":
            try:
                book_ids.append(int(row["Text#"]))
            except (KeyError, ValueError):
                continue
    return sorted(book_ids)


def deterministic_sample(book_ids: list[int], count: int) -> list[int]:
    """Spread ``count`` ids evenly across the sorted catalog for era/genre mix."""

    if count >= len(book_ids):
        return list(book_ids)
    stride = len(book_ids) / count
    return [book_ids[int(index * stride)] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Project Gutenberg English sample for silver labeling."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/gutenberg-eng"))
    parser.add_argument("--book-count", type=int, default=300)
    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    arguments = parser.parse_args()
    if arguments.book_count <= 0:
        parser.error("--book-count must be greater than zero")
    arguments.output.mkdir(parents=True, exist_ok=True)

    print("Fetching Project Gutenberg catalog ...", flush=True)
    catalog_bytes = _fetch(CATALOG_URL)
    all_ids = english_text_book_ids(catalog_bytes)
    # Oversample: many works have no plain-text cache file and are skipped.
    sample = deterministic_sample(all_ids, min(len(all_ids), arguments.book_count * 3))
    print(
        f"English text works in catalog: {len(all_ids)}; "
        f"downloading up to {arguments.book_count}.",
        flush=True,
    )

    downloaded = 0
    for book_id in sample:
        if downloaded >= arguments.book_count:
            break
        target = arguments.output / f"pg{book_id}.txt"
        if target.exists():
            downloaded += 1
            continue
        try:
            data = _fetch(BOOK_TEXT_URL.format(book_id=book_id))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
        target.write_bytes(data)
        downloaded += 1
        if downloaded % 25 == 0:
            print(f"  {downloaded}/{arguments.book_count} books", flush=True)
        time.sleep(arguments.request_delay_seconds)

    total_bytes = sum(path.stat().st_size for path in arguments.output.glob("*.txt"))
    print(
        f"Done: {downloaded} books, ~{total_bytes / 1e6:.0f} MB in {arguments.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
