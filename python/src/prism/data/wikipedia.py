"""CC BY-SA silver-source ingestion for Wikipedia XML dumps.

The Nynorsk register gap in the sakspapir corpus motivates a thematically
broad, modern-orthography second source. Wikimedia publishes complete
``pages-articles`` XML dumps whose revision text is raw wikitext, so this
adapter combines a streaming XML reader with a deliberately conservative
wikitext-to-plain-text step and the shared rule-based sentence extraction
from ``prism.data.segmentation``.

Markup handling intentionally prefers precision over recall: templates,
tables, references, images, and any line that still carries markup residue
after cleaning are discarded rather than repaired. Wikipedia is large, so it
is cheaper to drop a doubtful paragraph than to teach the student from a
malformed one. The cleaning policy is versioned so label manifests can pin
it, exactly like the sentence-extraction policy version.
"""

import bz2
import html
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import IO

from prism.data.segmentation import (
    SentenceExtractionPolicy,
    extract_pretokenized_sentences,
)
from prism.data.silver import PretokenizedSilverSentence, sentence_fingerprint


WIKIPEDIA_NNO_CORPUS_ID = "nnwiki-pages-articles"
WIKIPEDIA_NNO_LANGUAGE_TAG = "nn"
WIKIPEDIA_NNO_SOURCE_URL = (
    "https://dumps.wikimedia.org/nnwiki/latest/"
    "nnwiki-latest-pages-articles.xml.bz2"
)
WIKIPEDIA_LICENSE_ID = "CC-BY-SA-4.0"
WIKIPEDIA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
WIKITEXT_EXTRACTION_VERSION = "prism-wikitext-plain-v1"

_ARTICLE_NAMESPACE = "0"

_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_REMOVED_ELEMENT_PATTERN = re.compile(
    r"<(ref|math|gallery|code|source|syntaxhighlight|timeline|score|pre|nowiki)"
    r"\b[^<>]*?/>"
    r"|<(ref|math|gallery|code|source|syntaxhighlight|timeline|score|pre|nowiki)"
    r"\b[^<>]*>.*?</\2\s*>",
    re.DOTALL | re.IGNORECASE,
)
_TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
_WIKI_LINK_PATTERN = re.compile(r"\[\[([^\[\]]*)\]\]")
_EXTERNAL_LINK_PATTERN = re.compile(r"\[(?:https?|ftp)://[^\s\]]*(?:\s+([^\]]*))?\]")
_HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^<>]*>")
_MAGIC_WORD_PATTERN = re.compile(r"__[A-ZÆØÅ]+__")

_SKIPPED_LINE_PREFIXES = ("=", "*", "#", ":", ";", "|", "!", "{", "_")
_RESIDUAL_MARKUP_MARKERS = ("{", "}", "[", "]", "|", "<", ">", "://", "www.")


def _replace_wiki_link(match: re.Match[str]) -> str:
    inner = match.group(1)
    target, separator, label = inner.partition("|")
    if ":" in target:
        # Namespace links (files, categories, interwiki) are dropped whole;
        # captions are not worth the risk of leaking image parameters.
        return ""
    if separator:
        return label.rsplit("|", 1)[-1]
    return target.split("#", 1)[0]


def _remove_tables(text: str) -> str:
    """Drop line-oriented ``{| ... |}`` table blocks, including nesting."""

    kept_lines: list[str] = []
    table_depth = 0
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("{|"):
            table_depth += 1
            continue
        if table_depth > 0:
            if stripped.startswith("|}"):
                table_depth -= 1
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _substitute_until_stable(pattern: re.Pattern[str], text: str) -> str:
    while True:
        replaced = pattern.sub("", text)
        if replaced == text:
            return replaced
        text = replaced


def wikitext_plain_paragraphs(wikitext: str) -> Iterator[str]:
    """Yield prose paragraphs of one article as plain text.

    Headings, lists, tables, templates, references, and any line that still
    contains markup residue after cleaning are discarded.
    """

    text = _COMMENT_PATTERN.sub("", wikitext)
    text = _substitute_until_stable(_REMOVED_ELEMENT_PATTERN, text)
    text = _remove_tables(text)
    text = _substitute_until_stable(_TEMPLATE_PATTERN, text)
    while True:
        replaced = _WIKI_LINK_PATTERN.sub(_replace_wiki_link, text)
        if replaced == text:
            break
        text = replaced
    text = _EXTERNAL_LINK_PATTERN.sub(
        lambda match: match.group(1) or "",
        text,
    )
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = _MAGIC_WORD_PATTERN.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    text = html.unescape(text)

    for line in text.split("\n"):
        paragraph = line.strip()
        if not paragraph or paragraph.startswith(_SKIPPED_LINE_PREFIXES):
            continue
        if any(marker in paragraph for marker in _RESIDUAL_MARKUP_MARKERS):
            continue
        yield paragraph


def _open_dump(archive_path: Path) -> IO[bytes]:
    if archive_path.name.endswith(".bz2"):
        return bz2.open(archive_path, "rb")
    return archive_path.open("rb")


def _find_child_text(element: ElementTree.Element, path: str) -> str | None:
    child = element.find(path)
    if child is None:
        return None
    return child.text or ""


def _iter_article_pages(stream: IO[bytes]) -> Iterator[tuple[str, str]]:
    """Yield ``(page_id, wikitext)`` for every non-redirect article page."""

    events = ElementTree.iterparse(stream, events=("start", "end"))
    _, root = next(events)
    for event, element in events:
        if event != "end":
            continue
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name != "page":
            continue
        namespace = _find_child_text(element, "{*}ns")
        page_id = _find_child_text(element, "{*}id")
        if namespace is None or page_id is None:
            raise ValueError("Wikipedia page is missing its ns or id element.")
        is_redirect = element.find("{*}redirect") is not None
        wikitext = _find_child_text(element, "{*}revision/{*}text")
        if (
            namespace.strip() == _ARTICLE_NAMESPACE
            and not is_redirect
            and wikitext
        ):
            yield page_id.strip(), wikitext
        root.clear()


def iter_wikipedia_silver_sentences(
    *,
    archive_path: Path,
    extraction_policy: SentenceExtractionPolicy,
    excluded_sentence_fingerprints: Collection[str] = (),
) -> Iterator[PretokenizedSilverSentence]:
    seen_fingerprints: set[str] = set()
    excluded_fingerprints = set(excluded_sentence_fingerprints)
    with _open_dump(archive_path) as stream:
        for page_id, wikitext in _iter_article_pages(stream):
            sentence_index = 0
            for paragraph in wikitext_plain_paragraphs(wikitext):
                for model_input in extract_pretokenized_sentences(
                    paragraph,
                    extraction_policy,
                ):
                    fingerprint = sentence_fingerprint(model_input)
                    if (
                        fingerprint in excluded_fingerprints
                        or fingerprint in seen_fingerprints
                    ):
                        continue
                    seen_fingerprints.add(fingerprint)
                    yield PretokenizedSilverSentence(
                        document_id=page_id,
                        sentence_index=sentence_index,
                        model_input=model_input,
                    )
                    sentence_index += 1
