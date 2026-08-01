"""Deterministic sentence extraction from raw silver-source page text.

Some silver sources, such as the CC0 NBdigital corpus, already ship word
segmentation and sentence boundaries. Others, such as the CC0 Nynorsk
municipal-document corpus, provide only raw OCR page text. This module turns
such raw text into conservative, high-precision ``PretokenizedSentence``
values for offline teacher labeling.

This is an explicit offline data-preparation policy, not a runtime tokenizer:
LexKeep and the public Prism API continue to supply their own tokens, and raw
text tokenization remains a separate later product decision.

The extraction intentionally prefers precision over recall. Silver sources are
large, so it is cheaper to discard headers, tables, and OCR fragments than to
teach the student from malformed sentences.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from prism.data.examples import PretokenizedSentence


SENTENCE_EXTRACTION_POLICY_VERSION = "prism-rule-segmentation-v1"

_TERMINAL_CHARACTERS = ".!?…"
_OPENING_SENTENCE_CHARACTERS = "«\"'([„“"
_CLOSING_SENTENCE_CHARACTERS = "»\"')]”"
_SENTENCE_END_PATTERN = re.compile(r"[.!?…]+")
_TOKEN_PATTERN = re.compile(
    # URLs and e-mail addresses stay one token.
    r"(?:https?://|www\.)\S+"
    r"|[^\s@]+@[^\s@]+\.\w+"
    # Numbers, dates, times, and case numbers such as 10:00 or 2020/42652-5.
    r"|\d+(?:[.,:/\-]\d+)*"
    # Dotted abbreviations such as f.eks. or bl.a. stay one token.
    r"|\w+(?:\.\w+)+\.?"
    # Words, including hyphen compounds and apostrophes.
    r"|\w+(?:[-'’]\w+)*"
    # Any other visible character becomes its own token.
    r"|\S",
    re.UNICODE,
)
_ORDINAL_BEFORE_SPLIT_PATTERN = re.compile(r"(?:^|\s)\d{1,3}$")


@dataclass(frozen=True, slots=True, kw_only=True)
class SentenceExtractionPolicy:
    """Versioned, language-configurable raw-text extraction policy.

    ``abbreviation_tokens`` must contain lowercase abbreviations including
    their trailing period, for example ``"f.eks."`` or ``"nr."``. They protect
    sentence boundaries and keep the trailing period attached to the token.
    """

    abbreviation_tokens: frozenset[str]
    minimum_token_count: int
    maximum_token_count: int
    minimum_letter_token_ratio: float

    def __post_init__(self) -> None:
        for abbreviation in self.abbreviation_tokens:
            if (
                not abbreviation
                or abbreviation != abbreviation.strip()
                or abbreviation != abbreviation.lower()
                or not abbreviation.endswith(".")
            ):
                raise ValueError(
                    "Abbreviation tokens must be lowercase, trimmed, and end "
                    f"with a period: {abbreviation!r}."
                )
        if self.minimum_token_count <= 0:
            raise ValueError("Minimum token count must be positive.")
        if self.maximum_token_count < self.minimum_token_count:
            raise ValueError(
                "Maximum token count must not be below the minimum token count."
            )
        if not 0.0 <= self.minimum_letter_token_ratio <= 1.0:
            raise ValueError("Minimum letter-token ratio must be between zero and one.")


def _merge_wrapped_lines(text: str) -> Iterator[str]:
    """Join OCR line wraps into paragraphs without joining layout lines.

    A line continues the previous paragraph only when the paragraph has no
    terminal punctuation yet and the line starts with a lowercase letter. A
    trailing hyphen before such a continuation is treated as OCR hyphenation
    and removed. Everything else, including headers and table cells, starts a
    new paragraph and is later removed by the sentence quality filters.
    """

    paragraph = ""
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.split())
        if not line:
            if paragraph:
                yield paragraph
            paragraph = ""
            continue
        continues_previous = (
            bool(paragraph)
            and paragraph[-1] not in _TERMINAL_CHARACTERS + ":"
            and line[0].islower()
        )
        if continues_previous and paragraph.endswith("-"):
            paragraph = paragraph[:-1] + line
        elif continues_previous:
            paragraph = f"{paragraph} {line}"
        else:
            if paragraph:
                yield paragraph
            paragraph = line
    if paragraph:
        yield paragraph


def _is_protected_boundary(
    *,
    paragraph: str,
    match_start: int,
    match_end: int,
    abbreviation_tokens: frozenset[str],
) -> bool:
    preceding_start = match_start
    while preceding_start > 0 and not paragraph[preceding_start - 1].isspace():
        preceding_start -= 1
    preceding_word = paragraph[preceding_start:match_end]
    if preceding_word.lower() in abbreviation_tokens:
        return True
    # Ordinal or list numbers such as "17." in "17. mai" never end a sentence.
    return bool(
        _ORDINAL_BEFORE_SPLIT_PATTERN.search(paragraph[:match_start])
        and match_end - match_start == 1
        and paragraph[match_start] == "."
    )


def _split_paragraph_sentences(
    paragraph: str,
    abbreviation_tokens: frozenset[str],
) -> Iterator[str]:
    sentence_start = 0
    for match in _SENTENCE_END_PATTERN.finditer(paragraph):
        remainder = paragraph[match.end() :].lstrip()
        if not remainder:
            continue
        starts_new_sentence = remainder[0].isupper() or (
            remainder[0] in _OPENING_SENTENCE_CHARACTERS
        )
        if not starts_new_sentence:
            continue
        if not paragraph[match.end()].isspace():
            # The terminal characters must be followed by whitespace.
            continue
        if _is_protected_boundary(
            paragraph=paragraph,
            match_start=match.start(),
            match_end=match.end(),
            abbreviation_tokens=abbreviation_tokens,
        ):
            continue
        sentence = paragraph[sentence_start : match.end()].strip()
        if sentence:
            yield sentence
        sentence_start = match.end()
    tail = paragraph[sentence_start:].strip()
    if tail:
        yield tail


def _tokenize_with_spacing(
    sentence_text: str,
    abbreviation_tokens: frozenset[str],
) -> PretokenizedSentence | None:
    raw_tokens: list[str] = []
    raw_spans: list[tuple[int, int]] = []
    for match in _TOKEN_PATTERN.finditer(sentence_text):
        raw_tokens.append(match.group())
        raw_spans.append(match.span())
    if not raw_tokens:
        return None

    # Reattach the period of listed abbreviations, and of short ordinal
    # numbers such as "17." in "17. mai", to match the UD token convention.
    # A sentence-final period after a number remains its own token.
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        span = raw_spans[index]
        period_is_attached = (
            index + 1 < len(raw_tokens)
            and raw_tokens[index + 1] == "."
            and raw_spans[index + 1][0] == span[1]
        )
        keeps_period = period_is_attached and (
            f"{token.lower()}." in abbreviation_tokens
            or (token.isdigit() and len(token) <= 3 and index + 2 < len(raw_tokens))
        )
        if keeps_period:
            tokens.append(f"{token}.")
            spans.append((span[0], raw_spans[index + 1][1]))
            index += 2
            continue
        tokens.append(token)
        spans.append(span)
        index += 1

    has_space_before = tuple(
        False if index == 0 else spans[index - 1][1] < spans[index][0]
        for index in range(len(tokens))
    )
    return PretokenizedSentence(
        tokens=tuple(tokens),
        has_space_before=has_space_before,
    )


def _passes_quality_filters(
    *,
    sentence_text: str,
    sentence: PretokenizedSentence,
    policy: SentenceExtractionPolicy,
) -> bool:
    token_count = len(sentence.tokens)
    if not policy.minimum_token_count <= token_count <= policy.maximum_token_count:
        return False
    if "�" in sentence_text:
        return False
    # After optional opening quotes or brackets the sentence must start
    # uppercase; this also rejects quoted mid-sentence fragments like
    # '"... ikkje rekna som del av ...'.
    stripped_start = sentence_text.lstrip(_OPENING_SENTENCE_CHARACTERS)
    if not stripped_start or not stripped_start[0].isupper():
        return False
    # Closing quotes or brackets may follow the terminal punctuation, as in
    # "…gyldig.»".
    stripped_end = sentence_text.rstrip(_CLOSING_SENTENCE_CHARACTERS)
    if not stripped_end or stripped_end[-1] not in _TERMINAL_CHARACTERS:
        return False
    # Two or at least four trailing periods are OCR artifacts such as
    # truncated table-of-contents dot leaders; a single period and a real
    # three-dot ellipsis remain valid sentence ends.
    trailing_period_count = len(stripped_end) - len(stripped_end.rstrip("."))
    if trailing_period_count == 2 or trailing_period_count >= 4:
        return False
    if not any(character.islower() for character in sentence_text):
        return False
    letter_token_count = sum(
        1
        for token in sentence.tokens
        if any(character.isalpha() for character in token)
    )
    return letter_token_count / token_count >= policy.minimum_letter_token_ratio


RUNTIME_SEGMENTATION_POLICY_VERSION = "prism-runtime-segmentation-v1"

# E-book extraction frequently loses the space after sentence punctuation
# ("veien.Et sekund"). A lowercase letter, terminal punctuation, and an
# immediately following uppercase or opening character never form one token
# in Norwegian prose, so restoring the space is safe; abbreviation-protected
# sentence boundaries are still consulted afterwards.
_MISSING_SENTENCE_SPACE_PATTERN = re.compile(
    r"(?<=[a-zæøå])([.!?…])(?=[A-ZÆØÅ" + re.escape(_OPENING_SENTENCE_CHARACTERS) + r"])"
)


def _restore_missing_sentence_spaces(text: str) -> str:
    return _MISSING_SENTENCE_SPACE_PATTERN.sub(r"\1 ", text)


def _chunk_sentence(
    sentence: PretokenizedSentence,
    maximum_token_count: int,
) -> Iterator[PretokenizedSentence]:
    if len(sentence.tokens) <= maximum_token_count:
        yield sentence
        return
    for start in range(0, len(sentence.tokens), maximum_token_count):
        tokens = sentence.tokens[start : start + maximum_token_count]
        has_space_before = sentence.has_space_before[
            start : start + maximum_token_count
        ]
        yield PretokenizedSentence(
            tokens=tokens,
            has_space_before=(False,) + has_space_before[1:],
        )


def segment_pretokenized_sentences(
    text: str,
    policy: SentenceExtractionPolicy,
) -> Iterator[PretokenizedSentence]:
    """Segment raw text for runtime tagging without discarding content.

    This shares the line merging, protected sentence boundaries, and UD token
    conventions with the silver extraction, but is recall-oriented: user text
    is never dropped, so the quality filters do not apply, headings and
    fragments come out as sentences, and sentences beyond the policy maximum
    are chunked into windows instead of being discarded.
    """

    for paragraph in _merge_wrapped_lines(_restore_missing_sentence_spaces(text)):
        for sentence_text in _split_paragraph_sentences(
            paragraph,
            policy.abbreviation_tokens,
        ):
            sentence = _tokenize_with_spacing(
                sentence_text,
                policy.abbreviation_tokens,
            )
            if sentence is None:
                continue
            yield from _chunk_sentence(sentence, policy.maximum_token_count)


def extract_pretokenized_sentences(
    text: str,
    policy: SentenceExtractionPolicy,
) -> Iterator[PretokenizedSentence]:
    """Extract conservative, high-precision sentences from raw page text."""

    for paragraph in _merge_wrapped_lines(text):
        for sentence_text in _split_paragraph_sentences(
            paragraph,
            policy.abbreviation_tokens,
        ):
            sentence = _tokenize_with_spacing(
                sentence_text,
                policy.abbreviation_tokens,
            )
            if sentence is None:
                continue
            if _passes_quality_filters(
                sentence_text=sentence_text,
                sentence=sentence,
                policy=policy,
            ):
                yield sentence
