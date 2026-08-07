"""English configuration of the raw-text silver-sentence extraction.

The generic extraction mechanism lives in ``prism.data.segmentation``. This
module owns the English language decisions: the abbreviation inventory that
protects sentence boundaries in English prose — titles, Latin abbreviations,
and common organisation/measure abbreviations, covering both British and
American usage — and the predeclared quality-filter defaults for English
silver sources.

Abbreviations are lowercase and period-terminated as ``prism.data.segmentation``
requires; matching lowercases the preceding word, so capitalised titles such
as ``Mr.`` are covered by ``mr.``.
"""

from prism.data.segmentation import SentenceExtractionPolicy


ENGLISH_SILVER_ABBREVIATIONS = frozenset(
    {
        # Personal and honorific titles.
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "st.",
        "sr.",
        "jr.",
        "rev.",
        "hon.",
        "fr.",
        "messrs.",
        "mt.",
        # Ranks and offices.
        "gen.",
        "sen.",
        "rep.",
        "gov.",
        "lt.",
        "col.",
        "sgt.",
        "capt.",
        "maj.",
        "adm.",
        "pres.",
        # Latin and general abbreviations.
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "viz.",
        "cf.",
        "al.",
        "ca.",
        "ibid.",
        "a.m.",
        "p.m.",
        # Organisations, places, and measures.
        "inc.",
        "ltd.",
        "co.",
        "corp.",
        "dept.",
        "univ.",
        "ave.",
        "blvd.",
        "rd.",
        "no.",
        "approx.",
        "est.",
        "fig.",
        "figs.",
        "vol.",
        "vols.",
        "pp.",
        "p.",
        "ed.",
        "eds.",
        "pt.",
        "ch.",
        "chap.",
        "sec.",
    }
)

ENGLISH_MINIMUM_SILVER_TOKEN_COUNT = 4
ENGLISH_MINIMUM_LETTER_TOKEN_RATIO = 0.5


def english_sentence_extraction_policy(
    *,
    maximum_token_count: int,
) -> SentenceExtractionPolicy:
    return SentenceExtractionPolicy(
        abbreviation_tokens=ENGLISH_SILVER_ABBREVIATIONS,
        minimum_token_count=ENGLISH_MINIMUM_SILVER_TOKEN_COUNT,
        maximum_token_count=maximum_token_count,
        minimum_letter_token_ratio=ENGLISH_MINIMUM_LETTER_TOKEN_RATIO,
    )
