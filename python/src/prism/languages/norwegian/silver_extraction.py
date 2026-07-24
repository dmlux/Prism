"""Norwegian configuration of the raw-text silver-sentence extraction.

The generic extraction mechanism lives in ``prism.data.segmentation``. This
module owns the Norwegian language decisions: the abbreviation inventory that
protects sentence boundaries in Bokmål and Nynorsk administrative prose, and
the predeclared quality-filter defaults for Norwegian silver sources.
"""

from prism.data.segmentation import SentenceExtractionPolicy


NORWEGIAN_SILVER_ABBREVIATIONS = frozenset(
    {
        "adm.",
        "ang.",
        "bl.a.",
        "ca.",
        "d.v.s.",
        "dr.",
        "dvs.",
        "eks.",
        "ekskl.",
        "evt.",
        "f.eks.",
        "f.o.m.",
        "fylkeskomm.",
        "hhv.",
        "ifm.",
        "iht.",
        "inkl.",
        "jf.",
        "jfr.",
        "kap.",
        "kfr.",
        "kl.",
        "kr.",
        "m.a.",
        "m.fl.",
        "m.m.",
        "m.v.",
        "mht.",
        "mill.",
        "mrd.",
        "mv.",
        "nr.",
        "osv.",
        "p.g.a.",
        "pga.",
        "pkt.",
        "ref.",
        "saksnr.",
        "st.",
        "t.o.m.",
        "tlf.",
        "vedr.",
    }
)

NORWEGIAN_MINIMUM_SILVER_TOKEN_COUNT = 4
NORWEGIAN_MINIMUM_LETTER_TOKEN_RATIO = 0.5


def norwegian_sentence_extraction_policy(
    *,
    maximum_token_count: int,
) -> SentenceExtractionPolicy:
    return SentenceExtractionPolicy(
        abbreviation_tokens=NORWEGIAN_SILVER_ABBREVIATIONS,
        minimum_token_count=NORWEGIAN_MINIMUM_SILVER_TOKEN_COUNT,
        maximum_token_count=maximum_token_count,
        minimum_letter_token_ratio=NORWEGIAN_MINIMUM_LETTER_TOKEN_RATIO,
    )
