import bz2
from pathlib import Path

from prism.data import (
    WIKIPEDIA_LICENSE_ID,
    WIKIPEDIA_LICENSE_URL,
    WIKIPEDIA_NNO_CORPUS_ID,
    WIKIPEDIA_NNO_SOURCE_URL,
    iter_wikipedia_silver_sentences,
    load_silver_corpus_manifest,
    sentence_fingerprint,
    sha256_file,
    validate_silver_corpus,
    wikitext_plain_paragraphs,
    write_pretokenized_silver_corpus,
)
from prism.data.segmentation import SentenceExtractionPolicy


_POLICY = SentenceExtractionPolicy(
    abbreviation_tokens=frozenset({"f.eks."}),
    minimum_token_count=3,
    maximum_token_count=32,
    minimum_letter_token_ratio=0.5,
)

_NAMESPACE = "http://www.mediawiki.org/xml/export-0.11/"

_ARTICLE_WIKITEXT = """{{Infoboks fjell
| namn = Galdhøpiggen
| høgd = 2469
}}
'''Galdhøpiggen''' er det [[Noreg|høgaste fjellet]] i landet.<ref>NVE.</ref>

== Geografi ==
[[Fil:Galdhopiggen.jpg|thumb|Utsyn frå [[Juvasshytta]].]]
Fjellet ligg i [[Lom kommune|Lom]] og er 2469 meter høgt.

{| class="wikitable"
! Topp !! Høgd
|-
| Galdhøpiggen || 2469
|}
* Punktliste vert hoppa over.
Sjå òg [https://www.nve.no kartverket sine sider] for meir.
"""

_DUPLICATE_WIKITEXT = """Fjellet ligg i [[Lom kommune|Lom]] og er 2469 meter høgt.
Denne setninga finst berre i den andre artikkelen.
"""


def _page(
    *,
    title: str,
    namespace: str,
    page_id: str,
    wikitext: str,
    redirect_target: str | None = None,
) -> str:
    redirect_element = (
        f'<redirect title="{redirect_target}" />' if redirect_target else ""
    )
    return (
        "<page>"
        f"<title>{title}</title>"
        f"<ns>{namespace}</ns>"
        f"<id>{page_id}</id>"
        f"{redirect_element}"
        "<revision><id>9999</id>"
        f"<text>{wikitext}</text>"
        "</revision>"
        "</page>"
    )


def _write_dump(path: Path) -> None:
    document = (
        f'<mediawiki xmlns="{_NAMESPACE}" xml:lang="nn">'
        "<siteinfo><sitename>Wikipedia</sitename></siteinfo>"
        + _page(
            title="Galdhøpiggen",
            namespace="0",
            page_id="11",
            wikitext=_ARTICLE_WIKITEXT.replace("<", "&lt;").replace(">", "&gt;"),
        )
        + _page(
            title="Galdhopiggen",
            namespace="0",
            page_id="12",
            wikitext="#OMDIRIGER [[Galdhøpiggen]]",
            redirect_target="Galdhøpiggen",
        )
        + _page(
            title="Diskusjon:Galdhøpiggen",
            namespace="1",
            page_id="13",
            wikitext="Denne diskusjonssida skal ignorerast heilt her.",
        )
        + _page(
            title="Lom",
            namespace="0",
            page_id="14",
            wikitext=_DUPLICATE_WIKITEXT,
        )
        + "</mediawiki>"
    )
    path.write_bytes(bz2.compress(document.encode("utf-8")))


def test_wikitext_plain_paragraphs_strips_markup() -> None:
    paragraphs = tuple(wikitext_plain_paragraphs(_ARTICLE_WIKITEXT))

    assert paragraphs == (
        "Galdhøpiggen er det høgaste fjellet i landet.",
        "Fjellet ligg i Lom og er 2469 meter høgt.",
        "Sjå òg kartverket sine sider for meir.",
    )


def test_wikitext_plain_paragraphs_drops_residual_markup_lines() -> None:
    wikitext = (
        "Ei rein setning utan markup her.\n"
        "Ei linje med {{ubalansert mal utan slutt.\n"
        "Ei linje med www.nrk.no vert forkasta.\n"
        "__NOTOC__\n"
        "&lt;ref&gt;Escaped markup vert fanga etter unescape.&lt;/ref&gt;\n"
    )

    assert tuple(wikitext_plain_paragraphs(wikitext)) == (
        "Ei rein setning utan markup her.",
    )


def test_wikipedia_skips_redirects_and_non_articles(tmp_path: Path) -> None:
    dump_path = tmp_path / "nnwiki-pages-articles.xml.bz2"
    _write_dump(dump_path)

    sentences = tuple(
        iter_wikipedia_silver_sentences(
            archive_path=dump_path,
            extraction_policy=_POLICY,
        )
    )

    assert tuple(
        (sentence.document_id, sentence.sentence_index, sentence.model_input.tokens)
        for sentence in sentences
    ) == (
        (
            "11",
            0,
            ("Galdhøpiggen", "er", "det", "høgaste", "fjellet", "i", "landet", "."),
        ),
        (
            "11",
            1,
            (
                "Fjellet",
                "ligg",
                "i",
                "Lom",
                "og",
                "er",
                "2469",
                "meter",
                "høgt",
                ".",
            ),
        ),
        (
            "11",
            2,
            ("Sjå", "òg", "kartverket", "sine", "sider", "for", "meir", "."),
        ),
        (
            "14",
            0,
            (
                "Denne",
                "setninga",
                "finst",
                "berre",
                "i",
                "den",
                "andre",
                "artikkelen",
                ".",
            ),
        ),
    )


def test_wikipedia_respects_excluded_fingerprints(tmp_path: Path) -> None:
    dump_path = tmp_path / "nnwiki-pages-articles.xml.bz2"
    _write_dump(dump_path)
    all_sentences = tuple(
        iter_wikipedia_silver_sentences(
            archive_path=dump_path,
            extraction_policy=_POLICY,
        )
    )

    filtered = tuple(
        iter_wikipedia_silver_sentences(
            archive_path=dump_path,
            extraction_policy=_POLICY,
            excluded_sentence_fingerprints=(
                sentence_fingerprint(all_sentences[0].model_input),
            ),
        )
    )

    assert tuple(sentence.model_input.tokens for sentence in filtered) == tuple(
        sentence.model_input.tokens for sentence in all_sentences[1:]
    )


def test_wikipedia_manifest_round_trip(tmp_path: Path) -> None:
    dump_path = tmp_path / "nnwiki-pages-articles.xml.bz2"
    _write_dump(dump_path)
    output_path = tmp_path / "sentences.jsonl"
    manifest_path = tmp_path / "manifest.json"

    written_manifest = write_pretokenized_silver_corpus(
        sentences=iter_wikipedia_silver_sentences(
            archive_path=dump_path,
            extraction_policy=_POLICY,
        ),
        output_path=output_path,
        manifest_path=manifest_path,
        corpus_id=WIKIPEDIA_NNO_CORPUS_ID,
        language_tag="nn",
        source_url=WIKIPEDIA_NNO_SOURCE_URL,
        source_archive_sha256=sha256_file(dump_path),
        license_id=WIKIPEDIA_LICENSE_ID,
        license_url=WIKIPEDIA_LICENSE_URL,
        extraction_policy={"markup_removal": "prism-wikitext-plain-v1"},
    )
    loaded_manifest = load_silver_corpus_manifest(manifest_path)

    assert loaded_manifest == written_manifest
    assert loaded_manifest.sentence_count == 4
    assert loaded_manifest.document_count == 2
    assert loaded_manifest.license_id == "CC-BY-SA-4.0"
    validate_silver_corpus(sentences_path=output_path, manifest=loaded_manifest)
