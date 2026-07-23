"""Dataset-specific preprocessing contracts."""

from prism.data.batches import (
    TokenTaskTargetBatch,
    build_token_task_target_batch,
)
from prism.data.examples import (
    PretokenizedSentence,
    SupervisedCorpus,
    SupervisedSentence,
    TokenTargets,
)
from prism.data.norwegian import (
    NorwegianUdMorphologyDecoder,
    NorwegianUdLemmaDecoder,
    build_norwegian_schema,
    build_norwegian_ud_lemma_decoder,
    encode_norwegian_sentence,
    encode_norwegian_sentences,
    normalize_norwegian_ud_lemma,
)
from prism.data.nbdigital import (
    NBDIGITAL_CORPUS_ID,
    NBDIGITAL_LANGUAGE_TAG,
    NBDIGITAL_LICENSE_ID,
    NBDIGITAL_LICENSE_URL,
    NBDIGITAL_SOURCE_URL,
    NbDigitalDocumentMetadata,
    iter_nbdigital_silver_sentences,
    parse_nbdigital_document_name,
)
from prism.data.silver import (
    SILVER_CORPUS_FORMAT_VERSION,
    PretokenizedSilverSentence,
    SilverCorpusManifest,
    iter_pretokenized_silver_sentences,
    load_silver_corpus_manifest,
    sentence_fingerprint,
    sha256_file,
    validate_silver_corpus,
    write_pretokenized_silver_corpus,
)

__all__ = [
    "normalize_norwegian_ud_lemma",
    "NorwegianUdLemmaDecoder",
    "NorwegianUdMorphologyDecoder",
    "build_norwegian_ud_lemma_decoder",
    "encode_norwegian_sentence",
    "encode_norwegian_sentences",
    "PretokenizedSentence",
    "SupervisedSentence",
    "SupervisedCorpus",
    "TokenTargets",
    "build_norwegian_schema",
    "TokenTaskTargetBatch",
    "build_token_task_target_batch",
    "NBDIGITAL_CORPUS_ID",
    "NBDIGITAL_LANGUAGE_TAG",
    "NBDIGITAL_LICENSE_ID",
    "NBDIGITAL_LICENSE_URL",
    "NBDIGITAL_SOURCE_URL",
    "NbDigitalDocumentMetadata",
    "iter_nbdigital_silver_sentences",
    "parse_nbdigital_document_name",
    "SILVER_CORPUS_FORMAT_VERSION",
    "PretokenizedSilverSentence",
    "SilverCorpusManifest",
    "iter_pretokenized_silver_sentences",
    "load_silver_corpus_manifest",
    "sentence_fingerprint",
    "sha256_file",
    "validate_silver_corpus",
    "write_pretokenized_silver_corpus",
]
