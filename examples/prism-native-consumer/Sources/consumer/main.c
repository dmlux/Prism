/*
 * Third-party consumer proof for PrismNative: plain C, only the public
 * <prism/prism_c.h> contract, no Prism-internal include paths, no manual
 * linker flags. Exercises the complete C ABI against a local artifact:
 * raw-text and pretokenized tagging, UPOS/lemma/morphology with
 * confidences, Utf8ByteRange source mapping (including the two-fragment
 * mapping of a de-hyphenated line wrap), artifact metadata, error
 * handling, and handle cleanup. Running inference at all proves that the
 * XNNPACK backend — and with a fast artifact the quantized kernels — are
 * registered without any host-side -force_load/-all_load flags.
 *
 * Usage: consumer <artifact-directory>
 */

#include <stdio.h>
#include <string.h>

#include <prism/prism_c.h>

static int failures = 0;

static void check(int condition, const char* message)
{
    if (!condition) {
        fprintf(stderr, "FAILED: %s\n", message);
        failures += 1;
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: consumer <artifact-directory>\n");
        return 2;
    }

    /* Errors surface as NULL plus a readable message, never as a crash. */
    prism_tagger* missing = prism_tagger_create("/nonexistent/artifact");
    check(missing == NULL, "loading a nonexistent artifact must fail");
    check(strlen(prism_last_error()) > 0, "prism_last_error must describe the failure");

    check(prism_set_thread_count(6) == 1, "thread count override");

    prism_tagger* tagger = prism_tagger_create(argv[1]);
    if (tagger == NULL) {
        fprintf(stderr, "cannot load artifact: %s\n", prism_last_error());
        return 1;
    }

    /* Artifact metadata: language support comes from the manifest. */
    check(strcmp(prism_tagger_artifact_name(tagger), "prism-no") == 0, "artifact name");
    check(strlen(prism_tagger_artifact_version(tagger)) > 0, "artifact version");
    check(prism_tagger_language_tag_count(tagger) == 2, "language tag count");
    check(strcmp(prism_tagger_language_tag(tagger, 0), "nb") == 0, "language tag nb");
    check(strcmp(prism_tagger_language_tag(tagger, 1), "nn") == 0, "language tag nn");

    /* Raw text: decisions, confidences, and source ranges. */
    prism_result* result
        = prism_tagger_tag_text(tagger, "Hun kjøpte tre gamle bøker den 17. mai.");
    check(result != NULL, "raw-text tagging");
    if (result != NULL) {
        check(prism_result_sentence_count(result) == 1, "one sentence");
        check(prism_result_token_count(result, 0) == 9, "nine tokens");
        check(strcmp(prism_result_token_text(result, 0, 4), "bøker") == 0, "token text");
        check(strcmp(prism_result_token_upos(result, 0, 4), "NOUN") == 0, "UPOS");
        check(prism_result_token_upos_confidence(result, 0, 4) > 0.9, "UPOS confidence");
        check(strcmp(prism_result_token_lemma(result, 0, 4), "bok") == 0, "lemma");
        check(prism_result_token_lemma_confidence(result, 0, 4) > 0.9, "lemma confidence");
        check(prism_result_token_feature_count(result, 0, 4) >= 2, "morphology features");
        check(strstr(prism_result_token_features(result, 0, 4), "Number=Plur") != NULL,
            "feature string");
        int gender_seen = 0;
        for (size_t feature = 0;
            feature < prism_result_token_feature_count(result, 0, 4); ++feature) {
            if (strcmp(prism_result_token_feature_name(result, 0, 4, feature), "Gender")
                == 0) {
                gender_seen = 1;
                check(prism_result_token_feature_confidence(result, 0, 4, feature) > 0.5,
                    "feature confidence");
            }
        }
        check(gender_seen, "Gender feature present");

        /* Half-open UTF-8 byte ranges against the exact input. */
        check(prism_result_sentence_source_range_count(result, 0) == 1, "sentence range");
        check(prism_result_sentence_source_range_start(result, 0, 0) == 0
                && prism_result_sentence_source_range_end(result, 0, 0) == 41,
            "sentence range bytes");
        check(prism_result_token_source_range_count(result, 0, 4) == 1, "token range");
        check(prism_result_token_source_range_start(result, 0, 4, 0) == 22
                && prism_result_token_source_range_end(result, 0, 4, 0) == 28,
            "bøker occupies UTF-8 bytes [22, 28)");

        /* Out-of-range access degrades to 0/NULL. */
        check(prism_result_token_text(result, 0, 99) == NULL, "out-of-range text");
        check(prism_result_token_source_range_count(result, 0, 99) == 0,
            "out-of-range range count");
        prism_result_destroy(result);
    }

    /* A de-hyphenated line wrap: the model token stays one token, but its
     * source mapping points at the two real fragments — never at "-\n". */
    prism_result* wrapped
        = prism_tagger_tag_text(tagger, "Dette er spr\xC3\xA5k-\nmodellen.");
    check(wrapped != NULL, "wrapped-line tagging");
    if (wrapped != NULL) {
        check(prism_result_sentence_count(wrapped) == 1, "one wrapped sentence");
        check(strcmp(prism_result_token_text(wrapped, 0, 2), "spr\xC3\xA5kmodellen") == 0,
            "de-hyphenated token text");
        check(prism_result_token_source_range_count(wrapped, 0, 2) == 2,
            "two source fragments");
        check(prism_result_token_source_range_start(wrapped, 0, 2, 0) == 9
                && prism_result_token_source_range_end(wrapped, 0, 2, 0) == 15,
            "fragment 1 is 'språk'");
        check(prism_result_token_source_range_start(wrapped, 0, 2, 1) == 17
                && prism_result_token_source_range_end(wrapped, 0, 2, 1) == 25,
            "fragment 2 is 'modellen'");
        prism_result_destroy(wrapped);
    }

    /* Pretokenized input: same decisions, no invented source positions. */
    const char* tokens[] = {"Katten", "sov", "."};
    prism_result* pretokenized = prism_tagger_tag_tokens(tagger, tokens, 3);
    check(pretokenized != NULL, "pretokenized tagging");
    if (pretokenized != NULL) {
        check(strcmp(prism_result_token_upos(pretokenized, 0, 0), "NOUN") == 0,
            "pretokenized UPOS");
        check(strcmp(prism_result_token_lemma(pretokenized, 0, 1), "sove") == 0,
            "pretokenized lemma");
        check(prism_result_token_source_range_count(pretokenized, 0, 0) == 0,
            "pretokenized has no source ranges");
        prism_result_destroy(pretokenized);
    }

    prism_tagger_destroy(tagger);

    if (failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("PASSED: PrismNative consumer (artifact %s)\n", argv[1]);
    return 0;
}
