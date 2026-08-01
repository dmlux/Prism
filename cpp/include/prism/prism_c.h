/*
 * C ABI for the Prism tagger.
 *
 * A stable, plain-C surface for embedding Prism in applications whose core
 * links C libraries only (or crosses a language boundary such as a foreign
 * function interface). Every function is safe to call from C: only opaque
 * handles, C strings, and scalar types appear in the signatures.
 *
 * Ownership: prism_tagger_create/prism_result destroy pairs own their
 * handles; every const char* returned by an accessor stays valid for the
 * lifetime of the prism_result it came from. Handles are not thread-safe;
 * results are immutable after creation and may be read from any thread.
 *
 * Errors: functions returning a pointer return NULL on failure, and
 * prism_last_error() returns a UTF-8 description of the most recent
 * failure on the calling thread.
 */

#ifndef PRISM_C_H
#define PRISM_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct prism_tagger prism_tagger;
typedef struct prism_result prism_result;

/* Overrides the CPU backend thread count for the whole process (the
 * tagger otherwise installs a measured default). Call before creating
 * taggers; returns 1 on success, 0 on failure. */
int prism_set_thread_count(size_t thread_count);

/* Loads the artifact (manifest, labels, vocabulary, programs) from the
 * given directory. Returns NULL and sets the thread's error on failure. */
prism_tagger* prism_tagger_create(const char* artifact_directory);
void prism_tagger_destroy(prism_tagger* tagger);

/* UTF-8 description of the most recent failure on the calling thread;
 * empty string when no failure has been recorded. The pointer stays valid
 * until the next failing Prism call on the same thread. */
const char* prism_last_error(void);

/* Segments raw UTF-8 text (expected in Unicode NFC) and tags every
 * sentence. Returns NULL and sets the thread's error on failure. */
prism_result* prism_tagger_tag_text(prism_tagger* tagger, const char* utf8_text);

/* Tags one application-supplied sentence of word tokens (space assumed
 * between words). Returns NULL and sets the thread's error on failure. */
prism_result* prism_tagger_tag_tokens(
    prism_tagger* tagger, const char* const* tokens, size_t token_count);

void prism_result_destroy(prism_result* result);

/* Out-of-range indices yield 0 for counts and confidences and NULL for
 * strings; they never abort. */
size_t prism_result_sentence_count(const prism_result* result);
size_t prism_result_token_count(const prism_result* result, size_t sentence);

const char* prism_result_token_text(
    const prism_result* result, size_t sentence, size_t token);
/* 1 when the token is preceded by a space in the original text. */
int prism_result_token_has_space_before(
    const prism_result* result, size_t sentence, size_t token);

/* Universal part-of-speech tag with its calibrated confidence. */
const char* prism_result_token_upos(
    const prism_result* result, size_t sentence, size_t token);
double prism_result_token_upos_confidence(
    const prism_result* result, size_t sentence, size_t token);

/* Predicted morphology features, iterated by index in alphabetical name
 * order. Values of a multi-valued feature are joined with ','; the
 * confidence of a multi-valued feature is its least certain value. */
size_t prism_result_token_feature_count(
    const prism_result* result, size_t sentence, size_t token);
const char* prism_result_token_feature_name(
    const prism_result* result, size_t sentence, size_t token, size_t feature);
const char* prism_result_token_feature_value(
    const prism_result* result, size_t sentence, size_t token, size_t feature);
double prism_result_token_feature_confidence(
    const prism_result* result, size_t sentence, size_t token, size_t feature);
/* All features as one CoNLL-U style string ("Gender=Fem|Number=Plur");
 * empty string when the token carries no features. */
const char* prism_result_token_features(
    const prism_result* result, size_t sentence, size_t token);

const char* prism_result_token_lemma(
    const prism_result* result, size_t sentence, size_t token);
double prism_result_token_lemma_confidence(
    const prism_result* result, size_t sentence, size_t token);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PRISM_C_H */
