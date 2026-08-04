/* Usage: quickstart <artifact-directory> */
#include <prism/prism_c.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <artifact-directory>\n", argv[0]);
        return 2;
    }

    prism_tagger* tagger = prism_tagger_create(argv[1]);
    if (tagger == NULL) {
        fprintf(stderr, "Prism: %s\n", prism_last_error());
        return 1;
    }
    printf("Loaded %s %s\n", prism_tagger_artifact_name(tagger),
        prism_tagger_artifact_version(tagger));

    prism_result* result = prism_tagger_tag_text(tagger, "Hun kjøpte tre gamle bøker.");
    if (result == NULL) {
        fprintf(stderr, "Prism: %s\n", prism_last_error());
        prism_tagger_destroy(tagger);
        return 1;
    }

    for (size_t s = 0; s < prism_result_sentence_count(result); ++s) {
        for (size_t t = 0; t < prism_result_token_count(result, s); ++t) {
            printf("%s\t%s\t%s\t%.3f\n",
                prism_result_token_text(result, s, t),
                prism_result_token_upos(result, s, t),
                prism_result_token_lemma(result, s, t),
                prism_result_token_upos_confidence(result, s, t));
        }
    }

    prism_result_destroy(result);
    prism_tagger_destroy(tagger);
    return 0;
}
