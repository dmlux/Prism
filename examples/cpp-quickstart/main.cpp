// Usage: quickstart <artifact-directory>
#include <cstdio>
#include <prism>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <artifact-directory>\n", argv[0]);
        return 2;
    }

    prism::tagger::Tagger tagger(argv[1]);
    const auto& artifact = tagger.artifact();
    std::printf("Loaded %s %s\n", artifact.name().c_str(), artifact.version().c_str());

    for (const auto& sentence : tagger.TagText("Hun kjøpte tre gamle bøker.")) {
        for (const auto& token : sentence.tokens) {
            std::printf("%s\t%s\t%s\t%.3f\n", token.text.c_str(), token.upos.c_str(),
                token.lemma.c_str(), token.upos_confidence);
        }
    }
    return 0;
}
