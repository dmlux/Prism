// End-to-end chapter benchmark: tags a UTF-8 text file twice and reports
// the cold run (including lazy program loading) and the warm run.
//
//   prism_chapter_benchmark <artifact-directory> <text-file>
//
// Point the artifact directory at different manifest variants (for example
// a single-program copy) to compare program configurations.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <prism/tagger.h>

namespace {

double MillisecondsFor(prism::tagger::Tagger& tagger, const std::string& text,
    std::size_t& token_count)
{
    const auto begin = std::chrono::steady_clock::now();
    const auto tagged = tagger.TagText(text);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin);
    token_count = 0;
    for (const auto& sentence : tagged) {
        token_count += sentence.tokens.size();
    }
    return elapsed.count();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::fprintf(stderr,
            "usage: prism_chapter_benchmark <artifact-directory> <text-file>\n");
        return 2;
    }
    std::ifstream text_file(argv[2]);
    if (!text_file) {
        std::fprintf(stderr, "cannot read text file: %s\n", argv[2]);
        return 2;
    }
    std::ostringstream buffer;
    buffer << text_file.rdbuf();
    const auto text = buffer.str();

    try {
        prism::tagger::Tagger tagger(argv[1]);
        std::size_t token_count = 0;
        const auto cold = MillisecondsFor(tagger, text, token_count);
        const auto warm = MillisecondsFor(tagger, text, token_count);
        std::cout << "tokens: " << token_count << "\n";
        std::cout << "cold: " << cold << " ms\n";
        std::cout << "warm: " << warm << " ms\n";
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
