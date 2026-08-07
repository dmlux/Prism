#include "prism/artifact.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <tuple>

#include <nlohmann/json.hpp>

namespace prism::artifact {

namespace {

nlohmann::json ParseJsonFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Missing artifact file: " + path.string());
    }
    return nlohmann::json::parse(file);
}

// Splits UTF-8 text into codepoint-sized byte ranges; lemma edit rules count
// removals in codepoints, never in bytes.
std::vector<std::size_t> CodepointOffsets(const std::string& text)
{
    std::vector<std::size_t> offsets;
    for (std::size_t index = 0; index < text.size(); ++index) {
        // Continuation bytes are 0b10xxxxxx; every other byte starts a codepoint.
        if ((static_cast<unsigned char>(text[index]) & 0xC0U) != 0x80U) {
            offsets.push_back(index);
        }
    }
    offsets.push_back(text.size());
    return offsets;
}

} // namespace

std::optional<std::string> LemmaEditRule::Apply(const std::string& token) const
{
    const auto offsets = CodepointOffsets(token);
    const auto length = offsets.size() - 1;
    if (static_cast<std::size_t>(prefix_removal + suffix_removal) > length) {
        return std::nullopt;
    }
    const auto begin = offsets[static_cast<std::size_t>(prefix_removal)];
    const auto end = offsets[length - static_cast<std::size_t>(suffix_removal)];
    return prefix_addition + token.substr(begin, end - begin) + suffix_addition;
}

Artifact::Artifact(const std::filesystem::path& directory)
    : directory_(directory)
{
    const auto manifest = ParseJsonFile(directory / "manifest.json");

    // Consumers decide language support from these manifest values; fail
    // loudly instead of guessing from directory or artifact names.
    for (const auto* required : {"artifact_name", "artifact_version", "language_tags"}) {
        if (!manifest.contains(required)) {
            throw std::runtime_error(
                std::string("manifest.json misses required metadata: ") + required);
        }
    }
    name_ = manifest.at("artifact_name").get<std::string>();
    version_ = manifest.at("artifact_version").get<std::string>();
    if (!manifest.at("language_tags").is_array()) {
        throw std::runtime_error("manifest.json language_tags must be an array of strings.");
    }
    for (const auto& tag : manifest.at("language_tags")) {
        language_tags_.push_back(tag.get<std::string>());
    }

    // Optional: raw-text segmentation abbreviations. Absent in legacy
    // artifacts, in which case the runtime uses a built-in default.
    if (manifest.contains("segmentation")) {
        const auto& segmentation = manifest.at("segmentation");
        if (segmentation.contains("abbreviations")) {
            for (const auto& abbreviation : segmentation.at("abbreviations")) {
                segmentation_abbreviations_.push_back(abbreviation.get<std::string>());
            }
        }
    }

    const auto& tokenizer = manifest.at("tokenizer");
    tokenizer_.file_name = tokenizer.at("file_name").get<std::string>();
    tokenizer_.padding_token_id = tokenizer.at("padding_token_id").get<std::int64_t>();

    for (const auto& entry : manifest.at("programs")) {
        Program program;
        program.file_name = entry.at("file_name").get<std::string>();
        program.backend = entry.at("backend").get<std::string>();
        program.precision = entry.at("precision").get<std::string>();
        const auto& shapes = entry.at("shapes");
        program.shapes.batch_size = shapes.at("batch_size").get<int>();
        program.shapes.subword_count = shapes.at("subword_count").get<int>();
        program.shapes.token_count = shapes.at("token_count").get<int>();
        if (shapes.contains("character_count") && !shapes.at("character_count").is_null()) {
            program.shapes.character_count = shapes.at("character_count").get<int>();
        }
        if (entry.contains("data_files")) {
            for (const auto& data_file : entry.at("data_files")) {
                program.data_files.push_back(data_file.get<std::string>());
            }
        }
        // The C++ engine executes on the CPU; other backends belong to
        // platform-specific runtimes and are skipped here.
        if (program.backend == "xnnpack") {
            programs_.push_back(std::move(program));
        }
    }
    if (programs_.empty()) {
        throw std::runtime_error("Artifact provides no CPU (xnnpack) program.");
    }
    std::sort(programs_.begin(), programs_.end(), [](const auto& a, const auto& b) {
        return std::tuple(a.shapes.subword_count, a.shapes.token_count)
            < std::tuple(b.shapes.subword_count, b.shapes.token_count);
    });

    const auto labels_file = manifest.at("labels_file").get<std::string>();
    const auto labels = ParseJsonFile(directory / labels_file);
    const auto& schema = labels.at("schema");

    for (const auto& label : schema.at("upos").at("labels")) {
        labels_.upos_labels.push_back(label.get<std::string>());
    }
    for (const auto& entry : schema.at("morphology").at("features")) {
        MorphologyFeature feature;
        feature.name = entry.at("name").get<std::string>();
        for (const auto& value : entry.at("values")) {
            feature.values.push_back(value.get<std::string>());
        }
        feature.allows_multiple_values = entry.at("allows_multiple_values").get<bool>();
        labels_.features.push_back(std::move(feature));
    }
    for (const auto& entry : schema.at("lemma_rules").at("rules")) {
        LemmaEditRule rule;
        rule.prefix_removal = entry.at("prefix_removal").get<int>();
        rule.suffix_removal = entry.at("suffix_removal").get<int>();
        rule.prefix_addition = entry.at("prefix_addition").get<std::string>();
        rule.suffix_addition = entry.at("suffix_addition").get<std::string>();
        labels_.lemma_rules.push_back(std::move(rule));
    }
    for (const auto& character : labels.at("character_vocabulary").at("characters")) {
        labels_.character_vocabulary.push_back(character.get<std::string>());
    }
    labels_.maximum_character_count = labels.at("maximum_character_count").get<int>();
}

} // namespace prism::artifact
