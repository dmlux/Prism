// Artifact manifest metadata: name, version, and language tags come from
// manifest.json — never from directory names — and missing metadata fails
// loudly instead of being guessed.

#include "prism/artifact.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace {

constexpr const char* kLabelsJson = R"({
  "labels_format_version": 1,
  "schema": {
    "upos": {"labels": ["NOUN"]},
    "morphology": {"features": []},
    "lemma_rules": {"rules": []}
  },
  "character_vocabulary": {"characters": []},
  "maximum_character_count": 32
})";

std::string ManifestJson(const std::string& metadata_fields)
{
    return R"({
  "manifest_format_version": 1,
)" + metadata_fields
        + R"(
  "labels_file": "labels.json",
  "tokenizer": {"file_name": "vocabulary.json", "padding_token_id": 3},
  "programs": [
    {
      "file_name": "model-xnnpack.pte",
      "backend": "xnnpack",
      "precision": "fp32",
      "shapes": {"batch_size": 8, "subword_count": 24, "token_count": 16}
    }
  ]
})";
}

class TemporaryArtifact {
public:
    TemporaryArtifact(const std::string& manifest, const std::string& labels)
        : directory_(std::filesystem::temp_directory_path()
              / ("prism-artifact-test-" + std::to_string(std::rand())))
    {
        std::filesystem::create_directories(directory_);
        std::ofstream(directory_ / "manifest.json") << manifest;
        std::ofstream(directory_ / "labels.json") << labels;
    }

    ~TemporaryArtifact() { std::filesystem::remove_all(directory_); }

    const std::filesystem::path& directory() const { return directory_; }

private:
    std::filesystem::path directory_;
};

TEST(ArtifactMetadata, ExposesNameVersionAndLanguageTagsInManifestOrder)
{
    const TemporaryArtifact artifact(
        ManifestJson(R"(  "artifact_name": "prism-no",
  "artifact_version": "0.2.2",
  "language_tags": ["nb", "nn"],)"),
        kLabelsJson);

    const prism::artifact::Artifact loaded(artifact.directory());

    EXPECT_EQ(loaded.name(), "prism-no");
    EXPECT_EQ(loaded.version(), "0.2.2");
    EXPECT_EQ(loaded.language_tags(), (std::vector<std::string>{"nb", "nn"}));
}

TEST(ArtifactMetadata, MissingMetadataFailsLoudly)
{
    const TemporaryArtifact artifact(
        ManifestJson(R"(  "artifact_name": "prism-no",
  "artifact_version": "0.2.2",)"),
        kLabelsJson);

    EXPECT_THROW(prism::artifact::Artifact{artifact.directory()}, std::runtime_error);
}

TEST(ArtifactMetadata, InvalidLanguageTagsFailLoudly)
{
    const TemporaryArtifact artifact(
        ManifestJson(R"(  "artifact_name": "prism-no",
  "artifact_version": "0.2.2",
  "language_tags": "nb",)"),
        kLabelsJson);

    EXPECT_THROW(prism::artifact::Artifact{artifact.directory()}, std::runtime_error);
}

} // namespace
