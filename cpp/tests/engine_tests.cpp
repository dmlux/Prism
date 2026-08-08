// Executes the shipped program on a recorded fixture batch and verifies the
// outputs against the values the exporter recorded alongside the artifact.

#include "prism/engine.h"

#include <cmath>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

const std::string kRoot = PRISM_REPOSITORY_ROOT;

prism::engine::InputTensor InputFromJson(const nlohmann::json& payload)
{
    prism::engine::InputTensor input;
    for (const auto& size : payload.at("shape")) {
        input.shape.sizes.push_back(size.get<int>());
    }
    const auto& data = payload.at("data");
    if (payload.at("dtype").get<std::string>() == "bool") {
        input.kind = prism::engine::InputTensor::Kind::boolean;
        for (const auto& value : data) {
            input.boolean_data.push_back(value.get<bool>() ? 1 : 0);
        }
    } else {
        input.kind = prism::engine::InputTensor::Kind::int64;
        for (const auto& value : data) {
            input.int64_data.push_back(value.get<std::int64_t>());
        }
    }
    return input;
}

void ExpectFixtureParity(const std::string& artifact)
{
    std::ifstream fixtures_file(artifact + "/fixtures.json");
    if (!fixtures_file) {
        GTEST_SKIP() << "Local artifact is not present.";
    }
    const auto fixtures = nlohmann::json::parse(fixtures_file);
    const auto& fixture = fixtures.at("fixtures").at(0);
    // The tolerance travels with the artifact: fp32 fixtures record eager
    // outputs (tight), int8 fixtures record the quantized eager twin,
    // whose gap to the XNNPACK int8 kernels is inherently wider.
    const auto tolerance = fixtures.at("comparison")
                               .value("probability_tolerance", 5e-3);

    std::vector<prism::engine::InputTensor> inputs;
    for (const auto& payload : fixture.at("inputs")) {
        inputs.push_back(InputFromJson(payload));
    }

    // Programs with separated data list their shared .ptd files in the
    // manifest; they must be loaded alongside the program.
    std::ifstream manifest_file(artifact + "/manifest.json");
    ASSERT_TRUE(manifest_file);
    const auto manifest = nlohmann::json::parse(manifest_file);
    std::vector<std::filesystem::path> data_files;
    for (const auto& data_file :
        manifest.at("programs").at(0).value("data_files", nlohmann::json::array())) {
        data_files.push_back(artifact + "/" + data_file.get<std::string>());
    }

    prism::engine::Program program(artifact + "/model-xnnpack.pte", data_files);
    const auto outputs = program.Forward(inputs);

    // The fixture records the lemma head as two top-k tensors, so it holds
    // one entry more than the program's output list.
    const auto& expected_outputs = fixture.at("expected_outputs");
    ASSERT_EQ(outputs.size() + 1, expected_outputs.size());

    // The first output is the calibrated UPOS distribution; compare it
    // against the recorded expectation at every position.
    const auto& expected_upos = expected_outputs.at(0).at("data");
    ASSERT_EQ(outputs[0].data.size(), expected_upos.size());
    float largest_difference = 0.0F;
    for (std::size_t index = 0; index < outputs[0].data.size(); ++index) {
        largest_difference = std::max(largest_difference,
            std::abs(outputs[0].data[index] - expected_upos.at(index).get<float>()));
    }
    EXPECT_LE(largest_difference, static_cast<float>(tolerance));

    // Every row must be a probability distribution over the UPOS labels.
    const auto label_count = static_cast<std::size_t>(outputs[0].shape.sizes.back());
    float row_sum = 0.0F;
    for (std::size_t index = 0; index < label_count; ++index) {
        row_sum += outputs[0].data[index];
    }
    EXPECT_NEAR(row_sum, 1.0F, 1e-3F);
    std::cout << "engine parity: max |delta| = " << largest_difference << "\n";
}

TEST(Engine, ExecutesFixtureBatchWithRecordedParity)
{
    ExpectFixtureParity(kRoot + "/models/prism-no-0.2.4");
}

// The fast artifact's fixtures record its quantized eager twin; parity
// against them validates the int8 program end to end, including the
// quantized kernels (embedding_byte) the runtime must provide.
TEST(Engine, ExecutesFastArtifactFixturesWithRecordedParity)
{
    ExpectFixtureParity(kRoot + "/models/prism-no-0.2.4-fast");
}

// The English artifact uses the ModernBERT/Ettin backbone with a different
// tokenizer and special-token template; the language-independent runtime must
// reproduce its recorded Python parity through the same engine.
TEST(Engine, ExecutesEnglishFixtureBatchWithRecordedParity)
{
    ExpectFixtureParity(kRoot + "/models/prism-en-0.1.0");
}

} // namespace
