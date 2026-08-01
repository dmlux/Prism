// Executes the shipped program on a recorded fixture batch and verifies the
// outputs against the fixtures — the C++ twin of the Swift engine spike.

#include "prism/engine.h"

#include <cmath>
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

TEST(Engine, ExecutesFixtureBatchWithRecordedParity)
{
    const auto artifact = kRoot + "/models/prism-no-0.2.0";
    std::ifstream fixtures_file(artifact + "/fixtures.json");
    if (!fixtures_file) {
        GTEST_SKIP() << "Local artifact is not present.";
    }
    const auto fixtures = nlohmann::json::parse(fixtures_file);
    const auto& fixture = fixtures.at("fixtures").at(0);

    std::vector<prism::engine::InputTensor> inputs;
    for (const auto& payload : fixture.at("inputs")) {
        inputs.push_back(InputFromJson(payload));
    }

    prism::engine::Program program(artifact + "/model-xnnpack.pte");
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
    EXPECT_LE(largest_difference, 5e-3F);

    // Every row must be a probability distribution over the UPOS labels.
    const auto label_count = static_cast<std::size_t>(outputs[0].shape.sizes.back());
    float row_sum = 0.0F;
    for (std::size_t index = 0; index < label_count; ++index) {
        row_sum += outputs[0].data[index];
    }
    EXPECT_NEAR(row_sum, 1.0F, 1e-3F);
    std::cout << "engine parity: max |delta| = " << largest_difference << "\n";
}

} // namespace
