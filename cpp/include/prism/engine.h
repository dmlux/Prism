// ExecuTorch program execution for Prism artifacts.
//
// Loads one lowered .pte program and executes fixed-shape batches. The
// ExecuTorch dependency stays behind this boundary: the header exposes only
// standard types, and the runtime is version-pinned to the exporter that
// produced the artifacts (see cpp/CMakeLists.txt).

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace prism::engine {

struct TensorShape {
    std::vector<int> sizes;
};

// One program input: int64 identifiers or a boolean mask (stored as bytes).
struct InputTensor {
    enum class Kind { int64, boolean };

    Kind kind = Kind::int64;
    TensorShape shape;
    std::vector<std::int64_t> int64_data;
    std::vector<std::uint8_t> boolean_data;
};

struct OutputTensor {
    TensorShape shape;
    std::vector<float> data;
};

class Program {
public:
    // Loads a program plus the shared .ptd files holding externally stored
    // weights (empty for programs that embed their weights). Throws
    // std::runtime_error when loading fails.
    explicit Program(
        const std::filesystem::path& program_path,
        const std::vector<std::filesystem::path>& data_files = {});
    ~Program();

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    // Executes the forward method; throws std::runtime_error on failure.
    std::vector<OutputTensor> Forward(const std::vector<InputTensor>& inputs);

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace prism::engine
