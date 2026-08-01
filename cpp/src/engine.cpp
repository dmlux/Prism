#include "prism/engine.h"

#include <stdexcept>
#include <string>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>

namespace prism::engine {

namespace {

bool g_thread_count_overridden = false;

bool ResizeThreadpool(std::size_t thread_count)
{
    if (thread_count == 0) {
        return false;
    }
    auto* threadpool = ::executorch::extension::threadpool::get_threadpool();
    return threadpool != nullptr
        && threadpool->_unsafe_reset_threadpool(
            static_cast<std::uint32_t>(thread_count));
}

} // namespace

std::size_t ThreadCount()
{
    auto* threadpool = ::executorch::extension::threadpool::get_threadpool();
    return threadpool == nullptr ? 0 : threadpool->get_thread_count();
}

bool SetThreadCount(std::size_t thread_count)
{
    if (!ResizeThreadpool(thread_count)) {
        return false;
    }
    g_thread_count_overridden = true;
    return true;
}

void SetDefaultThreadCount(std::size_t thread_count)
{
    if (!g_thread_count_overridden && thread_count < ThreadCount()) {
        ResizeThreadpool(thread_count);
    }
}

using ::executorch::aten::ScalarType;
using ::executorch::extension::Module;
using ::executorch::extension::make_tensor_ptr;
using ::executorch::runtime::EValue;

namespace {

std::vector<std::string> DataFileStrings(
    const std::vector<std::filesystem::path>& data_files)
{
    std::vector<std::string> strings;
    strings.reserve(data_files.size());
    for (const auto& data_file : data_files) {
        strings.push_back(data_file.string());
    }
    return strings;
}

} // namespace

struct Program::Implementation {
    Implementation(
        const std::filesystem::path& program_path,
        const std::vector<std::filesystem::path>& data_files)
        : module(program_path.string(), DataFileStrings(data_files))
    {
        const auto error = module.load();
        if (error != ::executorch::runtime::Error::Ok) {
            throw std::runtime_error(
                "Cannot load ExecuTorch program: " + program_path.string());
        }
    }

    Module module;
};

Program::Program(
    const std::filesystem::path& program_path,
    const std::vector<std::filesystem::path>& data_files)
    : implementation_(std::make_unique<Implementation>(program_path, data_files))
{
}

Program::~Program() = default;

std::vector<OutputTensor> Program::Forward(const std::vector<InputTensor>& inputs)
{
    std::vector<EValue> values;
    values.reserve(inputs.size());
    // TensorPtrs own their storage; keep them alive across the call.
    std::vector<::executorch::extension::TensorPtr> tensors;
    tensors.reserve(inputs.size());

    for (const auto& input : inputs) {
        std::vector<::executorch::aten::SizesType> sizes(
            input.shape.sizes.begin(), input.shape.sizes.end());
        if (input.kind == InputTensor::Kind::int64) {
            tensors.push_back(make_tensor_ptr(
                std::move(sizes),
                std::vector<std::int64_t>(input.int64_data),
                {},
                {},
                ScalarType::Long));
        } else {
            tensors.push_back(make_tensor_ptr(
                std::move(sizes),
                std::vector<std::uint8_t>(input.boolean_data),
                {},
                {},
                ScalarType::Bool));
        }
        values.emplace_back(tensors.back());
    }

    auto result = implementation_->module.forward(values);
    if (!result.ok()) {
        throw std::runtime_error(
            "ExecuTorch forward failed with error "
            + std::to_string(static_cast<int>(result.error())));
    }

    std::vector<OutputTensor> outputs;
    outputs.reserve(result->size());
    for (const auto& value : *result) {
        if (!value.isTensor()) {
            throw std::runtime_error("ExecuTorch output is not a tensor.");
        }
        const auto& tensor = value.toTensor();
        OutputTensor output;
        for (const auto size : tensor.sizes()) {
            output.shape.sizes.push_back(static_cast<int>(size));
        }
        const auto* data = tensor.const_data_ptr<float>();
        output.data.assign(data, data + tensor.numel());
        outputs.push_back(std::move(output));
    }
    return outputs;
}

} // namespace prism::engine
