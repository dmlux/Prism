#include "prism/tagger.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "prism/engine.h"
#include "prism/subword.h"

namespace prism::tagger {

namespace {

// Reserved character-CNN identifiers; literal characters start at 5 in the
// order of the artifact's character vocabulary.
constexpr std::int64_t kCharacterPaddingId = 0;
constexpr std::int64_t kCharacterUnknownId = 1;
constexpr std::int64_t kCharacterStartId = 2;
constexpr std::int64_t kCharacterEndId = 3;
constexpr std::int64_t kCharacterTruncationId = 4;
constexpr std::int64_t kFirstLiteralCharacterId = 5;

// Splits UTF-8 text into one string per codepoint for vocabulary lookups.
std::vector<std::string> SplitCodepoints(const std::string& text)
{
    std::vector<std::string> codepoints;
    std::size_t start = 0;
    for (std::size_t index = 1; index <= text.size(); ++index) {
        if (index == text.size()
            || (static_cast<unsigned char>(text[index]) & 0xC0U) != 0x80U) {
            codepoints.push_back(text.substr(start, index - start));
            start = index;
        }
    }
    return codepoints;
}

struct Argmax {
    std::size_t index = 0;
    float value = -std::numeric_limits<float>::infinity();
};

Argmax ArgmaxIn(const std::vector<float>& values, std::size_t offset, std::size_t count)
{
    Argmax best;
    for (std::size_t index = 0; index < count; ++index) {
        if (values[offset + index] > best.value) {
            best.value = values[offset + index];
            best.index = index;
        }
    }
    return best;
}

void ValidateRangeList(const std::vector<Utf8ByteRange>& ranges, const char* what,
    std::size_t& previous_end)
{
    for (const auto& range : ranges) {
        if (range.start >= range.end) {
            throw std::invalid_argument(
                std::string(what) + " contains an empty or inverted Utf8ByteRange.");
        }
        if (range.start < previous_end) {
            throw std::invalid_argument(
                std::string(what) + " contains unordered or overlapping Utf8ByteRanges.");
        }
        previous_end = range.end;
    }
}

// Caller-supplied source mappings must uphold the Utf8ByteRange contract
// before they travel through chunking and batching. Codepoint alignment
// cannot be checked here because the raw text is not available; it stays
// the caller's contract.
void ValidateSourceMapping(const segmentation::PretokenizedSentence& sentence)
{
    std::size_t sentence_end = 0;
    ValidateRangeList(
        sentence.source_ranges, "PretokenizedSentence::source_ranges", sentence_end);
    if (sentence.token_source_ranges.empty()) {
        return;
    }
    if (sentence.token_source_ranges.size() != sentence.tokens.size()) {
        throw std::invalid_argument(
            "PretokenizedSentence::token_source_ranges must be empty or hold "
            "exactly one range list per token.");
    }
    // Ordering and non-overlap hold across the whole sentence, so repeated
    // identical tokens stay bound to their own occurrences.
    std::size_t token_end = 0;
    for (const auto& token_ranges : sentence.token_source_ranges) {
        if (token_ranges.empty()) {
            throw std::invalid_argument(
                "PretokenizedSentence::token_source_ranges contains a token "
                "without any Utf8ByteRange.");
        }
        ValidateRangeList(
            token_ranges, "PretokenizedSentence::token_source_ranges", token_end);
    }
}

// Build the runtime segmentation policy from the artifact's manifest-carried
// abbreviations, falling back to the built-in Norwegian policy for legacy
// artifacts that predate the field.
segmentation::SegmentationPolicy PolicyFromArtifact(
    const artifact::Artifact& artifact, std::size_t maximum_token_count)
{
    const auto& abbreviations = artifact.segmentation_abbreviations();
    if (abbreviations.empty()) {
        return segmentation::NorwegianPolicy(maximum_token_count);
    }
    return segmentation::SegmentationPolicy{
        std::unordered_set<std::string>(abbreviations.begin(), abbreviations.end()),
        maximum_token_count,
    };
}

} // namespace

struct Tagger::Implementation {
    explicit Implementation(const std::filesystem::path& directory)
        : artifact(directory)
        , tokenizer(directory / artifact.tokenizer().file_name)
        , policy(PolicyFromArtifact(
              artifact,
              static_cast<std::size_t>(artifact.programs().back().shapes.token_count)))
    {
        std::int64_t identifier = kFirstLiteralCharacterId;
        for (const auto& character : artifact.labels().character_vocabulary) {
            character_ids.emplace(character, identifier++);
        }
        // The runtime's own default parallelizes over every logical core,
        // which measurably oversubscribes the small fixed-shape batches
        // (24% slower on a 16-core machine). Six threads is the measured
        // sweet spot; callers override via engine::SetThreadCount.
        engine::SetDefaultThreadCount(kDefaultThreadCount);
    }

    static constexpr std::size_t kDefaultThreadCount = 6;

    engine::Program& ProgramFor(const artifact::Program& program)
    {
        auto found = programs.find(program.file_name);
        if (found == programs.end()) {
            std::vector<std::filesystem::path> data_files;
            data_files.reserve(program.data_files.size());
            for (const auto& data_file : program.data_files) {
                data_files.push_back(artifact.directory() / data_file);
            }
            found = programs
                        .emplace(program.file_name,
                            std::make_unique<engine::Program>(
                                artifact.directory() / program.file_name,
                                data_files))
                        .first;
        }
        return *found->second;
    }

    const artifact::Program& SmallestFittingProgram(
        std::size_t maximum_subwords, std::size_t maximum_tokens) const
    {
        for (const auto& program : artifact.programs()) {
            if (static_cast<std::size_t>(program.shapes.subword_count) >= maximum_subwords
                && static_cast<std::size_t>(program.shapes.token_count) >= maximum_tokens) {
                return program;
            }
        }
        return artifact.programs().back();
    }

    std::vector<std::int64_t> EncodeCharacters(const std::string& token) const
    {
        const auto maximum = static_cast<std::size_t>(
            artifact.labels().maximum_character_count);
        std::vector<std::int64_t> literal_ids;
        for (const auto& codepoint : SplitCodepoints(token)) {
            const auto found = character_ids.find(codepoint);
            literal_ids.push_back(
                found == character_ids.end() ? kCharacterUnknownId : found->second);
        }

        std::vector<std::int64_t> complete;
        if (literal_ids.size() + 2 <= maximum) {
            complete.push_back(kCharacterStartId);
            complete.insert(complete.end(), literal_ids.begin(), literal_ids.end());
            complete.push_back(kCharacterEndId);
            return complete;
        }

        // Middle truncation keeps prefix and suffix; the model was trained
        // with this exact policy.
        const auto retained = maximum - 3;
        const auto prefix_count = (retained + 1) / 2;
        const auto suffix_count = retained - prefix_count;
        complete.push_back(kCharacterStartId);
        complete.insert(complete.end(), literal_ids.begin(),
            literal_ids.begin() + static_cast<std::ptrdiff_t>(prefix_count));
        complete.push_back(kCharacterTruncationId);
        complete.insert(complete.end(),
            literal_ids.end() - static_cast<std::ptrdiff_t>(suffix_count),
            literal_ids.end());
        complete.push_back(kCharacterEndId);
        return complete;
    }

    std::vector<TaggedSentence> TagBatch(
        const std::vector<segmentation::PretokenizedSentence>& real_sentences,
        const std::vector<subword::EncodedSentence>& real_encoded,
        const artifact::Program& program_entry)
    {
        const auto& shapes = program_entry.shapes;
        const auto batch_size = static_cast<std::size_t>(shapes.batch_size);
        const auto subword_count = static_cast<std::size_t>(shapes.subword_count);
        const auto token_count = static_cast<std::size_t>(shapes.token_count);
        const auto character_count = static_cast<std::size_t>(
            shapes.character_count.value_or(artifact.labels().maximum_character_count));

        // Partial batches repeat the last sentence up to the fixed batch
        // size; the repeated rows are decoded but never returned.
        auto sentences = real_sentences;
        auto encoded = real_encoded;
        while (sentences.size() < batch_size) {
            sentences.push_back(sentences.back());
            encoded.push_back(encoded.back());
        }

        engine::InputTensor input_ids;
        input_ids.kind = engine::InputTensor::Kind::int64;
        input_ids.shape.sizes = {shapes.batch_size, shapes.subword_count};
        input_ids.int64_data.assign(
            batch_size * subword_count, artifact.tokenizer().padding_token_id);

        engine::InputTensor attention_mask;
        attention_mask.kind = engine::InputTensor::Kind::boolean;
        attention_mask.shape.sizes = {shapes.batch_size, shapes.subword_count};
        attention_mask.boolean_data.assign(batch_size * subword_count, 0);

        engine::InputTensor first_indices;
        first_indices.kind = engine::InputTensor::Kind::int64;
        first_indices.shape.sizes = {shapes.batch_size, shapes.token_count};
        first_indices.int64_data.assign(batch_size * token_count, 0);

        engine::InputTensor end_indices;
        end_indices.kind = engine::InputTensor::Kind::int64;
        end_indices.shape.sizes = {shapes.batch_size, shapes.token_count};
        end_indices.int64_data.assign(batch_size * token_count, 0);

        engine::InputTensor token_mask;
        token_mask.kind = engine::InputTensor::Kind::boolean;
        token_mask.shape.sizes = {shapes.batch_size, shapes.token_count};
        token_mask.boolean_data.assign(batch_size * token_count, 0);

        engine::InputTensor character_ids_tensor;
        character_ids_tensor.kind = engine::InputTensor::Kind::int64;
        character_ids_tensor.shape.sizes = {
            shapes.batch_size, shapes.token_count, static_cast<int>(character_count)};
        character_ids_tensor.int64_data.assign(
            batch_size * token_count * character_count, kCharacterPaddingId);

        engine::InputTensor character_mask;
        character_mask.kind = engine::InputTensor::Kind::boolean;
        character_mask.shape.sizes = {
            shapes.batch_size, shapes.token_count, static_cast<int>(character_count)};
        character_mask.boolean_data.assign(batch_size * token_count * character_count, 0);

        for (std::size_t row = 0; row < batch_size; ++row) {
            const auto& sentence = encoded[row];
            const auto subword_base = row * subword_count;
            for (std::size_t offset = 0; offset < sentence.input_ids.size(); ++offset) {
                input_ids.int64_data[subword_base + offset] = sentence.input_ids[offset];
                attention_mask.boolean_data[subword_base + offset] = 1;
            }
            const auto token_base = row * token_count;
            for (std::size_t token = 0; token < sentence.first_subword_indices.size();
                ++token) {
                first_indices.int64_data[token_base + token]
                    = sentence.first_subword_indices[token];
                end_indices.int64_data[token_base + token]
                    = sentence.subword_end_indices[token];
                token_mask.boolean_data[token_base + token] = 1;
                const auto character_base = (token_base + token) * character_count;
                const auto ids = EncodeCharacters(sentences[row].tokens[token]);
                for (std::size_t offset = 0; offset < ids.size(); ++offset) {
                    character_ids_tensor.int64_data[character_base + offset] = ids[offset];
                    character_mask.boolean_data[character_base + offset] = 1;
                }
            }
        }

        std::vector<engine::InputTensor> inputs;
        inputs.push_back(std::move(input_ids));
        inputs.push_back(std::move(attention_mask));
        inputs.push_back(std::move(first_indices));
        inputs.push_back(std::move(end_indices));
        inputs.push_back(std::move(token_mask));
        if (shapes.character_count.has_value()) {
            inputs.push_back(std::move(character_ids_tensor));
            inputs.push_back(std::move(character_mask));
        }

        const auto outputs = ProgramFor(program_entry).Forward(inputs);
        return Decode(real_sentences, outputs, token_count);
    }

    std::vector<TaggedSentence> Decode(
        const std::vector<segmentation::PretokenizedSentence>& sentences,
        const std::vector<engine::OutputTensor>& outputs, std::size_t token_count) const
    {
        const auto& labels = artifact.labels();
        const auto upos_count = labels.upos_labels.size();
        const auto rule_count = labels.lemma_rules.size();

        std::vector<TaggedSentence> tagged;
        tagged.reserve(sentences.size());
        for (std::size_t row = 0; row < sentences.size(); ++row) {
            const auto& sentence = sentences[row];
            TaggedSentence result;
            result.tokens.reserve(sentence.tokens.size());
            result.source_ranges = sentence.source_ranges;
            for (std::size_t token = 0; token < sentence.tokens.size(); ++token) {
                const auto flat = row * token_count + token;
                TaggedToken tagged_token;
                tagged_token.text = sentence.tokens[token];
                tagged_token.has_space_before = sentence.has_space_before[token];
                if (!sentence.token_source_ranges.empty()) {
                    tagged_token.source_ranges = sentence.token_source_ranges[token];
                }

                const auto upos = ArgmaxIn(outputs[0].data, flat * upos_count, upos_count);
                tagged_token.upos = labels.upos_labels[upos.index];
                tagged_token.upos_confidence = upos.value;
                tagged_token.upos_distribution.reserve(upos_count);
                for (std::size_t label = 0; label < upos_count; ++label) {
                    tagged_token.upos_distribution.push_back({labels.upos_labels[label],
                        static_cast<double>(outputs[0].data[flat * upos_count + label])});
                }
                std::sort(tagged_token.upos_distribution.begin(),
                    tagged_token.upos_distribution.end(),
                    [](const UposProbability& left, const UposProbability& right) {
                        return left.probability > right.probability;
                    });

                for (std::size_t feature_index = 0; feature_index < labels.features.size();
                    ++feature_index) {
                    const auto& feature = labels.features[feature_index];
                    const auto& output = outputs[1 + feature_index].data;
                    if (feature.allows_multiple_values) {
                        // Independent per-value decisions with a 0.5
                        // threshold; the reported confidence is the least
                        // certain selected value.
                        const auto count = feature.values.size();
                        std::vector<std::string> selected;
                        auto confidence = std::numeric_limits<float>::max();
                        for (std::size_t value = 0; value < count; ++value) {
                            const auto probability = output[flat * count + value];
                            if (probability > 0.5F) {
                                selected.push_back(feature.values[value]);
                                confidence = std::min(confidence, probability);
                            }
                        }
                        if (!selected.empty()) {
                            tagged_token.features[feature.name] = std::move(selected);
                            tagged_token.feature_confidences[feature.name] = confidence;
                        }
                    } else {
                        // Exclusive features carry an implicit leading
                        // "not present" class, so real values shift by one.
                        const auto count = feature.values.size() + 1;
                        const auto best = ArgmaxIn(output, flat * count, count);
                        if (best.index > 0) {
                            tagged_token.features[feature.name]
                                = {feature.values[best.index - 1]};
                            tagged_token.feature_confidences[feature.name] = best.value;
                        }
                    }
                }

                const auto rule = ArgmaxIn(
                    outputs.back().data, flat * rule_count, rule_count);
                const auto lemma
                    = labels.lemma_rules[rule.index].Apply(tagged_token.text);
                tagged_token.lemma = lemma.value_or(tagged_token.text);
                tagged_token.lemma_confidence = rule.value;

                result.tokens.push_back(std::move(tagged_token));
            }
            tagged.push_back(std::move(result));
        }
        return tagged;
    }

    artifact::Artifact artifact;
    subword::Tokenizer tokenizer;
    segmentation::SegmentationPolicy policy;
    std::unordered_map<std::string, std::int64_t> character_ids;
    std::unordered_map<std::string, std::unique_ptr<engine::Program>> programs;
};

Tagger::Tagger(const std::filesystem::path& artifact_directory)
    : implementation_(std::make_unique<Implementation>(artifact_directory))
{
}

Tagger::~Tagger() = default;

const artifact::Artifact& Tagger::artifact() const
{
    return implementation_->artifact;
}

std::vector<TaggedSentence> Tagger::TagText(std::string_view text)
{
    return Tag(segmentation::Segment(text, implementation_->policy));
}

std::vector<TaggedSentence> Tagger::TagPretokenized(
    const std::vector<std::vector<std::string>>& sentences)
{
    std::vector<segmentation::PretokenizedSentence> prepared;
    prepared.reserve(sentences.size());
    for (const auto& tokens : sentences) {
        if (tokens.empty()) {
            continue;
        }
        segmentation::PretokenizedSentence sentence;
        sentence.tokens = tokens;
        sentence.has_space_before.assign(tokens.size(), true);
        sentence.has_space_before[0] = false;
        prepared.push_back(std::move(sentence));
    }
    return Tag(prepared);
}

std::vector<TaggedSentence> Tagger::Tag(
    const std::vector<segmentation::PretokenizedSentence>& sentences)
{
    auto& state = *implementation_;
    const auto& largest = state.artifact.programs().back().shapes;

    std::vector<segmentation::PretokenizedSentence> prepared;
    for (const auto& sentence : sentences) {
        ValidateSourceMapping(sentence);
        for (auto& chunk : segmentation::Chunk(
                 sentence, static_cast<std::size_t>(largest.token_count))) {
            prepared.push_back(std::move(chunk));
        }
    }

    std::vector<subword::EncodedSentence> encoded;
    encoded.reserve(prepared.size());
    for (const auto& sentence : prepared) {
        encoded.push_back(state.tokenizer.Encode(sentence));
        if (encoded.back().input_ids.size()
            > static_cast<std::size_t>(largest.subword_count)) {
            throw std::runtime_error(
                "Sentence exceeds the largest program's subword capacity.");
        }
    }

    // Length-sorted batching keeps short sentences together so their
    // batches qualify for the smallest lowered program.
    std::vector<std::size_t> order(prepared.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&encoded](std::size_t a, std::size_t b) {
        return encoded[a].input_ids.size() < encoded[b].input_ids.size();
    });

    std::vector<TaggedSentence> tagged(prepared.size());
    const auto batch_size = static_cast<std::size_t>(largest.batch_size);
    for (std::size_t start = 0; start < order.size(); start += batch_size) {
        const auto end = std::min(start + batch_size, order.size());
        std::vector<segmentation::PretokenizedSentence> batch_sentences;
        std::vector<subword::EncodedSentence> batch_encoded;
        std::size_t maximum_subwords = 0;
        std::size_t maximum_tokens = 0;
        for (std::size_t position = start; position < end; ++position) {
            const auto index = order[position];
            batch_sentences.push_back(prepared[index]);
            batch_encoded.push_back(encoded[index]);
            maximum_subwords = std::max(maximum_subwords, encoded[index].input_ids.size());
            maximum_tokens = std::max(maximum_tokens, prepared[index].tokens.size());
        }
        const auto& program
            = state.SmallestFittingProgram(maximum_subwords, maximum_tokens);
        auto decoded = state.TagBatch(batch_sentences, batch_encoded, program);
        for (std::size_t position = start; position < end; ++position) {
            tagged[order[position]] = std::move(decoded[position - start]);
        }
    }
    return tagged;
}

} // namespace prism::tagger
