#include "prism/prism_c.h"

#include <exception>
#include <string>
#include <vector>

#include "prism/tagger.h"

namespace {

thread_local std::string g_last_error;

// One decoded feature with its display forms precomputed, so accessors can
// hand out stable const char* pointers for the result's lifetime.
struct FeatureEntry {
    std::string name;
    std::string value;
    double confidence = 0.0;
};

struct TokenEntry {
    prism::tagger::TaggedToken token;
    std::vector<FeatureEntry> features;
    std::string features_string;
};

std::string JoinValues(const std::vector<std::string>& values)
{
    std::string joined;
    for (const auto& value : values) {
        if (!joined.empty()) {
            joined += ',';
        }
        joined += value;
    }
    return joined;
}

} // namespace

struct prism_tagger {
    explicit prism_tagger(const char* directory)
        : tagger(directory)
    {
    }

    prism::tagger::Tagger tagger;
};

struct prism_result {
    explicit prism_result(std::vector<prism::tagger::TaggedSentence> tagged)
    {
        sentences.reserve(tagged.size());
        for (auto& sentence : tagged) {
            std::vector<TokenEntry> tokens;
            tokens.reserve(sentence.tokens.size());
            for (auto& token : sentence.tokens) {
                TokenEntry entry;
                // TaggedToken stores features in a std::map, so iteration
                // (and thus the feature indices and the CoNLL-U string)
                // is alphabetical by feature name.
                for (const auto& [name, values] : token.features) {
                    FeatureEntry feature;
                    feature.name = name;
                    feature.value = JoinValues(values);
                    feature.confidence = token.feature_confidences.at(name);
                    if (!entry.features_string.empty()) {
                        entry.features_string += '|';
                    }
                    entry.features_string += name + "=" + feature.value;
                    entry.features.push_back(std::move(feature));
                }
                entry.token = std::move(token);
                tokens.push_back(std::move(entry));
            }
            sentences.push_back(std::move(tokens));
        }
    }

    const TokenEntry* Token(size_t sentence, size_t token) const
    {
        if (sentence >= sentences.size() || token >= sentences[sentence].size()) {
            return nullptr;
        }
        return &sentences[sentence][token];
    }

    std::vector<std::vector<TokenEntry>> sentences;
};

extern "C" {

prism_tagger* prism_tagger_create(const char* artifact_directory)
{
    if (artifact_directory == nullptr) {
        g_last_error = "artifact_directory is NULL.";
        return nullptr;
    }
    try {
        return new prism_tagger(artifact_directory);
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return nullptr;
    }
}

void prism_tagger_destroy(prism_tagger* tagger)
{
    delete tagger;
}

const char* prism_last_error(void)
{
    return g_last_error.c_str();
}

prism_result* prism_tagger_tag_text(prism_tagger* tagger, const char* utf8_text)
{
    if (tagger == nullptr || utf8_text == nullptr) {
        g_last_error = "tagger or utf8_text is NULL.";
        return nullptr;
    }
    try {
        return new prism_result(tagger->tagger.TagText(utf8_text));
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return nullptr;
    }
}

prism_result* prism_tagger_tag_tokens(
    prism_tagger* tagger, const char* const* tokens, size_t token_count)
{
    if (tagger == nullptr || (tokens == nullptr && token_count > 0)) {
        g_last_error = "tagger or tokens is NULL.";
        return nullptr;
    }
    try {
        std::vector<std::string> sentence;
        sentence.reserve(token_count);
        for (size_t index = 0; index < token_count; ++index) {
            if (tokens[index] == nullptr) {
                g_last_error = "tokens contains a NULL entry.";
                return nullptr;
            }
            sentence.emplace_back(tokens[index]);
        }
        return new prism_result(tagger->tagger.TagPretokenized({sentence}));
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return nullptr;
    }
}

void prism_result_destroy(prism_result* result)
{
    delete result;
}

size_t prism_result_sentence_count(const prism_result* result)
{
    return result == nullptr ? 0 : result->sentences.size();
}

size_t prism_result_token_count(const prism_result* result, size_t sentence)
{
    if (result == nullptr || sentence >= result->sentences.size()) {
        return 0;
    }
    return result->sentences[sentence].size();
}

const char* prism_result_token_text(
    const prism_result* result, size_t sentence, size_t token)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    return entry == nullptr ? nullptr : entry->token.text.c_str();
}

int prism_result_token_has_space_before(
    const prism_result* result, size_t sentence, size_t token)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    return entry != nullptr && entry->token.has_space_before ? 1 : 0;
}

const char* prism_result_token_upos(
    const prism_result* result, size_t sentence, size_t token)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    return entry == nullptr ? nullptr : entry->token.upos.c_str();
}

double prism_result_token_upos_confidence(
    const prism_result* result, size_t sentence, size_t token)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    return entry == nullptr ? 0.0 : entry->token.upos_confidence;
}

size_t prism_result_token_feature_count(
    const prism_result* result, size_t sentence, size_t token)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    return entry == nullptr ? 0 : entry->features.size();
}

const char* prism_result_token_feature_name(
    const prism_result* result, size_t sentence, size_t token, size_t feature)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    if (entry == nullptr || feature >= entry->features.size()) {
        return nullptr;
    }
    return entry->features[feature].name.c_str();
}

const char* prism_result_token_feature_value(
    const prism_result* result, size_t sentence, size_t token, size_t feature)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    if (entry == nullptr || feature >= entry->features.size()) {
        return nullptr;
    }
    return entry->features[feature].value.c_str();
}

double prism_result_token_feature_confidence(
    const prism_result* result, size_t sentence, size_t token, size_t feature)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    if (entry == nullptr || feature >= entry->features.size()) {
        return 0.0;
    }
    return entry->features[feature].confidence;
}

const char* prism_result_token_features(
    const prism_result* result, size_t sentence, size_t token)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    return entry == nullptr ? nullptr : entry->features_string.c_str();
}

const char* prism_result_token_lemma(
    const prism_result* result, size_t sentence, size_t token)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    return entry == nullptr ? nullptr : entry->token.lemma.c_str();
}

double prism_result_token_lemma_confidence(
    const prism_result* result, size_t sentence, size_t token)
{
    const auto* entry = result == nullptr ? nullptr : result->Token(sentence, token);
    return entry == nullptr ? 0.0 : entry->token.lemma_confidence;
}

} // extern "C"
