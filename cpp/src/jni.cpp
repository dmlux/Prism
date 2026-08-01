// JNI bridge for the Java binding (java/, package dev.prism).
//
// The Java layer sends text as UTF-8 byte arrays and receives one flat
// parallel-array payload per call, so a tagging request costs a single
// JNI transition regardless of sentence count. Strings travel back
// through NewString (UTF-16) rather than NewStringUTF, because the JNI
// "modified UTF-8" encoding breaks for supplementary codepoints.

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include <jni.h>

#include "prism/tagger.h"

namespace {

std::u16string Utf8ToUtf16(const std::string& text)
{
    std::u16string result;
    result.reserve(text.size());
    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        char32_t codepoint = 0xFFFD;
        std::size_t length = 1;
        if (lead < 0x80U) {
            codepoint = lead;
        } else if ((lead & 0xE0U) == 0xC0U) {
            codepoint = lead & 0x1FU;
            length = 2;
        } else if ((lead & 0xF0U) == 0xE0U) {
            codepoint = lead & 0x0FU;
            length = 3;
        } else if ((lead & 0xF8U) == 0xF0U) {
            codepoint = lead & 0x07U;
            length = 4;
        }
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto byte = index + offset < text.size()
                ? static_cast<unsigned char>(text[index + offset])
                : 0U;
            if ((byte & 0xC0U) != 0x80U) {
                codepoint = 0xFFFD;
                length = 1;
                break;
            }
            codepoint = (codepoint << 6U) | (byte & 0x3FU);
        }
        if (codepoint >= 0x10000U) {
            codepoint -= 0x10000U;
            result.push_back(static_cast<char16_t>(0xD800U + (codepoint >> 10U)));
            result.push_back(static_cast<char16_t>(0xDC00U + (codepoint & 0x3FFU)));
        } else {
            result.push_back(static_cast<char16_t>(codepoint));
        }
        index += length;
    }
    return result;
}

jstring MakeString(JNIEnv* env, const std::string& utf8)
{
    const auto utf16 = Utf8ToUtf16(utf8);
    return env->NewString(
        reinterpret_cast<const jchar*>(utf16.data()), static_cast<jsize>(utf16.size()));
}

std::string TakeUtf8(JNIEnv* env, jbyteArray bytes)
{
    const auto length = env->GetArrayLength(bytes);
    std::string result(static_cast<std::size_t>(length), '\0');
    env->GetByteArrayRegion(bytes, 0, length, reinterpret_cast<jbyte*>(result.data()));
    return result;
}

void ThrowPrismException(JNIEnv* env, const std::string& message)
{
    jclass exception_class = env->FindClass("dev/prism/PrismException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message.c_str());
    }
}

// The result payload: eleven parallel arrays, unflattened on the Java side.
jobjectArray MarshalResult(
    JNIEnv* env, const std::vector<prism::tagger::TaggedSentence>& sentences)
{
    std::size_t token_total = 0;
    std::size_t feature_total = 0;
    for (const auto& sentence : sentences) {
        token_total += sentence.tokens.size();
        for (const auto& token : sentence.tokens) {
            feature_total += token.features.size();
        }
    }

    std::vector<jint> tokens_per_sentence;
    std::vector<jboolean> has_space_before;
    std::vector<jdouble> upos_confidences;
    std::vector<jint> feature_counts;
    std::vector<jdouble> feature_confidences;
    std::vector<jdouble> lemma_confidences;
    tokens_per_sentence.reserve(sentences.size());
    has_space_before.reserve(token_total);
    upos_confidences.reserve(token_total);
    feature_counts.reserve(token_total);
    feature_confidences.reserve(feature_total);
    lemma_confidences.reserve(token_total);

    jclass string_class = env->FindClass("java/lang/String");
    jobjectArray texts
        = env->NewObjectArray(static_cast<jsize>(token_total), string_class, nullptr);
    jobjectArray upos
        = env->NewObjectArray(static_cast<jsize>(token_total), string_class, nullptr);
    jobjectArray lemmas
        = env->NewObjectArray(static_cast<jsize>(token_total), string_class, nullptr);
    jobjectArray feature_names
        = env->NewObjectArray(static_cast<jsize>(feature_total), string_class, nullptr);
    jobjectArray feature_values
        = env->NewObjectArray(static_cast<jsize>(feature_total), string_class, nullptr);

    jsize token_index = 0;
    jsize feature_index = 0;
    for (const auto& sentence : sentences) {
        tokens_per_sentence.push_back(static_cast<jint>(sentence.tokens.size()));
        for (const auto& token : sentence.tokens) {
            env->SetObjectArrayElement(texts, token_index, MakeString(env, token.text));
            env->SetObjectArrayElement(upos, token_index, MakeString(env, token.upos));
            env->SetObjectArrayElement(lemmas, token_index, MakeString(env, token.lemma));
            has_space_before.push_back(token.has_space_before ? JNI_TRUE : JNI_FALSE);
            upos_confidences.push_back(token.upos_confidence);
            lemma_confidences.push_back(token.lemma_confidence);
            feature_counts.push_back(static_cast<jint>(token.features.size()));
            for (const auto& [name, values] : token.features) {
                std::string joined;
                for (const auto& value : values) {
                    if (!joined.empty()) {
                        joined += ',';
                    }
                    joined += value;
                }
                env->SetObjectArrayElement(
                    feature_names, feature_index, MakeString(env, name));
                env->SetObjectArrayElement(
                    feature_values, feature_index, MakeString(env, joined));
                feature_confidences.push_back(token.feature_confidences.at(name));
                ++feature_index;
            }
            ++token_index;
        }
    }

    jintArray tokens_per_sentence_array
        = env->NewIntArray(static_cast<jsize>(tokens_per_sentence.size()));
    env->SetIntArrayRegion(tokens_per_sentence_array, 0,
        static_cast<jsize>(tokens_per_sentence.size()), tokens_per_sentence.data());
    jbooleanArray has_space_before_array
        = env->NewBooleanArray(static_cast<jsize>(token_total));
    env->SetBooleanArrayRegion(
        has_space_before_array, 0, static_cast<jsize>(token_total), has_space_before.data());
    jdoubleArray upos_confidence_array
        = env->NewDoubleArray(static_cast<jsize>(token_total));
    env->SetDoubleArrayRegion(
        upos_confidence_array, 0, static_cast<jsize>(token_total), upos_confidences.data());
    jintArray feature_count_array = env->NewIntArray(static_cast<jsize>(token_total));
    env->SetIntArrayRegion(
        feature_count_array, 0, static_cast<jsize>(token_total), feature_counts.data());
    jdoubleArray feature_confidence_array
        = env->NewDoubleArray(static_cast<jsize>(feature_total));
    env->SetDoubleArrayRegion(feature_confidence_array, 0,
        static_cast<jsize>(feature_total), feature_confidences.data());
    jdoubleArray lemma_confidence_array
        = env->NewDoubleArray(static_cast<jsize>(token_total));
    env->SetDoubleArrayRegion(
        lemma_confidence_array, 0, static_cast<jsize>(token_total), lemma_confidences.data());

    jobjectArray payload
        = env->NewObjectArray(11, env->FindClass("java/lang/Object"), nullptr);
    env->SetObjectArrayElement(payload, 0, tokens_per_sentence_array);
    env->SetObjectArrayElement(payload, 1, texts);
    env->SetObjectArrayElement(payload, 2, has_space_before_array);
    env->SetObjectArrayElement(payload, 3, upos);
    env->SetObjectArrayElement(payload, 4, upos_confidence_array);
    env->SetObjectArrayElement(payload, 5, feature_count_array);
    env->SetObjectArrayElement(payload, 6, feature_names);
    env->SetObjectArrayElement(payload, 7, feature_values);
    env->SetObjectArrayElement(payload, 8, feature_confidence_array);
    env->SetObjectArrayElement(payload, 9, lemmas);
    env->SetObjectArrayElement(payload, 10, lemma_confidence_array);
    return payload;
}

prism::tagger::Tagger* TaggerFrom(jlong handle)
{
    return reinterpret_cast<prism::tagger::Tagger*>(handle);
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_dev_prism_PrismTagger_nativeCreate(
    JNIEnv* env, jclass, jbyteArray utf8_directory)
{
    try {
        return reinterpret_cast<jlong>(
            new prism::tagger::Tagger(TakeUtf8(env, utf8_directory)));
    } catch (const std::exception& error) {
        ThrowPrismException(env, error.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_dev_prism_PrismTagger_nativeDestroy(
    JNIEnv*, jclass, jlong handle)
{
    delete TaggerFrom(handle);
}

JNIEXPORT jobjectArray JNICALL Java_dev_prism_PrismTagger_nativeTagText(
    JNIEnv* env, jclass, jlong handle, jbyteArray utf8_text)
{
    try {
        return MarshalResult(env, TaggerFrom(handle)->TagText(TakeUtf8(env, utf8_text)));
    } catch (const std::exception& error) {
        ThrowPrismException(env, error.what());
        return nullptr;
    }
}

JNIEXPORT jobjectArray JNICALL Java_dev_prism_PrismTagger_nativeTagTokens(
    JNIEnv* env, jclass, jlong handle, jobjectArray utf8_tokens,
    jintArray tokens_per_sentence)
{
    try {
        const auto sentence_count = env->GetArrayLength(tokens_per_sentence);
        std::vector<jint> counts(static_cast<std::size_t>(sentence_count));
        env->GetIntArrayRegion(tokens_per_sentence, 0, sentence_count, counts.data());

        std::vector<std::vector<std::string>> sentences;
        sentences.reserve(counts.size());
        jsize token_index = 0;
        for (const auto count : counts) {
            std::vector<std::string> tokens;
            tokens.reserve(static_cast<std::size_t>(count));
            for (jint token = 0; token < count; ++token, ++token_index) {
                auto element = static_cast<jbyteArray>(
                    env->GetObjectArrayElement(utf8_tokens, token_index));
                tokens.push_back(TakeUtf8(env, element));
                env->DeleteLocalRef(element);
            }
            sentences.push_back(std::move(tokens));
        }
        return MarshalResult(env, TaggerFrom(handle)->TagPretokenized(sentences));
    } catch (const std::exception& error) {
        ThrowPrismException(env, error.what());
        return nullptr;
    }
}

} // extern "C"
