import XCTest

import PrismKit
import PrismNative

/// Coexistence and result-parity proof: PrismKit (Swift runtime on the
/// prebuilt ExecuTorch products) and PrismNative (binary C ABI with its own
/// bundled, symbol-hidden runtime) run in the same process against the same
/// artifact, and their linguistic results — token texts, UPOS, lemmas, and
/// UTF-8 source ranges — must be identical. Linking both also proves that
/// PrismNative cannot collide with PrismKit's ExecuTorch products.
///
/// Provide the artifact directory via the PRISM_ARTIFACT environment
/// variable; the test skips when it is absent.
final class CoexistenceTests: XCTestCase {
    func testSameProcessSameResults() throws {
        guard let artifact = ProcessInfo.processInfo.environment["PRISM_ARTIFACT"] else {
            throw XCTSkip("Set PRISM_ARTIFACT to an artifact directory.")
        }

        let kit = try PrismTagger(artifactURL: URL(fileURLWithPath: artifact), device: .cpu)
        let native = prism_tagger_create(artifact)
        guard let native else {
            XCTFail("PrismNative: \(String(cString: prism_last_error()))")
            return
        }
        defer { prism_tagger_destroy(native) }

        XCTAssertEqual(kit.artifactName, String(cString: prism_tagger_artifact_name(native)))
        XCTAssertEqual(kit.languageTags.count, prism_tagger_language_tag_count(native))

        let texts = [
            "Hun kjøpte tre gamle bøker den 17. mai.",
            "Dette er språk-\nmodellen til laget.",
        ]
        for text in texts {
            let kitSentences = try kit.tag(text: text)
            guard let result = prism_tagger_tag_text(native, text) else {
                XCTFail("PrismNative: \(String(cString: prism_last_error()))")
                continue
            }
            defer { prism_result_destroy(result) }

            XCTAssertEqual(kitSentences.count, prism_result_sentence_count(result))
            for (s, sentence) in kitSentences.enumerated() {
                XCTAssertEqual(sentence.tokens.count, prism_result_token_count(result, s))
                for (t, token) in sentence.tokens.enumerated() {
                    XCTAssertEqual(
                        token.text, String(cString: prism_result_token_text(result, s, t))
                    )
                    XCTAssertEqual(
                        token.upos, String(cString: prism_result_token_upos(result, s, t))
                    )
                    XCTAssertEqual(
                        token.lemma, String(cString: prism_result_token_lemma(result, s, t))
                    )
                    XCTAssertEqual(
                        token.sourceRanges.count,
                        prism_result_token_source_range_count(result, s, t)
                    )
                    for (r, range) in token.sourceRanges.enumerated() {
                        XCTAssertEqual(
                            range.start,
                            prism_result_token_source_range_start(result, s, t, r)
                        )
                        XCTAssertEqual(
                            range.end,
                            prism_result_token_source_range_end(result, s, t, r)
                        )
                    }
                }
            }
        }
    }
}
