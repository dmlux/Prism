import Foundation
import PrismKit

// Usage: swift run quickstart <artifact-directory>
// In an app, ship the artifact folder as a bundle resource instead and
// resolve it via Bundle.main.resourceURL.
guard CommandLine.arguments.count == 2 else {
    print("usage: quickstart <artifact-directory>")
    exit(2)
}
let artifactDirectory = URL(fileURLWithPath: CommandLine.arguments[1])

let tagger = try PrismTagger(artifactURL: artifactDirectory, device: .cpu)
print("Loaded \(tagger.artifactName) \(tagger.artifactVersion) [\(tagger.languageTags.joined(separator: ", "))]")

for sentence in try tagger.tag(text: "Hun kjøpte tre gamle bøker.") {
    for token in sentence.tokens {
        print("\(token.text)\t\(token.upos)\t\(token.lemma)\t\(String(format: "%.3f", token.uposConfidence))")
    }
}
