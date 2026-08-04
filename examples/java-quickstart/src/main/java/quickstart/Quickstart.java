package quickstart;

import io.github.dmlux.prism.PrismTagger;
import java.nio.file.Path;

/** Usage: Quickstart &lt;artifact-directory&gt; */
public final class Quickstart {

    public static void main(String[] arguments) {
        if (arguments.length != 1) {
            System.err.println("usage: quickstart <artifact-directory>");
            System.exit(2);
        }

        try (PrismTagger tagger = PrismTagger.load(Path.of(arguments[0]))) {
            System.out.println("Loaded " + tagger.artifactName() + " "
                + tagger.artifactVersion() + " " + tagger.languageTags());

            for (var sentence : tagger.tagText("Hun kjøpte tre gamle bøker.")) {
                for (var token : sentence.tokens()) {
                    System.out.printf("%s\t%s\t%s\t%.3f%n",
                        token.text(), token.upos(), token.lemma(),
                        token.uposConfidence());
                }
            }
        }
    }

    private Quickstart() {}
}
