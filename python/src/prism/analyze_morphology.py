from collections import Counter, defaultdict
from pathlib import Path

from prism.conllu import read_sentences

def main() -> None:
    path = Path(
        "data/raw/UD_Norwegian-Bokmaal/"
        "no_bokmaal-ud-train.conllu"
    )
    sentences = read_sentences(path)

    feature_counts: dict[str, Counter[str]] = defaultdict(
        Counter
    )

    for sentence in sentences:
        for token in sentence:
            for feature, value in token.features.items():
                feature_counts[feature][value] += 1

    for feature in sorted(feature_counts):
        values = feature_counts[feature]
        total = sum(values.values())

        print(f"{feature} ({total} Tokens)")

        for value, count in values.most_common():
            print(f"   {value}: {count}")

        print()

if __name__ == "__main__":
    main()