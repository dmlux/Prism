from collections import Counter, defaultdict

from vexo.conllu import Token


def train_pos_baseline(
    sentences: list[list[Token]],
) -> dict[str, str]:
    tag_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for sentence in sentences:
        for token in sentence:
            tag_counts[token.text][token.upos] += 1

    return {
        text: counts.most_common(1)[0][0]
        for text, counts in tag_counts.items()
    }