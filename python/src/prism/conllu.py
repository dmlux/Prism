from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    upos: str
    features: dict[str, str]
    space_after: bool = True


def parse_token(line: str) -> Token:
    columns = line.split("\t")

    return Token(
        text=columns[1],
        lemma=columns[2],
        upos=columns[3],
        features=(
            {}
            if columns[5] == "_"
            else dict(feature.split("=", 1) for feature in columns[5].split("|"))
        ),
        space_after="SpaceAfter=No" not in columns[9].split("|"),
    )


def read_first_sentence(path: Path) -> list[Token]:
    tokens: list[Token] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            break

        if line.startswith("#"):
            continue

        tokens.append(parse_token(line))

    return tokens


def read_sentences(path: Path) -> list[list[Token]]:
    sentences: list[list[Token]] = []
    current_sentence: list[Token] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue

        if line.startswith("#"):
            continue

        current_sentence.append(parse_token(line))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences
