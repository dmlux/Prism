from prism.conllu import Token, parse_token


def test_parse_token() -> None:
    line = (
        "1\tLam\tlam\tNOUN\tsubst\tDefinite=Ind|Gender=Neut|Number=Sing\t0\troot\t_\t_"
    )

    assert parse_token(line) == Token(
        text="Lam",
        lemma="lam",
        upos="NOUN",
        features={
            "Definite": "Ind",
            "Gender": "Neut",
            "Number": "Sing",
        },
    )
