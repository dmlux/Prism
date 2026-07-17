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


def test_parse_token_preserves_space_after() -> None:
    line = (
        "1\tKamskjell\tkamskjell\tNOUN\tsubst\t"
        "Definite=Ind|Gender=Neut|Number=Plur\t0\troot\t_\tSpaceAfter=No"
    )

    token = parse_token(line)

    assert token.space_after is False
