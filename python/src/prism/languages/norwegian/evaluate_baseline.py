from dataclasses import asdict
import json
from pathlib import Path

import torch

from prism.conllu import read_sentences
from prism.data import (
    build_norwegian_bokmaal_schema,
    encode_norwegian_bokmaal_sentences,
)
from prism.evaluation.classification import (
    calculate_classification_metrics,
)
from prism.languages.norwegian import (
    NORWEGIAN_BOKMAAL_PROFILE,
)
from prism.modeling import (
    build_pretrained_token_tagger,
    load_backbone_tokenizer,
)
from prism.schema.serialization import (
    serialize_token_task_schema,
)
from prism.training import (
    evaluate_supervised_token_task_epoch,
    iter_supervised_token_task_batches,
)


def main() -> None:
    checkpoint_path = Path("runs/nb-student-baseline/best.pt")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    data_root = Path("data/raw/UD_Norwegian-Bokmaal")
    training_tokens = read_sentences(data_root / "no_bokmaal-ud-train.conllu")
    development_tokens = read_sentences(data_root / "no_bokmaal-ud-dev.conllu")

    schema = build_norwegian_bokmaal_schema(training_tokens)

    if checkpoint["schema"] != (serialize_token_task_schema(schema)):
        raise ValueError("Checkpoint schema does not match the pinned training data.")

    backbone_spec = NORWEGIAN_BOKMAAL_PROFILE.student_backbone

    if checkpoint["backbone_model_id"] != (backbone_spec.model_id):
        raise ValueError("Checkpoint backbone model does not match.")
    if checkpoint["backbone_revision"] != (backbone_spec.revision):
        raise ValueError("Checkpoint backbone revision does not match.")

    development_corpus = encode_norwegian_bokmaal_sentences(
        development_tokens,
        schema=schema,
    )
    batch_size = int(checkpoint["training_config"]["batch_size"])
    development_sentence_batches = tuple(
        development_corpus.sentences[start : start + batch_size]
        for start in range(
            0,
            len(development_corpus.sentences),
            batch_size,
        )
    )

    tokenizer = load_backbone_tokenizer(backbone_spec)
    model = build_pretrained_token_tagger(
        backbone_spec=backbone_spec,
        schema=schema,
        dropout_probability=0.1,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    print(
        "Evaluating checkpoint epoch:",
        int(checkpoint["epoch_index"]) + 1,
    )

    metrics = evaluate_supervised_token_task_epoch(
        model=model,
        batches=iter_supervised_token_task_batches(
            tokenizer=tokenizer,
            sentence_batches=(development_sentence_batches),
        ),
        device=torch.device("mps"),
        morphology_schema=schema.morphology,
    )

    print("Development loss:", metrics.losses.total_loss)
    print("UPOS accuracy:", metrics.upos_accuracy)
    print(
        "Lemma-rule accuracy:",
        metrics.lemma_rule_accuracy,
    )

    for (
        feature,
        true_positive_counts,
        false_positive_counts,
        false_negative_counts,
    ) in zip(
        schema.morphology.features,
        metrics.morphology_true_positive_counts,
        metrics.morphology_false_positive_counts,
        metrics.morphology_false_negative_counts,
        strict=True,
    ):
        print()
        print(feature.name)

        for label, true_positive, false_positive, false_negative in zip(
            feature.labels,
            true_positive_counts,
            false_positive_counts,
            false_negative_counts,
            strict=True,
        ):
            label_metrics = calculate_classification_metrics(
                true_positive_count=(true_positive),
                false_positive_count=(false_positive),
                false_negative_count=(false_negative),
            )

            print(
                f"  {label}: "
                f"support={label_metrics.support}, "
                f"precision={label_metrics.precision:.4f}, "
                f"recall={label_metrics.recall:.4f}, "
                f"f1={label_metrics.f1:.4f}"
            )

    analysis_path = Path(
        "runs/nb-student-baseline/development-analysis-logit-zero.json"
    )
    analysis_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch_index": int(checkpoint["epoch_index"]),
                "schema": (serialize_token_task_schema(schema)),
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Analysis:", analysis_path)


if __name__ == "__main__":
    main()
