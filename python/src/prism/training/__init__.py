"""Training policies and loss calculations for Prism models."""

from prism.training.batches import (
    SupervisedTokenTaskBatch,
    build_supervised_sentence_batches,
    build_supervised_token_task_batch,
    iter_supervised_token_task_batches,
)
from prism.training.class_weights import (
    build_token_task_loss_weights,
)
from prism.training.morphology_bundle_reranking import (
    build_morphology_bundle_reranker_spec,
    deserialize_morphology_bundle_reranker_spec,
    serialize_morphology_bundle_reranker_spec,
)
from prism.training.morphology_agreement import (
    deserialize_morphology_agreement_refiner_spec,
    serialize_morphology_agreement_refiner_spec,
)
from prism.training.checkpoints import (
    TOKEN_TASK_CHECKPOINT_FORMAT_VERSION,
    backbone_layer_aggregation_strategy_from_checkpoint,
    token_pooling_strategy_from_checkpoint,
    token_task_head_architecture_from_checkpoint,
    validate_token_task_checkpoint_format,
    character_vocabulary_from_checkpoint,
    maximum_character_count_from_checkpoint,
    morphology_logit_correction_from_checkpoint,
    morphology_bundle_reranker_spec_from_checkpoint,
    morphology_agreement_refiner_spec_from_checkpoint,
)
from prism.training.config import SupervisedTrainingConfig
from prism.training.distillation import (
    CombinedTokenTaskLosses,
    TokenTaskDistillationPolicy,
    calculate_binary_distillation_loss,
    calculate_categorical_distillation_loss,
    calculate_decoupled_categorical_distillation_loss,
    combine_token_task_losses,
    compute_token_task_distillation_loss,
)
from prism.training.epochs import (
    DistilledEpochMetrics,
    SupervisedEpochMetrics,
    SupervisedEvaluationMetrics,
    evaluate_supervised_token_task_epoch,
    train_distilled_token_task_epoch,
    train_supervised_token_task_epoch,
)
from prism.training.losses import (
    MorphologyBundleLossPolicy,
    MorphologyBundleLossResult,
    TokenTaskLosses,
    TokenTaskLossWeights,
    calculate_morphology_bundle_loss,
    compute_token_task_loss,
)
from prism.training.optimizers import build_supervised_adamw_optimizer
from prism.training.runner import (
    SupervisedTrainingEpochResult,
    SupervisedTrainingRunResult,
    run_supervised_training_epochs,
)
from prism.training.schedulers import build_linear_warmup_decay_scheduler
from prism.training.steps import (
    evaluate_supervised_token_task_step,
    train_distilled_token_task_step,
    train_supervised_token_task_step,
)

__all__ = [
    "TokenTaskLosses",
    "MorphologyBundleLossPolicy",
    "MorphologyBundleLossResult",
    "calculate_morphology_bundle_loss",
    "compute_token_task_loss",
    "SupervisedTokenTaskBatch",
    "build_supervised_sentence_batches",
    "build_supervised_token_task_batch",
    "iter_supervised_token_task_batches",
    "evaluate_supervised_token_task_step",
    "train_supervised_token_task_step",
    "SupervisedTrainingConfig",
    "build_supervised_adamw_optimizer",
    "build_linear_warmup_decay_scheduler",
    "SupervisedEpochMetrics",
    "SupervisedEvaluationMetrics",
    "evaluate_supervised_token_task_epoch",
    "train_supervised_token_task_epoch",
    "train_distilled_token_task_step",
    "SupervisedTrainingEpochResult",
    "SupervisedTrainingRunResult",
    "run_supervised_training_epochs",
    "TokenTaskLossWeights",
    "build_token_task_loss_weights",
    "build_morphology_bundle_reranker_spec",
    "deserialize_morphology_bundle_reranker_spec",
    "serialize_morphology_bundle_reranker_spec",
    "deserialize_morphology_agreement_refiner_spec",
    "serialize_morphology_agreement_refiner_spec",
    "calculate_binary_distillation_loss",
    "calculate_categorical_distillation_loss",
    "calculate_decoupled_categorical_distillation_loss",
    "compute_token_task_distillation_loss",
    "CombinedTokenTaskLosses",
    "TokenTaskDistillationPolicy",
    "combine_token_task_losses",
    "DistilledEpochMetrics",
    "train_distilled_token_task_epoch",
    "TOKEN_TASK_CHECKPOINT_FORMAT_VERSION",
    "backbone_layer_aggregation_strategy_from_checkpoint",
    "token_pooling_strategy_from_checkpoint",
    "token_task_head_architecture_from_checkpoint",
    "validate_token_task_checkpoint_format",
    "character_vocabulary_from_checkpoint",
    "maximum_character_count_from_checkpoint",
    "morphology_logit_correction_from_checkpoint",
    "morphology_bundle_reranker_spec_from_checkpoint",
    "morphology_agreement_refiner_spec_from_checkpoint",
]
