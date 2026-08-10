"""GPT-BERT configuration (vendored).

Vendored from ``BabyLM-community/babylm-baseline-100m-gpt-bert-mixed`` (Apache
License 2.0), which implements the LTG GPT-BERT architecture
(https://github.com/ltgoslo/gpt-bert; Charpentier & Samuel, 2024). Adapted for
transformers 5.x and Prism:

* the original ``__init__`` hard-coded the base dimensions in the
  ``config_file is None`` branch, silently overwriting any kwargs — reworked to
  real keyword defaults so ``ModelConfig(hidden_size=320, num_layers=12, …)``
  actually takes effect;
* the custom ``to_dict`` / ``to_json_string`` / ``to_json_file`` overrides
  (whose ``to_json_file`` lacked the ``use_diff`` kwarg transformers 5.x passes
  during ``save_pretrained``, breaking checkpoint saving) are removed so
  serialization inherits the standard ``PretrainedConfig`` behaviour;
* ``model_type`` is declared so the config registers cleanly.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "gpt_bert"

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 2560,
        max_position_embeddings: int = 512,
        position_bucket_size: int = 32,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        vocab_size: int = 16384,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.position_bucket_size = position_bucket_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        # The modeling code reads both names; keep them in lock-step.
        self.num_hidden_layers = num_layers
        self.vocab_size = vocab_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
