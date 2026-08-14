# Vendored from ltg/norbert4-base (Hugging Face Hub, revision
# 386ba2dc5ae5f95fec86d580c5fc4af34d380126) — the NorBERT4 "GPT-BERT" encoder by
# the Language Technology Group (LTG), University of Oslo. Distributed under the
# Apache License 2.0 (model card: "The checkpoints are distributed freely under
# Apache 2.0, anyone can use our models").
#
# VENDORED AND MODIFIED for Prism: this file is a copy adapted to power Prism's
# per-language PrismBERT pretraining (Apple-Silicon/MPS-friendly non-FlashAttention
# path, our own tokenizer/config sizing, AutoModel registration). Original code
# (c) the NorBERT/LTG authors under Apache-2.0; Prism modifications (c) Scoop
# Software GmbH under Apache-2.0. See NOTICE / release docs for attribution.

from __future__ import annotations

import json
from pathlib import Path
import copy
from transformers.configuration_utils import PretrainedConfig


class GptBertConfig(PretrainedConfig):

    def __init__(
        self,
        config_file: Path | str | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = "norbert4"

        if config_file is not None:
            if type(config_file) is str:
                config_file = Path(config_file)
            assert type(config_file) is not Path, "The config_file should either be a Path or str"
            with config_file.open("r") as file:
                config = json.load(file)

            for attr, value in config.items():
                if isinstance(value, str):
                    value = value.lower()
                setattr(self, attr, value)

        for attr, value in kwargs.items():
            if isinstance(value, str):
                value = value.lower()
            setattr(self, attr, value)
