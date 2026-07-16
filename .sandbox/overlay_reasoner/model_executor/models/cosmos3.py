# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import regex

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    # Cosmos3 unified checkpoints store a Qwen3-VL understanding tower
    # alongside a generation tower in a flat key layout. This mapper drops
    # the generation tower weights and rewrites the understanding tower keys
    # into the nested form expected by Qwen3VLForConditionalGeneration.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            regex.compile(
                r"^(layers\.|embed_tokens\.|norm\.)(.+)$"
            ): r"language_model.model.\1\2",
            regex.compile(
                r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)"
            ): r"visual.\1",
            regex.compile(r"^audio_modality_embed(?:\..*)?$"): None,
            regex.compile(r"^action_modality_embed(?:\..*)?$"): None,
        },
        orig_to_new_substr={
            "_moe_gen": None,
            ".add_q_proj.": None,
            ".add_k_proj.": None,
            ".add_v_proj.": None,
            ".to_add_out.": None,
            ".norm_added_q.": None,
            ".norm_added_k.": None,
            ".to_q.": ".q_proj.",
            ".to_k.": ".k_proj.",
            ".to_v.": ".v_proj.",
            ".to_out.": ".o_proj.",
            ".norm_q.": ".q_norm.",
            ".norm_k.": ".k_norm.",
            # ModelOpt-native dialect (diffusers/transformers read these; vLLM reads
            # weight_scale/input_scale instead) -- drop so AutoWeightsLoader doesn't choke.
            ".input_quantizer.": None,
            ".weight_quantizer.": None,
            ".output_quantizer.": None,
        },
        orig_to_new_prefix={
            "proj_in.": None,
            "proj_out.": None,
            "time_embedder.": None,
            "audio_proj_in.": None,
            "audio_proj_out.": None,
            "action_proj_in.": None,
            "action_proj_out.": None,
            "lm_head.": "language_model.lm_head.",
        },
    )

    allow_patterns_overrides = ["transformer/*.safetensors"]

    """
    Cosmos3 checkpoint separates transformer weights and vision_encoder weights
    into separate directories, as it's in diffusers checkpoint format.
    Using secondary_weights here to load all necessary weights for
    the Reasoner-only part.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=vllm_config.model_config.model,
                revision=vllm_config.model_config.revision,
                prefix="",
                allow_patterns_overrides=["vision_encoder/*.safetensors"],
            ),
        ]
        # Reasoning activation-scale overlay (Approach A). The transformer/ shards
        # carry generation-calibrated input_scales; a checkpoint that also declares
        # `activation_scale_overlay.reasoning` in hf_quant_config.json ships a small
        # side-car with reasoning-calibrated input_scales. Loading it as a secondary
        # source (i.e. AFTER the primary transformer weights) makes those reasoning
        # scales OVERRIDE the generation ones on the understanding tower before
        # process_weights_after_loading finalizes them. Same hf_to_vllm_mapper maps
        # `layers.N.self_attn.to_q.input_scale` -> the vLLM q_proj.input_scale param.
        overlay = self._reasoning_overlay_source(vllm_config)
        if overlay is not None:
            self.secondary_weights.append(overlay)

    @staticmethod
    def _reasoning_overlay_source(
        vllm_config: VllmConfig,
    ) -> "DefaultModelLoader.Source | None":
        model = vllm_config.model_config.model
        cfg_path = os.path.join(model, "hf_quant_config.json")
        if not os.path.isfile(cfg_path):
            return None
        try:
            with open(cfg_path) as f:
                hq = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        rel = (hq.get("activation_scale_overlay") or {}).get("reasoning")
        if not rel:
            return None
        # Point the source at the sidecar's own directory rather than the
        # checkpoint root. _prepare_weights runs filter_duplicate_safetensors_files
        # against `<model_or_path>/model.safetensors.index.json`; the top-level
        # index does not register the overlay sidecar, so rooting here would filter
        # it out ("Cannot find any model weights"). The reasoner/ subdir has no
        # index file, so the filter is skipped and the sidecar loads. Tensor names
        # inside the file are unchanged, so hf_to_vllm_mapper still applies.
        overlay_dir = os.path.join(model, os.path.dirname(rel))
        overlay_file = os.path.basename(rel)
        return DefaultModelLoader.Source(
            model_or_path=overlay_dir,
            revision=vllm_config.model_config.revision,
            prefix="",
            allow_patterns_overrides=[overlay_file],
        )
