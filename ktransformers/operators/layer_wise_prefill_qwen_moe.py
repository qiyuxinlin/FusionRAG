# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2MoE model."""

import inspect
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock, Qwen2MoeMLP, Qwen2MoeDecoderLayer
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.operators.experts import KTransformersMLPExpert
from ktransformers.operators.linear import KTransformerLinear

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen1.5-MoE-A2.7B"
_CONFIG_FOR_DOC = "Qwen2MoeConfig"

QWEN2MOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2MoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2MoE Model outputting raw hidden-states without any specific head on top.",
    QWEN2MOE_START_DOCSTRING,
)
class Qwen2MoePreTrainedModel(PreTrainedModel):
    config_class = Qwen2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2MOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
            should not be returned during inference.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

from ktransformers.util.custom_gguf import GGUFLoader
from transformers.configuration_utils import PretrainedConfig
import time
@add_start_docstrings(
    "The bare Qwen2MoE Model outputting raw hidden-states without any specific head on top.",
    QWEN2MOE_START_DOCSTRING,
)
class Qwen2MoeModelPerLayerPrefill(BaseInjectedModule):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2MoeDecoderLayer`]

    Args:
        config: Qwen2MoeConfig
    """
    @add_start_docstrings_to_model_forward(QWEN2MOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        per_layer_prefill_intput_threshod: int | None = 1, # if None, no per-layer prefill
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        # print(f'Total length of input_ids: {input_ids.size(1)}, {input_ids.size()}')
        per_layer_prefill_flag = False
        import time
        if per_layer_prefill_intput_threshod and per_layer_prefill_intput_threshod < input_ids.size(1):
            per_layer_prefill_flag = True
            # print("to cpu")
            t1 = time.time()
            torch.cuda.empty_cache()
            for layer in self.layers:
                self.load_layer_to(layer, "cpu")
            t2 = time.time()
            torch.cuda.empty_cache()
        else:
            pass

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            if torch.cuda.is_current_stream_capturing():
                pass
            else:
                pass
                # input_ids = input_ids.to("cpu")
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None
        t_to_gpu = 0
        t_forward = 0
        t_to_cpu = 0
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                )
            else:
                t3 = time.time()
                # hidden_states = hidden_states.to("cuda")
                # position_ids = position_ids.to("cuda")
                # cache_position = cache_position.to("cuda")
                if per_layer_prefill_flag:
                    # print(f"to gpu")
                    self.load_layer_to(decoder_layer, "cuda")
                    torch.cuda.empty_cache()
                t4 = time.time()
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
                t5 = time.time()
                if per_layer_prefill_flag:
                    # print(f"to cpu")
                    self.load_layer_to(decoder_layer, "cpu")
                    torch.cuda.empty_cache()
                t6 = time.time()
                t_forward += t5-t4
                t_to_cpu += t6-t5
                t_to_gpu += t4-t3
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)


        t7 = time.time()
        # hidden_states = hidden_states.to("cpu")
        if per_layer_prefill_flag:
            # print(f"restore")
            per_layer_prefill_flag = False
            for layer in self.layers:
                self.load_layer_to(layer, "restore")
            torch.cuda.empty_cache()
        t8 = time.time()
        # print(f"{t2-t1} seconds for loading {len(self.layers)} layers to cpu")
        # print(f"{t_to_gpu} seconds for loading {len(self.layers)} layers to cuda")
        # print(f"{t_forward} seconds for forward {len(self.layers)} layers")
        # print(f"{t_to_cpu} seconds for loading {len(self.layers)} layers to cpu")
        # print(f"{t8-t7} seconds for restoring {len(self.layers)} layers to cuda")
        # add hidden states from the last decoder n1ayer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


    # def load_layer_to(self,  layer:Qwen2MoeDecoderLayer, target:str):
    #     assert target.lower() in ["cpu", "restore"] or "cuda" in target.lower(), "target should be 'cpu' or 'cuda' or 'restore'"
    #     assert isinstance(layer, Qwen2MoeDecoderLayer), "module should be nn.ModuleList of decoder layers"

    #     # TODO Support restore to original device, not only cuda
    #     device = "cpu" if target.lower() == "cpu" else "cuda" 
    #     import time
    #     # attn部分
    #     start_time = time.time()
    #     layer.self_attn.q_proj.load_to(target)
    #     print("q_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #     start_time = time.time()
    #     layer.self_attn.k_proj.load_to(target)
    #     print("k_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #     start_time = time.time()
    #     layer.self_attn.v_proj.load_to(target)
    #     print("v_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #     start_time = time.time()
    #     layer.self_attn.o_proj.load_to(target)
    #     print("o_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #     start_time = time.time()
    #     layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)
    #     print("rotary_emb moved in {:.6f} seconds".format(time.time() - start_time))

    #     # MLP部分
    #     if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
    #         start_time = time.time()
    #         layer.mlp.gate.load_to(target)
    #         print("mlp.gate loaded in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.experts.load_to(target)
    #         print("mlp.experts loaded in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.shared_expert.gate_proj.load_to(target)
    #         print("mlp.shared_expert.gate_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.shared_expert.up_proj.load_to(target)
    #         print("mlp.shared_expert.up_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.shared_expert.down_proj.load_to(target)
    #         print("mlp.shared_expert.down_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.shared_expert.act_fn.to(device)
    #         print("mlp.shared_expert.act_fn moved in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.shared_expert_gate.to(device)
    #         print("mlp.shared_expert_gate moved in {:.6f} seconds".format(time.time() - start_time))
    #     else:
    #         start_time = time.time()
    #         layer.mlp.gate_proj.load_to(target)
    #         print("mlp.gate_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.up_proj.load_to(target)
    #         print("mlp.up_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.down_proj.load_to(target)
    #         print("mlp.down_proj loaded in {:.6f} seconds".format(time.time() - start_time))

    #         start_time = time.time()
    #         layer.mlp.act_fn.to(device)
    #         print("mlp.act_fn moved in {:.6f} seconds".format(time.time() - start_time))

    #     # Layer Norm部分
    #     start_time = time.time()
    #     layer.input_layernorm.to(device)
    #     print("input_layernorm moved in {:.6f} seconds".format(time.time() - start_time))

    #     start_time = time.time()
    #     layer.post_attention_layernorm.to(device)
    #     print("post_attention_layernorm moved in {:.6f} seconds".format(time.time() - start_time))

    def load_layer_to(self,  layer:Qwen2MoeDecoderLayer, target:str):
        assert target.lower() in ["cpu", "restore"] or "cuda" in target.lower(), "target should be 'cpu' or 'cuda' or 'restore'"
        assert isinstance(layer, Qwen2MoeDecoderLayer), "module should be nn.ModuleList of decoder layers"

        # TODO Support restore to original device, not only cuda
        device = "cpu" if target.lower() == "cpu" else "cuda" 

        # attn
        layer.self_attn.q_proj.load_to(target)
        layer.self_attn.k_proj.load_to(target)
        layer.self_attn.v_proj.load_to(target)
        layer.self_attn.o_proj.load_to(target)
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

        # mlp
        if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
            layer.mlp.gate.load_to(target)
            layer.mlp.experts.load_to(target)
            layer.mlp.shared_expert.gate_proj.load_to(target)
            layer.mlp.shared_expert.up_proj.load_to(target)
            layer.mlp.shared_expert.down_proj.load_to(target)
            layer.mlp.shared_expert.act_fn.to(device)
            layer.mlp.shared_expert_gate.to(device)
        else:
            layer.mlp.gate_proj.load_to(target)
            layer.mlp.up_proj.load_to(target)
            layer.mlp.down_proj.load_to(target)
            layer.mlp.act_fn.to(device)
        # layer norm
        layer.input_layernorm.to(device)
        layer.post_attention_layernorm.to(device)
