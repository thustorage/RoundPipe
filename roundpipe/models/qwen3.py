from typing_extensions import *
import inspect
import warnings

import torch
import torch.nn as nn
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from ..context import doing_recompute, save_for_recompute, get_recompute_data
from ..roundpipe import RoundPipe
from .function import CompileForCausalLMLoss, ChunkedCompileLinearForCausalLMLoss


def _resolve_attention_type(layer: nn.Module) -> str:
    self_attn = layer.self_attn
    attention_type = getattr(self_attn, "layer_type", None)
    if attention_type is None:
        attention_type = getattr(self_attn, "attention_type", None)
    if attention_type is None:
        attention_type = getattr(layer, "attention_type", None)
    if attention_type is None:
        raise AttributeError("Qwen3 decoder layer does not expose its attention type")
    return cast(str, attention_type)


def _create_mask_compat(
    mask_function: Callable[..., torch.Tensor],
    *,
    config: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Any],
    position_ids: torch.Tensor,
    cache_position: Optional[torch.Tensor],
) -> torch.Tensor:
    parameters = inspect.signature(mask_function).parameters
    mask_kwargs: Dict[str, Any] = {
        "config": config,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
    }
    embeds_parameter = (
        "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
    )
    mask_kwargs[embeds_parameter] = inputs_embeds
    if "position_ids" in parameters:
        mask_kwargs["position_ids"] = position_ids
    if "cache_position" in parameters:
        mask_kwargs["cache_position"] = cache_position
    return mask_function(**mask_kwargs)


class Qwen3ForCausalLMPrefix(nn.Module):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.rotary_emb = model.model.rotary_emb
        self.config = model.model.config
        self.has_sliding_layers = model.model.has_sliding_layers

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = cast(torch.Tensor, self.embed_tokens(input_ids))

        # Early return to avoid host-device synchronization in create_causal_mask
        if doing_recompute():
            causal_mask_mapping, position_ids, position_embeddings = (
                get_recompute_data()
            )
            return (
                inputs_embeds,
                causal_mask_mapping,
                position_ids,
                position_embeddings,
                kwargs,
                labels,
                logits_to_keep,
            )

        if use_cache:
            warnings.warn(
                "`use_cache` will set to False. Caching behavior is not supported in RoundPipe."
            )
        use_cache = False
        if past_key_values is not None:
            warnings.warn(
                "`past_key_values` will be ignored. Caching behavior is not supported in RoundPipe."
            )
        past_key_values = None

        if cache_position is None:
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Create the masks
            causal_mask_mapping = {
                "full_attention": _create_mask_compat(
                    create_causal_mask,
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    cache_position=cache_position,
                ),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    _create_mask_compat(
                        create_sliding_window_causal_mask,
                        config=self.config,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        cache_position=cache_position,
                    )
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        save_for_recompute(causal_mask_mapping, position_ids, position_embeddings)
        return (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        )


class Qwen3ForCausalLMWrappedLayer(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.attention_type = _resolve_attention_type(layer)

    def forward(self, input):
        (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        ) = input
        hidden_states = self.layer(
            hidden_states,
            attention_mask=causal_mask_mapping[self.attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        )


class Qwen3ForCausalLMPostfix(nn.Module):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self.norm = model.model.norm
        self.vocab_size = model.config.vocab_size
        self.lm_head = model.lm_head
        self.loss_function = model.loss_function

    def forward(self, input):
        (
            hidden_states,
            causal_mask_mapping,
            position_ids,
            position_embeddings,
            kwargs,
            labels,
            logits_to_keep,
        ) = input
        hidden_states = self.norm(hidden_states)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = None
        if kwargs.get("return_logits", True):
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if logits is None:
                loss = ChunkedCompileLinearForCausalLMLoss(
                    hidden_states[:, slice_indices, :],
                    self.lm_head,
                    labels,
                    **kwargs,
                )
            else:
                loss = self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs
                )

        return CausalLMOutputWithPast(
            loss=cast(torch.FloatTensor, loss),
            logits=logits,
        )


EXPECTED_MODEL_CLASS = Qwen3ForCausalLM


def wrap_model(model: Qwen3ForCausalLM, **roundpipe_kwargs: Any) -> RoundPipe:
    model.loss_function = CompileForCausalLMLoss
    prefix = Qwen3ForCausalLMPrefix(model)
    layers = [Qwen3ForCausalLMWrappedLayer(layer) for layer in model.model.layers]
    postfix = Qwen3ForCausalLMPostfix(model)
    wrapped_model = RoundPipe(
        nn.Sequential(prefix, *layers, postfix), **roundpipe_kwargs
    )
    wrapped_model.set_original_model(model)
    return wrapped_model
