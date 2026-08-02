from types import SimpleNamespace

import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from roundpipe.models import gpt_oss, qwen3


def test_qwen3_resolves_current_and_legacy_attention_type_locations():
    current_layer = SimpleNamespace(
        self_attn=SimpleNamespace(layer_type="sliding_attention")
    )
    legacy_layer = SimpleNamespace(
        self_attn=SimpleNamespace(), attention_type="full_attention"
    )

    assert qwen3._resolve_attention_type(current_layer) == "sliding_attention"
    assert qwen3._resolve_attention_type(legacy_layer) == "full_attention"


def test_qwen3_calls_current_mask_contract():
    inputs_embeds = torch.randn(1, 3, 4)
    position_ids = torch.arange(3).unsqueeze(0)

    def current_mask(
        *,
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids,
    ):
        return (
            config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
        )

    result = qwen3._create_mask_compat(
        current_mask,
        config="config",
        inputs_embeds=inputs_embeds,
        attention_mask="mask",
        past_key_values=None,
        position_ids=position_ids,
        cache_position=None,
    )

    assert result == ("config", inputs_embeds, "mask", None, position_ids)


def test_qwen3_calls_legacy_mask_contract():
    inputs_embeds = torch.randn(1, 3, 4)
    position_ids = torch.arange(3).unsqueeze(0)
    cache_position = position_ids.squeeze(0)

    def legacy_mask(
        *,
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids,
    ):
        return (
            config,
            input_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids,
        )

    result = qwen3._create_mask_compat(
        legacy_mask,
        config="config",
        inputs_embeds=inputs_embeds,
        attention_mask="mask",
        past_key_values=None,
        position_ids=position_ids,
        cache_position=cache_position,
    )

    assert result == (
        "config",
        inputs_embeds,
        "mask",
        cache_position,
        None,
        position_ids,
    )


def _make_gpt_oss_experts() -> GptOssExperts:
    config = GptOssConfig(
        hidden_size=8,
        intermediate_size=4,
        num_local_experts=3,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
    )
    torch.manual_seed(0)
    experts = GptOssExperts(config)
    for parameter in experts.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    return experts


def test_gpt_oss_optimized_experts_support_current_2d_topk_contract():
    experts = _make_gpt_oss_experts()
    optimized = gpt_oss.GptOssOptExperts(experts)
    router_indices = torch.tensor([[0, 2], [1, 0], [2, 1]])
    routing_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
    reference_input = torch.randn(3, 8, requires_grad=True)
    optimized_input = reference_input.detach().clone().requires_grad_(True)

    expected = experts(reference_input, router_indices, routing_weights)
    actual = optimized(optimized_input, router_indices, routing_weights)

    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(optimized_input.grad, reference_input.grad)


def test_gpt_oss_optimized_experts_support_legacy_3d_full_router_contract():
    experts = _make_gpt_oss_experts()
    optimized = gpt_oss.GptOssOptExperts(experts)
    hidden_states = torch.randn(2, 2, 8)
    router_indices = torch.tensor([[0, 2], [1, 0], [2, 1], [1, 2]])
    full_routing_weights = torch.tensor(
        [
            [0.7, 0.0, 0.3],
            [0.6, 0.4, 0.0],
            [0.0, 0.2, 0.8],
            [0.0, 0.4, 0.6],
        ]
    )
    topk_routing_weights = torch.gather(
        full_routing_weights, 1, router_indices
    )

    expected = experts(
        hidden_states.reshape(-1, 8), router_indices, topk_routing_weights
    ).reshape_as(hidden_states)
    actual = optimized(hidden_states, router_indices, full_routing_weights)

    torch.testing.assert_close(actual, expected)


def test_gpt_oss_resolves_current_and_legacy_attention_type_locations():
    current_layer = SimpleNamespace(
        self_attn=SimpleNamespace(layer_type="sliding_attention")
    )
    intermediary_layer = SimpleNamespace(
        self_attn=SimpleNamespace(attention_type="full_attention")
    )
    legacy_layer = SimpleNamespace(
        self_attn=SimpleNamespace(), attention_type="sliding_attention"
    )

    assert gpt_oss._resolve_attention_type(current_layer) == "sliding_attention"
    assert gpt_oss._resolve_attention_type(intermediary_layer) == "full_attention"
    assert gpt_oss._resolve_attention_type(legacy_layer) == "sliding_attention"


def test_gpt_oss_optimized_experts_reject_packed_weights_at_wrap_time():
    packed_experts = SimpleNamespace(
        num_experts=3,
        hidden_size=8,
        alpha=1.702,
        limit=7.0,
        gate_up_proj=torch.empty(3, 8, 2, dtype=torch.uint8),
        gate_up_proj_bias=torch.empty(3, 8),
        down_proj=torch.empty(3, 4, 2, dtype=torch.uint8),
        down_proj_bias=torch.empty(3, 8),
    )

    with pytest.raises(TypeError, match="floating-point expert weights"):
        gpt_oss.GptOssOptExperts(packed_experts)
