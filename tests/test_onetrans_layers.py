"""Check mixed parameters, routing, cached attention, and residual ownership."""

import inspect
import unittest
from dataclasses import replace

import torch

from onemodel.models.onetrans_v2.config import OneTransV2Config
from onemodel.models.onetrans_v2.layers import (
    GatedAttention, MixedFFN, MixedLinear, OnetransBlock, SparseMoE, SwiGLU,
    build_history_mask, build_stage_mask,
)


def make_config() -> OneTransV2Config:
    return OneTransV2Config(
        dim=16,
        layers=2,
        heads=4,
        kv_heads=2,
        ff_dim=32,
        experts=4,
        top_k=2,
    )


def make_block() -> OnetransBlock:
    return OnetransBlock(make_config())


class ResidualOwnershipTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.x = torch.randn(2, 5, 16)
        self.mask = torch.ones(2, 5, 5, dtype=torch.bool).tril()

    def test_attention_returns_only_its_projected_branch(self) -> None:
        attention = GatedAttention(make_config())
        with torch.no_grad():
            attention.Wo.weight.zero_()
            attention.Wo.bias.fill_(0.75)

        actual = attention(self.x, self.mask)
        torch.testing.assert_close(actual, torch.full_like(self.x, 0.75))

    def test_zero_residual_scale_suppresses_projection_bias(self) -> None:
        block = make_block()
        block.gamma = 0.0
        with torch.no_grad():
            block.attention.Wo.bias.fill_(1.5)

        torch.testing.assert_close(block(self.x, self.mask), self.x, atol=0, rtol=0)

    def test_zero_branches_preserve_the_input_without_an_extra_skip(self) -> None:
        block = make_block()
        with torch.no_grad():
            block.attention.Wo.weight.zero_()
            block.attention.Wo.bias.zero_()
            for module in block.ffn.modules():
                if isinstance(module, SwiGLU):
                    module.down.weight.zero_()

        torch.testing.assert_close(block(self.x, self.mask), self.x, atol=0, rtol=0)

    def test_both_branches_receive_gradients(self) -> None:
        block = make_block()
        x = self.x.clone().requires_grad_()
        output = block(x, self.mask)
        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()

        for parameter in (
            block.attention.Wo.weight,
            block.ffn.ffns[0].shared.down.weight,
            block.ffn.ffns[0].router.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)
        )


class MixedLayersTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.cfg = make_config()
        self.x = torch.randn(2, 3, self.cfg.dim)

    def test_every_layer_constructor_only_accepts_cfg(self):
        for cls in (SwiGLU, MixedLinear, GatedAttention, SparseMoE, MixedFFN, OnetransBlock):
            self.assertEqual(list(inspect.signature(cls).parameters), ["cfg"])

    def test_config_layout_is_disjoint_and_serializable(self):
        ids = [0, *self.cfg.retrieval_types, self.cfg.pre_type, *self.cfg.fine_types]
        self.assertEqual(ids, list(range(self.cfg.num_types)))
        self.assertEqual(OneTransV2Config(**self.cfg.to_dict()), self.cfg)

    def test_mixed_linear_selects_parameters_and_only_updates_used_types(self):
        layer = MixedLinear(replace(self.cfg, projection_dim=8))
        ids = [0, self.cfg.pre_type, 0]
        expected = torch.cat([
            layer.linears[t](self.x[:, i:i + 1]) for i, t in enumerate(ids)
        ], dim=1)
        actual = layer(self.x, ids)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertIsNotNone(layer.linears[0].weight.grad)
        self.assertIsNotNone(layer.linears[self.cfg.pre_type].weight.grad)
        self.assertIsNone(layer.linears[1].weight.grad)
        self.assertIsNot(layer.linears[0].weight, layer.linears[1].weight)
        torch.testing.assert_close(layer(self.x, 0), layer.linears[0](self.x))

    def test_mixed_ffn_selects_independent_moes(self):
        layer = MixedFFN(self.cfg)
        ids = self.cfg.fine_types
        expected = torch.cat([
            layer.ffns[t](self.x[:, i:i + 1]) for i, t in enumerate(ids)
        ], dim=1)
        actual = layer(self.x, ids)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertIsNone(layer.ffns[0].router.weight.grad)
        for t in ids:
            self.assertGreater(layer.ffns[t].router.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(layer(self.x, 0), layer.ffns[0](self.x))

    def test_invalid_layout_and_feature_width_are_rejected(self):
        for layer in (MixedLinear(self.cfg), MixedFFN(self.cfg)):
            for ids in ([0], -1, self.cfg.num_types):
                with self.assertRaises(ValueError):
                    layer(self.x, ids)
        with self.assertRaises(ValueError):
            SparseMoE(self.cfg)(torch.randn(2, 4, 8))

    def test_sparse_moe_matches_dense_expert_oracle_on_noncontiguous_input(self):
        cfg = replace(self.cfg, routed_scale=1.7)
        layer = SparseMoE(cfg)
        x = torch.randn(3, 2, cfg.dim).transpose(0, 1).requires_grad_()
        self.assertFalse(x.is_contiguous())
        tokens = x.reshape(-1, cfg.dim)
        scores = layer.router(tokens).float().sigmoid()
        weights, ids = scores.topk(cfg.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True) * cfg.routed_scale
        gates = torch.zeros_like(scores).scatter(1, ids, weights)
        expected = layer.shared(tokens).float()
        for i, expert in enumerate(layer.experts):
            expected = expected + gates[:, i:i + 1] * expert(tokens).float()
        actual = layer(x)
        torch.testing.assert_close(actual, expected.reshape_as(x))
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(layer.router.weight.grad.abs().sum().item(), 0)


class CachedAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.cfg = make_config()
        self.block = OnetransBlock(self.cfg)
        self.history = torch.randn(2, 4, self.cfg.dim)
        self.stage = torch.randn(2, 3, self.cfg.dim)
        self.valid = torch.ones(2, 4, dtype=torch.bool)
        self.visible = torch.tensor([[True, True, False, False], [True, True, True, False]])

    def test_cached_history_equals_joint_forward_and_retains_gradients(self):
        history = self.history.requires_grad_()
        history_out, cache = self.block.history(history, self.valid)
        actual = self.block.stage(self.stage, self.cfg.fine_types, cache, self.visible)
        full_mask = torch.zeros(2, 7, 7, dtype=torch.bool)
        full_mask[:, :4, :4] = build_history_mask(self.valid)
        full_mask[:, 4:, :] = build_stage_mask(self.visible, 3)
        joint = self.block(torch.cat((history, self.stage), dim=1), full_mask,
                           [0] * 4 + self.cfg.fine_types)
        torch.testing.assert_close(joint[:, :4], history_out)
        torch.testing.assert_close(joint[:, 4:], actual)
        self.assertEqual(cache[0].shape, (2, self.cfg.kv_heads, 4, 4))
        actual.square().sum().backward()
        self.assertGreater(history.grad.abs().sum().item(), 0)
        # 被截断的未来行为不能影响当前阶段。
        torch.testing.assert_close(history.grad[~self.visible], torch.zeros_like(history.grad[~self.visible]))

    def test_temporal_rotation_uses_relative_time_and_preserves_norm(self):
        attention = self.block.attention
        q = torch.randn(2, self.cfg.heads, 3, 4)
        k = torch.randn(2, self.cfg.heads, 4, 4)
        qt = torch.tensor([[1700000101] * 3, [1700000111] * 3])
        kt = torch.tensor([[1700000000, 1700000020, 1700000050, 1700000090]] * 2)
        rq = attention._temporal_rotate(q, qt)
        rk = attention._temporal_rotate(k, kt)
        torch.testing.assert_close(rq.square().sum(-1), q.square().sum(-1))
        shifted = (attention._temporal_rotate(q, qt + 12345)
                   @ attention._temporal_rotate(k, kt + 12345).transpose(-2, -1))
        torch.testing.assert_close(rq @ rk.transpose(-2, -1), shifted, atol=2e-5, rtol=2e-5)

    def test_time_only_changes_stage_to_history_and_masks_block_future(self):
        _, cache = self.block.history(self.history, self.valid)
        requests = torch.tensor([1700000101, 1700000111])
        times = torch.tensor([[1700000000, 1700000020, 1700000050, 1700000090]] * 2)
        invisible = torch.zeros_like(self.visible)
        untimed = self.block.stage(self.stage, self.cfg.fine_types, cache, invisible)
        timed = self.block.stage(self.stage, self.cfg.fine_types, cache, invisible, requests, times)
        torch.testing.assert_close(timed, untimed)
        baseline = self.block.stage(self.stage, self.cfg.fine_types, cache, self.visible, requests, times)
        visible_untimed = self.block.stage(self.stage, self.cfg.fine_types, cache, self.visible)
        self.assertGreater((baseline - visible_untimed).abs().max().item(), 1e-5)
        equal_times = requests[:, None].expand(-1, 4)
        aligned = self.block.stage(self.stage, self.cfg.fine_types, cache, self.visible,
                                   requests, equal_times)
        torch.testing.assert_close(aligned, visible_untimed)
        changed_history = self.history.clone()
        changed_history[~self.visible] += 100
        _, changed_cache = self.block.history(changed_history, self.valid)
        changed_stage = self.stage.clone()
        changed_stage[:, 2] += 100
        actual = self.block.stage(changed_stage, self.cfg.fine_types, changed_cache,
                                  self.visible, requests, times)
        torch.testing.assert_close(actual[:, :2], baseline[:, :2])
        # 不同 batch 行对应不同曝光，互不干扰。
        changed_stage[1] += 200
        changed = self.block.stage(changed_stage, self.cfg.fine_types, changed_cache,
                                   self.visible, requests, times)
        torch.testing.assert_close(changed[0], actual[0])

    def test_fully_masked_rows_have_finite_forward_and_backward(self):
        x = self.stage.requires_grad_()
        mask = torch.zeros(2, 3, 3, dtype=torch.bool)
        output = self.block.attention(x, mask)
        torch.testing.assert_close(output, self.block.attention.Wo.bias.expand_as(output))
        output.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
