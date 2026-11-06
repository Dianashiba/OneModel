"""Test the composed model with token tensors, independently of data tokenization."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from onemodel.models.onetrans_v2.config import OneTransV2Config
from onemodel.models.onetrans_v2.model import OneTransV2, select_context


class OneTransModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.cfg = OneTransV2Config(dim=16, layers=2, heads=4, kv_heads=2,
                                   ff_dim=24, experts=3, top_k=2, codebook_size=8)
        self.model = OneTransV2(self.cfg)
        self.batch = {
            "history_tokens": torch.randn(2, 4, 16, requires_grad=True),
            "history_valid": torch.tensor([[True, True, True, True], [True, True, False, False]]),
            "history_times": torch.tensor([[100, 110, 120, 130], [100, 110, 0, 0]]),
            "owners": torch.tensor([0, 1, 0]),
            "anchors": torch.tensor([1, 1, 3]),
            "times": torch.tensor([115, 115, 140]),
            "retrieval_context": torch.randn(3, 16, requires_grad=True),
            "pre_tokens": torch.randn(3, 1, 16, requires_grad=True),
            "fine_tokens": torch.randn(3, 3, 16, requires_grad=True),
            "decisions": torch.tensor([[0, 1, 0, 2], [1, 0, 1, 3], [1, 1, 0, 0]]),
            "sid_targets": torch.tensor([[1, 2, 3], [4, 5, 6], [2, 4, 7]]),
        }

    def context(self):
        return select_context(self.model.encode(self.batch), self.batch["owners"])

    def test_joint_forward_shapes_single_history_pass_and_backward(self):
        with patch.object(self.model.blocks[0], "history", wraps=self.model.blocks[0].history) as first:
            with patch.object(self.model.blocks[1], "history", wraps=self.model.blocks[1].history) as second:
                output = self.model(self.batch)
        self.assertEqual(first.call_count, 1)
        self.assertEqual(second.call_count, 1)
        self.assertEqual(set(output.logits), {
            "decision_0", "decision_1", "decision_2", "decision_3",
            "sid_0", "sid_1", "sid_2", "pre", "fine",
        })
        loss = 0
        for i, size in enumerate(self.cfg.decision_sizes):
            logits = output.logits[f"decision_{i}"]
            self.assertEqual(logits.shape, (3, size))
            loss = loss + F.cross_entropy(logits, self.batch["decisions"][:, i])
        for i in range(self.cfg.sid_levels):
            logits = output.logits[f"sid_{i}"]
            self.assertEqual(logits.shape, (3, self.cfg.codebook_size))
            loss = loss + F.cross_entropy(logits, self.batch["sid_targets"][:, i])
        for stage in ("pre", "fine"):
            logits = output.logits[stage]
            self.assertEqual(logits.shape, (3, len(self.cfg.tasks)))
            loss = loss + F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        loss.backward()
        for name in ("history_tokens", "retrieval_context", "pre_tokens", "fine_tokens"):
            grad = self.batch[name].grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_each_stage_backpropagates_to_shared_history(self):
        output = self.model(self.batch)
        for name in ("sid_2", "pre", "fine"):
            grad, = torch.autograd.grad(output.logits[name].square().sum(),
                                       self.batch["history_tokens"], retain_graph=True)
            self.assertGreater(grad.abs().sum().item(), 0)
            torch.testing.assert_close(grad[1, 2:], torch.zeros_like(grad[1, 2:]))

    def test_teacher_forcing_matches_short_prefix_predictions(self):
        full = self.model(self.batch).logits
        context = self.context()
        # CTX -> DTp -> DTd -> SID0 -> SID1: each prefix predicts the next token.
        cases = [(None, None, ["decision_0", "decision_1", "decision_2"]),
                 (self.batch["decisions"][:, :3], None, ["decision_3"])]
        cases.extend((self.batch["decisions"], self.batch["sid_targets"][:, :i], [f"sid_{i}"])
                     for i in range(self.cfg.sid_levels))
        for decisions, sids, keys in cases:
            hidden = self.model.retrieval(self.batch["retrieval_context"], context,
                                          self.batch["anchors"], self.batch["times"], decisions, sids)
            logits = self.model.retrieval_logits(hidden)
            for key in keys:
                torch.testing.assert_close(logits[key], full[key])

    def test_target_cannot_leak_into_its_own_prediction_or_ranking(self):
        baseline = self.model(self.batch).logits
        changed = dict(self.batch)
        changed["sid_targets"] = (self.batch["sid_targets"] + 1) % self.cfg.codebook_size
        after = self.model(changed).logits
        for name in ("decision_0", "decision_1", "decision_2", "decision_3", "sid_0", "pre", "fine"):
            torch.testing.assert_close(after[name], baseline[name])
        # Only the final target changes: it is not fed into any input position.
        changed["sid_targets"] = self.batch["sid_targets"].clone()
        changed["sid_targets"][:, -1] = (changed["sid_targets"][:, -1] + 1) % self.cfg.codebook_size
        for name, logits in self.model(changed).logits.items():
            torch.testing.assert_close(logits, baseline[name])
        changed = dict(self.batch)
        changed["decisions"] = self.batch["decisions"].clone()
        changed["decisions"][:, 3] = (changed["decisions"][:, 3] + 1) % self.cfg.decision_sizes[3]
        after = self.model(changed).logits
        for name in ("decision_0", "decision_1", "decision_2", "decision_3", "pre", "fine"):
            torch.testing.assert_close(after[name], baseline[name])

    def test_owner_selection_matches_individual_exposure_scoring(self):
        base_context = self.model.encode(self.batch)
        expanded = select_context(base_context, self.batch["owners"])
        self.assertEqual(len(expanded["cache"]), self.cfg.layers)
        actual = self.model.score(self.batch["fine_tokens"], expanded,
                                  self.batch["anchors"], self.batch["times"])
        for row in range(3):
            one = select_context(base_context, self.batch["owners"][row:row + 1])
            expected = self.model.score(self.batch["fine_tokens"][row:row + 1], one,
                                        self.batch["anchors"][row:row + 1],
                                        self.batch["times"][row:row + 1])
            torch.testing.assert_close(actual[row:row + 1], expected)

    def test_future_history_and_other_stages_do_not_change_current_score(self):
        baseline = self.model(self.batch).logits
        changed = dict(self.batch)
        changed["history_tokens"] = self.batch["history_tokens"].detach().clone()
        changed["history_tokens"][0, 2:] += 100
        changed["history_tokens"][1, 2:] += 100  # padding
        changed["pre_tokens"] = self.batch["pre_tokens"] + 100
        changed["retrieval_context"] = self.batch["retrieval_context"] + 100
        after = self.model(changed).logits
        torch.testing.assert_close(after["fine"][:2], baseline["fine"][:2])

    def test_start_only_history_and_single_sid_level(self):
        cfg = replace(self.cfg, sid_levels=1, sid_weights=[1.0])
        model = OneTransV2(cfg)
        batch = dict(self.batch)
        for name in ("history_tokens", "history_valid", "history_times"):
            batch[name] = batch[name][:, :1]
        batch["anchors"] = torch.zeros(3, dtype=torch.long)
        batch["sid_targets"] = batch["sid_targets"][:, :1]
        output = model(batch)
        self.assertIn("sid_0", output.logits)
        self.assertNotIn("sid_1", output.logits)
        self.assertTrue(all(torch.isfinite(x).all() for x in output.logits.values()))

    def test_invalid_context_and_stage_are_rejected(self):
        context = self.model.encode(self.batch)
        with self.assertRaises(ValueError):
            select_context(context, torch.tensor([-1]))
        context = select_context(context, self.batch["owners"])
        with self.assertRaises(ValueError):
            self.model.score(self.batch["pre_tokens"], context, torch.tensor([4, 1, 1]), self.batch["times"], "pre")
        with self.assertRaises(ValueError):
            self.model.score(self.batch["fine_tokens"], context, self.batch["anchors"], self.batch["times"], "other")
        with self.assertRaises(ValueError):
            self.model.retrieval(self.batch["retrieval_context"], context, self.batch["anchors"],
                                 self.batch["times"], sids=self.batch["sid_targets"])


if __name__ == "__main__":
    unittest.main()
