import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.leaderboard_v8_dataset import (
    ACTION_SLOTS,
    AMOUNT_CLASSES,
    dedupe_actions,
    nearest_amount_class,
    build_samples_from_replay,
    write_dataset_shards,
)
from tools.v8_metrics import compute_action_metrics, match_action_slots
from tools.v8_model import V8ModelConfig, build_tiny_model, export_model_bin, read_model_bin
from tools.v8_gpu_policy_loss import compute_resident_policy_loss
from tools.v8_resident_device_batch import ResidentBatchTensorView
from tools.v8_selfplay import GeneticRelaxationConfig, mutate_generation
from tools.v8_train import collate_samples, compute_supervised_loss, write_telemetry


def planet_row(pid, owner, ships, x=None, y=None):
    x = float(pid * 5 + 10) if x is None else x
    y = float(pid * 3 + 20) if y is None else y
    return [pid, owner, x, y, 2.0, ships, 1.0]


class V8DatasetTests(unittest.TestCase):
    def test_replay_alignment_uses_previous_observation_for_action_labels(self):
        replay = {
            "rewards": [1, 0],
            "steps": [
                [
                    {"action": [], "observation": {"player": 0, "step": 0, "planets": [planet_row(0, 0, 10), planet_row(1, -1, 5)], "fleets": []}},
                    {"action": [], "observation": {"player": 1, "planets": [planet_row(0, 0, 10), planet_row(1, -1, 5)], "fleets": []}},
                ],
                [
                    {"action": [[0.0, 0.0, 10.0]], "observation": {"player": 0, "step": 1, "planets": [planet_row(0, 0, 5), planet_row(1, -1, 5)], "fleets": []}},
                    {"action": [], "observation": {"player": 1, "planets": [planet_row(0, 0, 5), planet_row(1, -1, 5)], "fleets": []}},
                ],
            ],
        }
        samples, stats = build_samples_from_replay(replay, "unit.json")

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["labels_amount"][0], AMOUNT_CLASSES.index("100%"))
        self.assertEqual(samples[0]["labels_source"][0], 0)
        self.assertEqual(stats["aligned_samples"], 1)

    def test_target_label_uses_future_fleet_hit_before_angle_fallback(self):
        replay = {
            "rewards": [1, 0],
            "steps": [
                [
                    {
                        "action": [],
                        "observation": {
                            "player": 0,
                            "step": 0,
                            "planets": [
                                planet_row(0, 0, 20, x=0.0, y=0.0),
                                planet_row(1, -1, 5, x=10.0, y=0.0),
                                planet_row(2, -1, 5, x=0.0, y=10.0),
                            ],
                            "fleets": [],
                        },
                    }
                ],
                [
                    {
                        "action": [[0.0, 0.0, 10.0]],
                        "observation": {
                            "player": 0,
                            "step": 1,
                            "planets": [
                                planet_row(0, 0, 10, x=0.0, y=0.0),
                                planet_row(1, -1, 5, x=10.0, y=0.0),
                                planet_row(2, -1, 5, x=0.0, y=10.0),
                            ],
                            "fleets": [[100, 0, 0.0, 0.0, 0.0, 0, 10]],
                        },
                    }
                ],
                [
                    {
                        "action": [],
                        "observation": {
                            "player": 0,
                            "step": 2,
                            "planets": [
                                planet_row(0, 0, 10, x=0.0, y=0.0),
                                planet_row(1, -1, 5, x=10.0, y=0.0),
                                planet_row(2, -1, 5, x=0.0, y=10.0),
                            ],
                            "fleets": [[100, 0, 0.0, 9.7, 0.0, 0, 10]],
                        },
                    }
                ],
                [
                    {
                        "action": [],
                        "observation": {
                            "player": 0,
                            "step": 3,
                            "planets": [
                                planet_row(0, 0, 10, x=0.0, y=0.0),
                                planet_row(1, -1, 5, x=10.0, y=0.0),
                                planet_row(2, 0, 5, x=0.0, y=10.0),
                            ],
                            "fleets": [],
                        },
                    }
                ],
            ],
        }

        samples, stats = build_samples_from_replay(replay, "unit.json")

        self.assertEqual(samples[0]["labels_target"][0], 2)
        self.assertEqual(stats["target_trace_hits"], 1)
        self.assertEqual(stats["target_angle_fallbacks"], 0)

    def test_dedupe_actions_keeps_largest_per_source_and_truncates_to_slots(self):
        actions = [[0.0, 0.0, 3.0], [0.0, 1.0, 9.0]]
        actions.extend([[float(i), 0.0, float(i)] for i in range(1, 11)])

        kept, stats = dedupe_actions(actions)

        self.assertEqual(len(kept), ACTION_SLOTS)
        self.assertEqual(kept[0].source, 10)
        self.assertEqual(stats["duplicate_actions"], 1)
        self.assertEqual(stats["dropped_overflow_actions"], 3)

    def test_dataset_shard_roundtrip_writes_required_fields(self):
        replay = {
            "rewards": [1, 0],
            "steps": [
                [{"action": [], "observation": {"player": 0, "step": 0, "planets": [planet_row(0, 0, 20), planet_row(1, -1, 5)], "fleets": []}}],
                [{"action": [[0.0, 0.0, 10.0]], "observation": {"player": 0, "step": 1, "planets": [planet_row(0, 0, 10), planet_row(1, -1, 5)], "fleets": []}}],
            ],
        }
        samples, _ = build_samples_from_replay(replay, "unit.json")
        with tempfile.TemporaryDirectory() as tmp:
            metadata = write_dataset_shards(samples, Path(tmp), shard_size=1)
            with np.load(Path(tmp) / metadata["shards"][0]["file"]) as shard:
                for field in [
                    "tokens",
                    "token_type_ids",
                    "owner_ids",
                    "sample_offsets",
                    "planet_mask",
                    "labels_fire",
                    "labels_source",
                    "labels_target",
                    "labels_amount",
                    "episode_id",
                    "step",
                    "player",
                ]:
                    self.assertIn(field, shard.files)

    def test_amount_classes_cover_all_plan_values(self):
        self.assertEqual(nearest_amount_class(100, 100), AMOUNT_CLASSES.index("100%"))
        self.assertEqual(nearest_amount_class(100, 50), AMOUNT_CLASSES.index("50%"))
        self.assertEqual(nearest_amount_class(10_000, 500), AMOUNT_CLASSES.index("500"))


class V8MetricAndModelTests(unittest.TestCase):
    def test_match_action_slots_finds_best_assignment_without_scipy(self):
        predicted = [
            {"fire": 0.8, "source": 1, "target": 3, "amount": 2},
            {"fire": 0.9, "source": 0, "target": 2, "amount": 0},
        ]
        truth = [
            {"source": 0, "target": 2, "amount": 0},
            {"source": 1, "target": 3, "amount": 2},
        ]
        matches = match_action_slots(predicted, truth)

        self.assertEqual(matches, [(1, 0), (0, 1)])

    def test_compute_action_metrics_reports_non_all_in_amount_accuracy(self):
        metrics = compute_action_metrics(
            [[{"source": 0, "target": 1, "amount": 0}, {"source": 2, "target": 3, "amount": 2}]],
            [[{"fire": 0.8, "source": 0, "target": 1, "amount": 0}, {"fire": 0.7, "source": 2, "target": 3, "amount": 2}]],
        )

        self.assertEqual(metrics["amountAccuracy"], 1.0)
        self.assertEqual(metrics["amountNonAllInAccuracy"], 1.0)

    def test_model_bin_roundtrip_has_header_tensors_and_checksum(self):
        config = V8ModelConfig(d_model=16, encoder_layers=1, decoder_layers=1, heads=4)
        model = build_tiny_model(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.bin"
            export_model_bin(model, config, path)
            parsed = read_model_bin(path)

        self.assertEqual(parsed["schema"], "OWV8")
        self.assertIn("token_projection.weight", parsed["tensors"])
        self.assertGreater(parsed["checksum"], 0)

    def test_collate_and_loss_accept_dataset_samples(self):
        replay = {
            "rewards": [1, 0],
            "steps": [
                [{"action": [], "observation": {"player": 0, "step": 0, "planets": [planet_row(0, 0, 20), planet_row(1, -1, 5)], "fleets": []}}],
                [{"action": [[0.0, 0.0, 20.0]], "observation": {"player": 0, "step": 1, "planets": [planet_row(0, 0, 0), planet_row(1, -1, 5)], "fleets": []}}],
            ],
        }
        samples, _ = build_samples_from_replay(replay, "unit.json")
        batch = collate_samples(samples)
        model = build_tiny_model(V8ModelConfig(d_model=16, encoder_layers=1, decoder_layers=1, heads=4))
        outputs = model(batch["tokens"], batch["token_type_ids"], batch["owner_ids"], padding_mask=batch["padding_mask"], planet_mask=batch["planet_mask"])

        loss, parts = compute_supervised_loss(outputs, batch)

        self.assertGreater(float(loss.detach().cpu()), 0.0)
        self.assertIn("fire", parts)

    def test_write_telemetry_creates_latest_dashboard_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_telemetry(
                Path(tmp),
                run_id="unit",
                generation=0,
                metrics={"actionF1": 0.5, "sourceTop1": 0.5, "targetTop1": 0.4, "targetTop3": 0.7, "amountAccuracy": 0.6, "amountNonAllInAccuracy": 0.3, "duplicateSourceRate": 0.1, "fireRate": 0.2},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["runId"], "unit")
        self.assertIn("amountHistogram", payload)

    def test_genetic_relaxation_defaults_create_population_children(self):
        config = GeneticRelaxationConfig(population=4, elites=1, mutations_per_elite=3)
        children = mutate_generation(["best.bin"], config=config, generation=1)

        self.assertEqual(len(children), 4)
        self.assertEqual(children[0]["parent"], "best.bin")

    def test_resident_policy_loss_ignores_inactive_slot_masked_logits(self):
        batch = ResidentBatchTensorView(
            tokens=torch.zeros((1, 2, 3)),
            token_type_ids=torch.zeros((1, 2), dtype=torch.long),
            owner_ids=torch.zeros((1, 2), dtype=torch.long),
            padding_mask=torch.zeros((1, 2), dtype=torch.bool),
            planet_mask=torch.ones((1, 64), dtype=torch.bool),
            labels_fire=torch.tensor([[0, 1]], dtype=torch.int32),
            labels_source=torch.tensor([[-1, 3]], dtype=torch.int32),
            labels_target=torch.tensor([[-1, 4]], dtype=torch.int32),
            labels_amount=torch.tensor([[-1, 0]], dtype=torch.int32),
            labels_confidence=torch.zeros((1, 2)),
        )
        outputs = {
            "fire_logits": torch.tensor([[-2.0, 2.0]]),
            "source_logits": torch.zeros((1, 2, 64)),
            "target_logits": torch.zeros((1, 2, 64)),
            "amount_logits": torch.zeros((1, 2, 16)),
        }
        outputs["source_logits"][0, 0, :] = -torch.finfo(torch.float32).max
        outputs["target_logits"][0, 0, :] = -torch.finfo(torch.float32).max

        loss, _metrics = compute_resident_policy_loss(outputs, batch, torch.tensor([1.0]))

        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
