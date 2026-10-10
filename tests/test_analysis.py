"""核对绘图所依赖的真实时序、反转位置和模型保存，而非图片像素。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from rl_rnn_example.analysis import (
    collect_predictions,
    compute_vector_field,
    load_results,
    save_results,
)
from rl_rnn_example.dataset import load_session
from rl_rnn_example.gru import GRUModel
from rl_rnn_example.rw import RWModel
from rl_rnn_example.training import fit_gru


def example_block():
    """短序列只用于核对模型计算，不代表真实实验结果。"""
    return {
        "animal_name": "V",
        "session_name": "V_example",
        "block_order": 1,
        "block_type": "what",
        "trial_numbers": np.arange(11, 15),
        "actions": np.array([0, 1, 1, 0]),
        "rewards": np.array([1, 0, 1, 0]),
        "reversal_trial": 13,
    }


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.block = example_block()
        self.models = {"RW": RWModel(), "GRU": GRUModel()}

    def test_reversal_survives_trial_truncation(self):
        # 反转在原始第 34 次，保留 11–20 次后仍应知道其原始位置。
        y = np.zeros((160, 13), dtype=int)
        for index in range(2):
            rows = y[index * 80 : (index + 1) * 80]
            rows[:, 0] = np.arange(80) % 2
            rows[:, 1] = 1 - rows[:, 0]
            rows[:, 2] = 1
            rows[:, 3] = 1
            rows[:, 5] = np.arange(1, 81)
            rows[33, 6] = 1
            rows[:, 7] = 1 if index == 0 else 13
            rows[:, 8] = index + 1
            rows[:, 9] = index + 1
            rows[:, 12] = 1
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "SPKcounts_V20161005cue_MW_250X250ms.mat"
            sio.savemat(file_path, {"Y": y})
            blocks = load_session(file_path, trial_start=11, trial_end=20)
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(block["reversal_trial"], 34)
            np.testing.assert_array_equal(block["trial_numbers"], np.arange(11, 21))
        np.testing.assert_array_equal(blocks[0]["actions"], y[10:20, 0])
        np.testing.assert_array_equal(blocks[1]["actions"], y[90:100, 1])

    def test_sequence_matches_repeated_single_steps(self):
        # 单步接口必须与整条序列完全对齐，包括最后一次反馈后的终点。
        for name, model in self.models.items():
            with self.subTest(model=name):
                result = model.predict_block(self.block["actions"], self.block["rewards"])
                state = np.zeros(2)
                for index, (action, reward) in enumerate(
                    zip(self.block["actions"], self.block["rewards"])
                ):
                    np.testing.assert_allclose(result["states"][index], state, atol=1e-6)
                    np.testing.assert_allclose(
                        result["probabilities"][index],
                        model.choice_probabilities(state),
                        atol=1e-6,
                    )
                    original_state = state.copy()
                    next_state = model.update_state(state, action, reward)
                    np.testing.assert_array_equal(state, original_state)
                    state = next_state
                np.testing.assert_allclose(result["final_state"], state, atol=1e-6)

    def test_current_feedback_cannot_change_current_prediction(self):
        actions = self.block["actions"].copy()
        rewards = self.block["rewards"].copy()
        actions[2] = 1 - actions[2]
        rewards[2] = 1 - rewards[2]
        for name, model in self.models.items():
            with self.subTest(model=name):
                before = model.predict_block(self.block["actions"], self.block["rewards"])
                after = model.predict_block(actions, rewards)
                np.testing.assert_allclose(
                    before["probabilities"][:3], after["probabilities"][:3], atol=1e-6
                )

    def test_gru_forward_keeps_gradients(self):
        model = self.models["GRU"]
        inputs = torch.tensor(
            np.column_stack([self.block["actions"], self.block["rewards"]]),
            dtype=torch.float32,
        ).unsqueeze(1)
        targets = torch.tensor(self.block["actions"])
        logits = model(inputs)["logits"].squeeze(1)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        self.assertGreater(model.gru.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertGreater(model.linear.weight.grad.abs().sum().item(), 0)

    def test_rw_vector_field_matches_update_formula(self):
        model = RWModel(alpha=0.25, iTemp=3)
        field = compute_vector_field(model, np.array([[0.1, 0.2], [0.8, 0.9]]), 4, (0, 1))
        for condition in field["conditions"]:
            if condition["action"] == 0:
                expected_dx = 0.25 * (condition["reward"] - field["x"])
                np.testing.assert_allclose(condition["dx"], expected_dx)
                np.testing.assert_array_equal(condition["dy"], 0)
            else:
                expected_dy = 0.25 * (condition["reward"] - field["y"])
                np.testing.assert_array_equal(condition["dx"], 0)
                np.testing.assert_allclose(condition["dy"], expected_dy)
        expected_probability = 1 / (1 + np.exp(3 * (field["y"] - field["x"])))
        np.testing.assert_allclose(field["probability"], expected_probability)

    def test_saved_models_reproduce_predictions_and_dynamics(self):
        # 同时验证非默认 float64 参数能被单步接口和重载流程正确处理。
        self.models["GRU"] = self.models["GRU"].double()
        records = collect_predictions(self.models, [self.block])
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_results(
                output_dir,
                self.models,
                records,
                config={"seed": 0},
                splits={"test": [self.block]},
                metrics={},
                history={},
            )
            loaded_models, loaded_records, experiment = load_results(output_dir)
        self.assertEqual(experiment["config"]["seed"], 0)
        self.assertEqual(loaded_records[0]["reversal_trial"], 13)
        for name, model in loaded_models.items():
            result = model.predict_block(self.block["actions"], self.block["rewards"])
            for key in ("probabilities", "states", "final_state"):
                np.testing.assert_allclose(result[key], records[0]["models"][name][key])
                np.testing.assert_array_equal(
                    loaded_records[0]["models"][name][key], records[0]["models"][name][key]
                )
            np.testing.assert_allclose(
                model.update_state(np.array([0.2, -0.1]), 0, 1),
                self.models[name].update_state(np.array([0.2, -0.1]), 0, 1),
            )

    def test_training_restores_recorded_best_validation_model(self):
        model = self.models["GRU"]
        history = fit_gru(model, [self.block], [self.block], max_epochs=5, patience=2)
        self.assertEqual(history["epoch"], list(range(1, len(history["epoch"]) + 1)))
        self.assertEqual(len(history["train_nll"]), len(history["validation_nll"]))
        best_index = int(np.argmin(history["validation_nll"]))
        self.assertEqual(history["best_epoch"], best_index + 1)
        prediction = model.predict_block(self.block["actions"], self.block["rewards"])
        true_probabilities = prediction["probabilities"][np.arange(4), self.block["actions"]]
        actual_loss = -np.log(true_probabilities).mean()
        self.assertAlmostEqual(float(actual_loss), min(history["validation_nll"]), places=6)


if __name__ == "__main__":
    unittest.main()
