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
    get_readout,
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
        expected_maxima = []
        for condition in field["conditions"]:
            if condition["action"] == 0:
                expected_dx = 0.25 * (condition["reward"] - field["x"])
                np.testing.assert_allclose(condition["dx"], expected_dx)
                np.testing.assert_array_equal(condition["dy"], 0)
                chosen_values = field["x"]
            else:
                expected_dy = 0.25 * (condition["reward"] - field["y"])
                np.testing.assert_array_equal(condition["dx"], 0)
                np.testing.assert_allclose(condition["dy"], expected_dy)
                chosen_values = field["y"]
            # RW 只移动一个坐标，因此欧氏距离就是 alpha × 绝对预测误差。
            expected_speed = model.alpha * np.abs(condition["reward"] - chosen_values)
            np.testing.assert_allclose(condition["speed"], expected_speed)
            expected_maxima.append(float(expected_speed.max()))
            self.assertEqual(condition["observed_states"].shape, (0, 2))
            self.assertEqual(condition["observed_changes"].shape, (0, 2))
        # 四个条件共用最大更新幅度，不能在各面板内分别归一化颜色。
        self.assertAlmostEqual(field["speed_max"], max(expected_maxima))
        self.assertFalse(field["has_observed_events"])
        np.testing.assert_array_equal(field["readout_vector"], [3, -3])
        self.assertEqual(field["readout_bias"], 0)
        expected_probability = 1 / (1 + np.exp(3 * (field["y"] - field["x"])))
        np.testing.assert_allclose(field["probability"], expected_probability)

    def test_readout_direction_and_boundary_match_choice_probability(self):
        # 特意设置非对角权重及非零偏置，避免只检验特殊的 h1-h2 读出。
        gru_model = GRUModel().double()
        with torch.no_grad():
            gru_model.linear.weight.copy_(
                torch.tensor([[1.7, -0.3], [-0.4, 0.9]], dtype=torch.float64)
            )
            gru_model.linear.bias.copy_(torch.tensor([0.35, -0.15], dtype=torch.float64))
        examples = [
            ("RW", RWModel(iTemp=2.5), np.array([2.5, -2.5]), 0.0),
            ("GRU", gru_model, np.array([2.1, -1.2]), 0.5),
        ]
        for name, model, expected_weights, expected_bias in examples:
            with self.subTest(model=name):
                weights, bias = get_readout(model)
                np.testing.assert_allclose(weights, expected_weights)
                self.assertAlmostEqual(bias, expected_bias)
                for state in (np.array([0.2, 0.7]), np.array([0.8, -0.3])):
                    expected_probability = 1 / (1 + np.exp(-(weights @ state + bias)))
                    self.assertAlmostEqual(
                        model.choice_probabilities(state)[0], expected_probability, places=12
                    )

                # 求边界上离原点最近的点，并沿边界方向再取两个不同的点。
                # 边界上的概率都应为 0.5，沿读出方向移动则偏向动作 0。
                boundary_center = -bias * weights / (weights @ weights)
                readout_direction = weights / np.linalg.norm(weights)
                boundary_direction = np.array([-weights[1], weights[0]])
                boundary_direction /= np.linalg.norm(boundary_direction)
                for offset in (-0.3, 0.0, 0.3):
                    boundary_state = boundary_center + offset * boundary_direction
                    self.assertAlmostEqual(
                        model.choice_probabilities(boundary_state)[0], 0.5, places=12
                    )
                    self.assertGreater(
                        model.choice_probabilities(boundary_state + 0.2 * readout_direction)[0],
                        0.5,
                    )
                    self.assertLess(
                        model.choice_probabilities(boundary_state - 0.2 * readout_direction)[0],
                        0.5,
                    )

    def test_observed_arrows_follow_the_corresponding_events(self):
        # 四种条件交错出现，并重复部分条件，检验筛选时保留正确的时间顺序。
        actions = np.array([0, 1, 0, 0, 1, 1])
        rewards = np.array([1, 0, 0, 1, 1, 0])
        expected_indices = {(0, 0): [2], (0, 1): [0, 3], (1, 0): [1, 5], (1, 1): [4]}
        for name, model in self.models.items():
            with self.subTest(model=name):
                prediction = model.predict_block(actions, rewards)
                states = np.vstack([prediction["states"], prediction["final_state"]])
                field = compute_vector_field(
                    model, states, grid_size=3, actions=actions, rewards=rewards
                )
                self.assertTrue(field["has_observed_events"])
                observed_count = 0
                for condition in field["conditions"]:
                    indices = expected_indices[(condition["action"], condition["reward"])]
                    np.testing.assert_allclose(condition["observed_states"], states[indices])
                    observed_count += len(condition["observed_states"])
                    for arrow_index, trial_index in enumerate(indices):
                        arrow_start = condition["observed_states"][arrow_index]
                        arrow_end = arrow_start + condition["observed_changes"][arrow_index]
                        # 箭头终点既要接到真实序列的下一状态，也要等于同一事件的单步更新。
                        np.testing.assert_allclose(arrow_end, states[trial_index + 1], atol=1e-7)
                        np.testing.assert_allclose(
                            arrow_end,
                            model.update_state(arrow_start, actions[trial_index], rewards[trial_index]),
                            atol=1e-7,
                        )
                self.assertEqual(observed_count, len(actions))

    def test_vector_field_rejects_misaligned_events_and_non_2d_models(self):
        states = np.zeros((5, 2))
        actions = self.block["actions"]
        rewards = self.block["rewards"]
        invalid_events = [
            {"actions": actions},
            {"rewards": rewards},
            {"actions": actions[:-1], "rewards": rewards},
            {"actions": actions, "rewards": rewards[:-1]},
            {"actions": actions[:, None], "rewards": rewards},
        ]
        for events in invalid_events:
            with self.subTest(events=events), self.assertRaises(ValueError):
                compute_vector_field(self.models["RW"], states, grid_size=3, **events)
        for hidden_dim in (1, 3):
            with self.subTest(hidden_dim=hidden_dim), self.assertRaises(ValueError):
                compute_vector_field(GRUModel(hidden_dim), np.zeros((5, hidden_dim)), 3)
        with self.assertRaises(ValueError):
            compute_vector_field(GRUModel(hidden_dim=3), states, grid_size=3)

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
