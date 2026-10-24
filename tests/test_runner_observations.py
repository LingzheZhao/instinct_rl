# SPDX-License-Identifier: BSD-3-Clause
"""Tensor-only runner startup regression; no simulator required."""
import copy
import tempfile
import unittest

import torch

from instinct_rl.runners import OnPolicyRunner


class TensorEnv:
    num_envs, num_actions, num_rewards = 2, 1, 1

    def __init__(self):
        self.policy = torch.tensor([[3., 5.], [7., 9.]])
        self.critic = torch.tensor([[11.], [17.]])
        self.cache = {"observations": {"policy": self.policy, "critic": self.critic}}
        self.resets = self.reads = self.steps = 0

    def get_obs_format(self):
        return {"policy": {"state": (2,)}, "critic": {"state": (1,)}}

    def reset(self):
        self.resets += 1
        return self.policy, self.cache

    def get_observations(self):
        self.reads += 1
        return self.policy, self.cache

    def step(self, actions):
        self.steps += 1
        return self.policy, torch.ones(2, 1), torch.zeros(2, dtype=torch.bool), self.cache


def config():
    return dict(policy=dict(class_name="ActorCritic", actor_hidden_dims=[4], critic_hidden_dims=[4]),
                algorithm=dict(class_name="PPO", num_learning_epochs=1, num_mini_batches=1),
                normalizers={name: dict(class_name="EmpiricalNormalization") for name in ("policy", "critic")},
                num_steps_per_env=2, save_interval=10, log_interval=10)


class RunnerObservationsTest(unittest.TestCase):
    def test_no_normalizers_preserves_raw_actor_critic_and_environment_cache(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config()
            cfg["normalizers"] = {}
            env = TensorEnv()
            runner = OnPolicyRunner(env, cfg, log_dir=root, device="cpu")
            runner.current_learning_iteration = 1
            actor, critic = [], []
            runner.alg.actor_critic.actor.register_forward_pre_hook(lambda module, args: actor.append(args[0].clone()))
            runner.alg.actor_critic.critic.register_forward_pre_hook(lambda module, args: critic.append(args[0].clone()))
            runner.learn(1)
            torch.testing.assert_close(actor[0], env.policy)
            torch.testing.assert_close(critic[0], env.critic)
            torch.testing.assert_close(runner.alg.storage.observations[0], env.policy)
            torch.testing.assert_close(runner.alg.storage.critic_observations[0], env.critic)
            self.assertEqual((env.resets, env.reads, env.steps), (1, 1, 2))
            self.assertIs(env.cache["observations"]["policy"], env.policy)
            self.assertIs(env.cache["observations"]["critic"], env.critic)

    def test_restored_normalization_precedes_first_actor_and_critic(self):
        with tempfile.TemporaryDirectory() as root:
            source = OnPolicyRunner(TensorEnv(), config(), device="cpu")
            for name, normalizer in source.normalizers.items():
                normalizer(torch.arange(8 if name == "policy" else 4, dtype=torch.float32).reshape(4, -1))
            source.current_learning_iteration = 1  # no CUDA telemetry at the CPU update
            source.save(root + "/source.pt")
            env = TensorEnv()
            runner = OnPolicyRunner(env, config(), log_dir=root, device="cpu")
            runner.load(root + "/source.pt")
            expected = {name: copy.deepcopy(value)(getattr(env, name)) for name, value in runner.normalizers.items()}
            actor, critic = [], []
            runner.alg.actor_critic.actor.register_forward_pre_hook(lambda module, args: actor.append(args[0].clone()))
            runner.alg.actor_critic.critic.register_forward_pre_hook(lambda module, args: critic.append(args[0].clone()))
            runner.learn(1)
            torch.testing.assert_close(actor[0], expected["policy"])
            torch.testing.assert_close(critic[0], expected["critic"])
            self.assertEqual((env.resets, env.reads, env.steps), (1, 1, 2))
            for name, normalizer in runner.normalizers.items():
                self.assertEqual(int(normalizer.count), 10)  # four restored + three two-env observations
                self.assertIs(env.cache["observations"][name], getattr(env, name))


if __name__ == "__main__":
    unittest.main()
