import time
import unittest

import numpy as np

from game_GP import Game
from game_env.game_env import GameEnv


class GameEnvTest(unittest.TestCase):
    def test_reset_implementation(self):
        seed = 42

        gym_env = GameEnv()
        original_game = Game()
        original_game.T = 0  # to test reset only

        obs, _ = gym_env.reset(seed=seed)
        data, _, _, _, _, _ = original_game.run(seed=seed)

        self.assertTrue(np.isclose(obs, data[0]).all())

    def test_step_implementation(self):
        seed = 42

        gym_env = GameEnv(render_mode="human")
        original_game = Game()

        done = False
        obs, _ = gym_env.reset(seed=seed)

        i = 0
        while not done:
            # self.assertTrue(np.isclose(obs, data[i]).all(), f"step: {i} \n obs: \n{obs} \n expected: \n{data[i]}")
            obs, reward, done, _, info = gym_env.step(None)
            gym_env.render()
            i += 1

        data, data_u, success, collision_flag, i, min_dist = original_game.run(
            seed=seed
        )

    def test_reproduce_results(self):
        seed = 42
        np.random.seed(seed)

        gym_env = GameEnv(render_mode=None)

        n_episodes = 100
        n_success = 0
        n_collisions = 0
        ep_lengths_success = []
        ep_min_dist_success = []
        for i in range(n_episodes):
            done = False
            obs, _ = gym_env.reset(seed=None)

            t0 = time.time()

            ep_length = 0
            while not done:
                obs, reward, done, truncated, info = gym_env.step(None)
                gym_env.render()
                ep_length += 1

            if truncated:
                n_collisions += 1
            else:
                n_success += 1
                ep_lengths_success.append(ep_length)
                ep_min_dist_success.append(info["min_dist"])

            print(
                f"episode {i}: success: {not truncated}, collision: {truncated}, length: {ep_length}, "
                f"min dist: {info['min_dist']}, elapsed time: {time.time() - t0}"
            )

        success_rate = n_success / n_episodes
        collision_rate = n_collisions / n_episodes
        mean_success_ep_length = np.mean(ep_lengths_success)
        mean_success_ep_min_dist = np.mean(ep_min_dist_success)

        print("success rate: ", success_rate)
        print("collision rate: ", collision_rate)
        print("mean success episode length: ", mean_success_ep_length)
        print("mean success episode min dist: ", mean_success_ep_min_dist)

        # according to the paper (Table 1):
        #   we expect a crash rate of 1.5 % over 1000 episodes
        #   and an average min dist of 7.4 +- 2.3
        # we check that we are in the same ballpark
        self.assertTrue(success_rate > 0.98)
        self.assertTrue(collision_rate < 0.02)
        self.assertTrue(mean_success_ep_min_dist < 10)
