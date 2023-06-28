import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from game_env.game_env import GameEnv


def collect_stats(env, gamma: float, seed: int, n_episodes: int):
    np.random.seed(seed)

    ep_success = []
    ep_collision = []
    ep_lengths = []
    ep_min_dist = []
    ep_unfeasible = []
    for i in range(n_episodes):
        done = False
        obs, _ = env.reset(seed=None)

        t0 = time.time()

        ep_length = 0
        while not done:
            # action as normalized gamma, from 0-1 to -1 to 1
            action = 2 * gamma - 1.0
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            ep_length += 1

            ep_success.append(not truncated)
            ep_collision.append(truncated)
            ep_lengths.append(ep_length)
            ep_min_dist.append(info["min_dist"])
            ep_unfeasible.append(info["unfeasible"])

        min_dist = info["min_dist"]
        unfeasible_k = np.sum(ep_unfeasible)
        print(f"episode {i}: success: {not truncated}, collision: {truncated}, length: {ep_length}, " \
              f"min dist: {min_dist}, unfeasible k: {unfeasible_k}, elapsed time: {time.time() - t0}")

    results = {
        "success": ep_success,
        "collision": ep_collision,
        "ep_length": ep_lengths,
        "ep_min_dist": ep_min_dist,
        "ep_unfeasible": ep_unfeasible,
    }

    return results

def main(args):
    gamma_grid = np.linspace(args.gamma_min, args.gamma_max, args.gamma_n)
    n_episodes = args.n_episodes
    seed = args.seed
    render_mode = args.render_mode
    outdir = args.outdir

    env = GameEnv(render_mode=render_mode)

    for gamma in gamma_grid:
        print(f"gamma: {gamma}")
        results = collect_stats(env, gamma, seed, n_episodes)

        print(f"success rate: {np.mean(results['success'])}")
        print(f"collision rate: {np.mean(results['collision'])}")
        print(f"mean ep length: {np.mean(results['ep_length'])}")
        print(f"mean ep min dist: {np.mean(results['ep_min_dist'])}")
        print(f"mean unfeasible count: {np.mean(results['ep_unfeasible'])}")

        print()

        # save results to file
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(results)
            df.to_csv(outdir / f"gamma_{gamma}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gamma_min", type=float, default=0.0)
    parser.add_argument("--gamma_max", type=float, default=1.0)
    parser.add_argument("--gamma_n", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--outdir", type=pathlib.Path, default=None)

    args = parser.parse_args()
    main(args)
