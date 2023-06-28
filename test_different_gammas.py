import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from game_env.game_env import GameEnv


def collect_stats(env, gamma: float, n_agents: int, seed: int, n_episodes: int):
    np.random.seed(seed)

    ep_success = []
    ep_collision = []
    ep_lengths = []
    ep_min_dist = []
    ep_unfeasible = []
    for i in range(n_episodes):
        done = False
        obs, _ = env.reset(seed=None, options={"n_agents": n_agents})

        t0 = time.time()

        ep_length = 0
        ep_unfeasible_k = 0
        while not done:
            # action as normalized gamma, from 0-1 to -1 to 1
            action = 2 * gamma - 1.0
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            ep_length += 1
            ep_unfeasible_k += info["unfeasible"]

        ep_success.append(not truncated)
        ep_collision.append(truncated)
        ep_lengths.append(ep_length)
        ep_min_dist.append(info["min_dist"])
        ep_unfeasible.append(ep_unfeasible_k)

        min_dist = info["min_dist"]
        print(f"episode {i}: success: {not truncated}, collision: {truncated}, length: {ep_length}, " \
              f"min dist: {min_dist}, unfeasible k: {ep_unfeasible_k}, elapsed time: {time.time() - t0}")

    results = {
        "success": np.array(ep_success),
        "collision": np.array(ep_collision),
        "ep_length": np.array(ep_lengths),
        "ep_min_dist": np.array(ep_min_dist),
        "ep_unfeasible": np.array(ep_unfeasible)
    }

    return results


def main(args):
    gamma_grid = np.linspace(args.gamma_min, args.gamma_max, args.gamma_n)
    agents_grid = np.arange(args.agents_min, args.agents_max, args.agents_incr)
    params_grid = np.array(np.meshgrid(gamma_grid, agents_grid)).T.reshape(-1, 2)

    n_episodes = args.n_episodes
    seed = args.seed
    render_mode = args.render_mode
    outdir = args.outdir

    if outdir is not None:
        date_string = time.strftime("%Y%m%d-%H%M%S")
        outdir = pathlib.Path(outdir) / date_string
        outdir.mkdir(parents=True, exist_ok=True)

        # save config
        with open(outdir / "args.yaml", "w+") as f:
            yaml.dump(args, f)

    env = GameEnv(render_mode=render_mode)

    all_stats = {
        "gamma": [],
        "agents": [],
        "success_rate": [],
        "collision_rate": [],
        "succ_length_mu": [],
        "succ_length_std": [],
        "succ_min_dist_mu": [],
        "succ_min_dist_std": [],
        "ep_unfeasible_mu": [],
        "ep_unfeasible_std": []
    }
    for gamma, n_agents in params_grid:
        print(f"gamma: {gamma}, n_agents: {n_agents}")
        results = collect_stats(env, gamma, n_agents, seed, n_episodes)

        sr = np.mean(results["success"])
        cr = np.mean(results["collision"])
        unfeasibility = results["ep_unfeasible"]
        success_ids = np.where(results["success"])[0]
        if len(success_ids) > 0:
            succlengths = results["ep_length"][success_ids]
            succmindist = results["ep_min_dist"][success_ids]
        else:
            succlengths = [0]
            succmindist = [0]

        for k, v in zip(["gamma", "agents", "success_rate", "collision_rate", "succ_length_mu", "succ_length_std",
                         "succ_min_dist_mu", "succ_min_dist_std", "ep_unfeasible_mu", "ep_unfeasible_std"],
                        [gamma, n_agents, sr, cr, np.mean(succlengths), np.std(succlengths), np.mean(succmindist),
                         np.std(succmindist), np.mean(unfeasibility), np.std(unfeasibility)]):
            all_stats[k].append(v)

        for k, v in all_stats.items():
            print(f"{k}: {v[-1]}")
        print()

        # save results to file
        if outdir:
            filename = f"gamma_{gamma}_sr_{sr:.2f}_cr_{cr:.2f}.csv"
            df = pd.DataFrame(results)
            df.to_csv(outdir / filename, index=False)

    plotting(all_stats)

    if outdir:
        plt.savefig(outdir / "all_stats.png")
    else:
        plt.show()


def plotting(all_stats):
    # plot all stats
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    ax = axes[0, 0]
    ax.set_title("Success rate")
    ax.step(all_stats["gamma"], all_stats["success_rate"], where="post")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("gamma")
    ax.set_ylabel("Higher is better")

    ax = axes[0, 1]
    ax.set_title("Collision rate")
    ax.step(all_stats["gamma"], all_stats["collision_rate"], where="post")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("gamma")
    ax.set_ylabel("Lower is better")

    ax = axes[1, 0]
    ax.set_title("Min distance - Successfull episodes")
    ax.errorbar(all_stats["gamma"], all_stats["succ_min_dist_mu"], yerr=all_stats["succ_min_dist_std"], fmt='o')
    ax.set_ylim(-1, 30.0)
    ax.set_xlabel("gamma")
    ax.set_ylabel("Higher is better")

    ax = axes[1, 1]
    ax.set_title("Unfeasibility counts")
    ax.errorbar(all_stats["gamma"], all_stats["ep_unfeasible_mu"], yerr=all_stats["ep_unfeasible_std"], fmt='o')
    ax.set_ylim(-1, 151)
    ax.set_xlabel("gamma")
    ax.set_ylabel("Lower is better")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gamma_min", type=float, default=0.0)
    parser.add_argument("--gamma_max", type=float, default=1.0)
    parser.add_argument("--gamma_n", type=int, default=5)

    parser.add_argument("--agents_min", type=int, default=3)
    parser.add_argument("--agents_max", type=int, default=10)
    parser.add_argument("--agents_incr", type=int, default=3)

    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--outdir", type=pathlib.Path, default=None)

    args = parser.parse_args()
    main(args)
