import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from game_env.game_env import GameEnv


def main(args):
    # parse args
    n_episodes = args.n_episodes
    seed = args.seed
    render_mode = args.render_mode
    indir = pathlib.Path(args.indir) if args.indir else None

    gamma_grid = np.linspace(args.gamma_min, args.gamma_max, args.gamma_n)
    agents_grid = np.arange(args.agents_min, args.agents_max, args.agents_incr)
    params_grid = np.array(np.meshgrid(gamma_grid, agents_grid)).T.reshape(-1, 2)

    if indir:
        all_stats = load_stats_from_dir(indir=indir)
        outdir = indir
    else:
        # prepare outdir
        subdir = f"Nep{n_episodes}_{time.strftime('%Y%m%d-%H%M%S')}"
        outdir = pathlib.Path(args.outdir) / subdir if args.outdir else None
        all_stats = collect_stats_grid_search(params_grid=params_grid, n_episodes=n_episodes,
                                              seed=seed, render_mode=render_mode, outdir=outdir)

    plotting(all_stats)

    if outdir:
        plt.savefig(outdir / f"all_stats_{int(time.time())}.png")
    else:
        plt.show()


def load_stats_from_dir(indir: pathlib.Path, file_regex: str = "*.csv") -> dict:
    """
    Load stats from a directory containing previous experiments results in csv format.

    :param indir: path to the directory containing the results

    :return: a dictionary containing the stats
    """
    all_stats = {}

    for f in indir.glob(file_regex):
        # read csv as dict
        results = pd.read_csv(f).to_dict(orient="list")
        # parse run
        run_id = f.stem
        tokens = run_id.split("_")
        assert len(tokens) == 2, f"Invalid run id: {run_id}"
        gamma = float(tokens[0].replace("gamma", ""))
        n_agents = int(tokens[1].replace("agents", ""))
        # parse stats
        sr, cr, succlengths, succmindist, unfeasibility = process_results(results)

        for k, v in zip(["gamma", "agents", "success_rate", "collision_rate", "succ_length_mu", "succ_length_std",
                         "succ_min_dist_mu", "succ_min_dist_std", "ep_unfeasible_mu", "ep_unfeasible_std"],
                        [gamma, n_agents, sr, cr, np.mean(succlengths), np.std(succlengths), np.mean(succmindist),
                         np.std(succmindist), np.mean(unfeasibility), np.std(unfeasibility)]):
            if k not in all_stats:
                all_stats[k] = []
            all_stats[k].append(v)

    return all_stats


def collect_stats_grid_search(params_grid, seed, n_episodes, outdir, render_mode):
    """
    Collect stats for `n_episodes` for each entry in the given `params_grid`.

    :param params_grid: list of param tuples (gamma, n_agents)
    :param seed: int or None
    :param n_episodes: number of episodes to run for each param tuple
    :param outdir: path to the output directory, or None
    :param render_mode: how to render the simulation, or None
    :return: a dictionary containing the stats
    """
    # create env
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

    # create output directory
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

        # save config
        with open(outdir / "args.yaml", "w+") as f:
            yaml.dump(vars(args), f)

    # simulation
    for gamma, n_agents in params_grid:
        results = collect_stats(env, gamma, n_agents, seed, n_episodes, verbose=True)
        sr, cr, succlengths, succmindist, unfeasibility = process_results(results)

        for k, v in zip(["gamma", "agents", "success_rate", "collision_rate", "succ_length_mu", "succ_length_std",
                         "succ_min_dist_mu", "succ_min_dist_std", "ep_unfeasible_mu", "ep_unfeasible_std"],
                        [gamma, n_agents, sr, cr, np.mean(succlengths), np.std(succlengths), np.mean(succmindist),
                         np.std(succmindist), np.mean(unfeasibility), np.std(unfeasibility)]):
            all_stats[k].append(v)

        print("[results]")
        for k, v in all_stats.items():
            print(f"\t{k}: {v[-1]}")
        print()

        # save results to file
        if outdir:
            filename = f"gamma{gamma:.2f}_agents{n_agents:.0f}.csv"
            df = pd.DataFrame(results)
            df.to_csv(outdir / filename, index=False)

    return all_stats


def process_results(results: dict) -> tuple:
    sr = np.mean(results["success"])
    cr = np.mean(results["collision"])
    unfeasibility = results["ep_unfeasible"]

    success_ids = np.where(results["success"])[0]
    no_coll_ids = np.where(np.logical_not(results["collision"]))[0]
    if len(success_ids) > 0:
        succlengths = np.array(results["ep_length"])[success_ids]
    else:
        succlengths = [0]

    if len(no_coll_ids) > 0:
        succmindist = np.array(results["ep_min_dist"])[no_coll_ids]
    else:
        succmindist = [0]

    return sr, cr, succlengths, succmindist, unfeasibility


def collect_stats(env, gamma: float, n_agents: int, seed: int, n_episodes: int, verbose: bool = False) -> dict:
    """
    Simulate n_episodes with a given `gamma` and `n_agents` and return the results.

    :param env: the simulation environment
    :param gamma: the cbf coefficient, between 0 and 1
    :param n_agents: the number of agents in the simulation
    :param seed: initial seed for the random number generator, or None
    :param n_episodes: the number of episodes to simulate
    :param verbose: whether to print intermediate simulations and results
    :return:
    """
    assert 0 <= gamma <= 1, "gamma must be between 0 and 1"
    assert n_agents > 0, "n_agents must be greater than 0"
    assert seed is None or seed >= 0, "seed must be greater than 0 or None"
    assert n_episodes > 0, "n_episodes must be greater than 0"

    np.random.seed(seed)

    if verbose:
        print(f"[info] gamma: {gamma}, n_agents: {n_agents}, seed: {seed}")

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

        ep_success.append(info["success"])
        ep_collision.append(info["collision"])
        ep_lengths.append(ep_length)
        ep_min_dist.append(info["min_dist"])
        ep_unfeasible.append(ep_unfeasible_k)

        if verbose:
            min_dist = info["min_dist"]
            print(f"\tepisode {i}: success: {info['success']}, collision: {info['collision']}, "
                  f"length: {ep_length}, min dist: {min_dist:.2f}, unfeasible k: {ep_unfeasible_k}, "
                  f"elapsed time: {time.time() - t0:.2f} sec")

    results = {
        "success": np.array(ep_success),
        "collision": np.array(ep_collision),
        "ep_length": np.array(ep_lengths),
        "ep_min_dist": np.array(ep_min_dist),
        "ep_unfeasible": np.array(ep_unfeasible),
    }

    return results


def plotting(all_stats):
    # plot all stats
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    metrics = ["success_rate", "collision_rate", "succ_min_dist_mu", "ep_unfeasible_mu"]
    titles = ["Success rate", "Collision rate", "Minimum distance - Safe episodes", "Unfeasible rate"]
    ylabels = ["Higher is better", "Lower is better", "Higher is better", "Lower is better"]
    min_max_y = [(0, 1), (0, 1), (0.0, 30.0), (0, 150.0)]
    margin_percent = 0.1

    all_gammas = set(all_stats["gamma"])
    all_agents = set(all_stats["agents"])

    # create list of different linestyles and colors for each agent
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]

    for i, n_agents in enumerate(all_agents):
        stat_ids = np.where(np.array(all_stats["agents"]) == n_agents)[0]
        marker = markers[i % len(markers)]

        for j, (metric, label) in enumerate(zip(metrics, titles)):
            r, c = j // 2, j % 2
            gammas = [all_stats["gamma"][i] for i in stat_ids]
            values = [all_stats[metric][i] for i in stat_ids]

            # sort by gamma
            gammas, values = zip(*sorted(zip(gammas, values)))

            y_label = ylabels[j]
            min_y, max_y = min_max_y[j]
            y_range = max_y - min_y

            ax = axes[r, c]
            ax.set_title(f"{label}")
            ax.step(gammas, values, where="post", marker=marker,
                    label=f"NAgents={n_agents:.0f}")
            ax.set_ylim(min_y - margin_percent * y_range, max_y + margin_percent * y_range)
            ax.set_xlabel("gamma")
            ax.set_ylabel(y_label)

    ax.legend()

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

    parser.add_argument("--indir", type=str, default=None, help="Directory from which to load results")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save results")

    args = parser.parse_args()

    t0 = time.time()

    main(args)

    print(f"[done] elapsed time: {time.time() - t0}")
