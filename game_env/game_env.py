import pathlib

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Box

from GP_predict import GP
from car import Car
from control.control import get_trajectory, filter_output_primal
from control.gp_controller import GPCar

data_dir = pathlib.Path(__file__).parent.parent.absolute() / "data"


class GameEnv(gymnasium.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # env params
        self.params = {
            "T": 150,  # simulation time
            "min_agents": 3,
            "max_agents": 10,
            "dist_threshold": 1.0,
            "coll_threshold": 4.9,
            "noise_a": 0.001,  # noise on acceleration input
        }

        # interval variables
        self.min_dist = np.inf
        self.N_a = None  # TODO
        self.agents = None  # TODO
        self.agents_ctrl = None  # TODO
        self.ego_controller = None  # TODO

        # obs and action spaces
        self.observation_space = None  # TODO
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))

        # rendering
        self.window_size = 1024
        self.size = 60.0  # size of the world
        self.ticks = 60

        self.render_mode = render_mode
        if self.render_mode is not None:
            pygame.init()
            pygame.display.set_caption("CBF Test")
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
        assert self.render_mode in [
            None,
            "human",
        ], "rgb_array rendering not implemented yet"

    def reset(self, seed=None, options=None):
        # seeding
        if seed is not None:
            np.random.seed(seed)

        # reset interval variables
        self.min_dist = np.inf
        self.steps = 0

        # initialize agents
        if options is not None and "n_agents" in options:
            min_agents = max_agents = options["n_agents"]
        else:
            min_agents, max_agents = (
                self.params["min_agents"],
                self.params["max_agents"],
            )
        self.N_a = np.random.randint(min_agents, max_agents + 1)

        # initialize starting and goal positions
        self.agents = []
        self.agents_ctrl = np.zeros((self.N_a, 2))

        for i in range(self.N_a):
            other_starting_positions = np.array(
                [self.agents[j].position for j in range(i)]
            )
            x0, y0 = self._sample_free_position(other_starting_positions, dist=8.0)

            if i == 0:
                agent = GPCar(x0, y0, N_a=self.N_a)
            else:
                agent = Car(x0, y0)

            self.agents.append(agent)

        for i in range(len(self.agents)):
            other_final_positions = np.array([self.agents[j].goal for j in range(i)])
            xf, yf = self._sample_free_position(other_final_positions, dist=8.0)
            self.agents[i].goal = np.array([xf, yf])

        self.agents[0].max_acceleration = 8.0

        # Set barrier for each agent
        horizon_set = [0, 0, 0, 0, 7, 8]
        for i in range(1, self.N_a):
            self.agents[i].Ds = horizon_set[np.random.randint(len(horizon_set))]

        obs = self._get_obs()
        info = {}
        return obs, info

    @staticmethod
    def _sample_free_position(
        occupied_positions: np.ndarray, dist: float
    ) -> np.ndarray:
        """
        Samples a position in the environment at random, and ensures
        that the position is at least a distance dist away from any
        occupied position.

        :param occupied_positions: A list of positions that are already occupied, shape (K, 2).
        :param dist: The minimum distance to any occupied position.
        :return: an array of shape (2,) representing the sampled position.
        """
        max_x = max_y = 60
        in_collision = True
        x, y = -1, -1
        while in_collision:
            in_collision = False
            x, y = max_x * np.random.rand(), max_y * np.random.rand()
            if occupied_positions.shape[0] > 0:
                dists = np.linalg.norm(occupied_positions - np.array([x, y]), axis=1)
                if np.any(dists < dist):
                    in_collision = True
        return np.array([x, y])

    def _get_obs(self):
        states = np.array([agent.state for agent in self.agents])
        return states

    def step(self, action):
        # gamma by denormalizing action from +-1 to 0-1
        gamma = (action + 1.0) / 2.0

        rel_state = np.zeros((self.N_a, 4))
        rel_state[0, :] = self.agents[0].state
        d_state = np.zeros((self.N_a, 4))

        # compute input for other agents
        for j in range(1, self.N_a):
            # Obtain (CBF) controller for other agent (if applicable)
            try:
                u2, x2_path, x2_0 = get_trajectory(self.agents[j])
                if self.agents[j].Ds > 0:
                    u2 = filter_output_primal(j, self.agents, x2_path)
            except ValueError as e:
                u2 = np.zeros(2)
                x2_0 = self.agents[j].state
            self.agents_ctrl[j] = u2
            # get agent's relative state
            rel_state[j, :] = x2_0 - self.agents[0].state
            # update min dist
            self.min_dist = min(self.min_dist, np.linalg.norm(rel_state[j, :2]))

        # compute input for our robot
        self.agents_ctrl[0], opt_info = self.agents[0].plan(
            state=rel_state, d_state=d_state, agents=self.agents, gamma=gamma
        )

        # step simulation
        noise_a = self.params["noise_a"]
        for j in range(self.N_a):
            xj = self.agents[j].state

            noisy_ctrl = self.agents_ctrl[j] + noise_a * (np.random.rand(2) - 0.5)
            self.agents[j].update(noisy_ctrl)

            p, v = self.agents[j].fh_err(xj)
            d_state[j, :] = self.agents[j].state - np.concatenate((p, v))

        # check termination conditions
        coll_threshold = self.params["coll_threshold"]
        dist_threshold = self.params["dist_threshold"]
        success, collision_flag = False, False
        for j in range(1, self.N_a):
            if (
                np.linalg.norm(self.agents[0].position - self.agents[j].position)
                < coll_threshold
            ):
                success = False
                collision_flag = True
        if (
            np.linalg.norm(self.agents[0].position - self.agents[0].goal)
            < dist_threshold
            and collision_flag
        ):
            success = False
        elif (
            np.linalg.norm(self.agents[0].position - self.agents[0].goal)
            < dist_threshold
            and not collision_flag
        ):
            success = True

        # compute reward
        self.steps += 1
        obs = self._get_obs()
        reward = -0.1
        info = {
            "min_dist": self.min_dist,
            "unfeasible": int(opt_info["opt_status"] < 0),
            "success": int(success),
            "collision": int(collision_flag),
        }
        done = self.steps >= self.params["T"] or success or collision_flag
        truncated = collision_flag

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_mode:
            ppu = self.window_size / self.size

            self.screen.fill((220, 220, 220))
            # Draw other agents
            radius = self.params["coll_threshold"] / 2 * ppu
            for j in range(1, self.N_a):
                pygame.draw.circle(
                    self.screen,
                    [200, 0, 0],
                    (self.agents[j].position * ppu).astype(int),
                    radius,
                )
            # Draw our goal
            agent1_goal = pygame.image.load(f"{data_dir}/star.png")
            rect = agent1_goal.get_rect()
            self.screen.blit(
                agent1_goal,
                self.agents[0].goal * ppu - (rect.width / 2, rect.height / 2),
            )
            # Draw our agent
            pygame.draw.circle(
                self.screen,
                [0, 0, 200],
                (self.agents[0].position * ppu).astype(int),
                radius,
            )
            pygame.display.flip()

    def close(self):
        pass
