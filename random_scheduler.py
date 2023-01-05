import numpy as np
import time
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv


def random_action_loop(jsp_instance: np.ndarray) -> None:

    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp_instance,
        scaling_divisor=40,
        perform_left_shift_if_possible=True,
        action_mode='task',  # alternative 'job'
        flat_observation_space=True,
        normalize_observation_space=True,
    )

    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        # noinspection PyTupleAssignmentBalance
        state, reward, done, info = env.step(action)
        # chose the visualisation you want to see using the show parameter
        # note: gantt_window and graph_window use a lot of resources
        # env.render(show=["gantt_console", "gantt_window", "graph_console", "graph_window"])
        # env.render()
        score += reward

    # console rendering
    env.render(show=["gantt_console", "graph_console"])
    # console + window rendering
    # env.render(wait=1_000)  # render window closes automatically after 1 seconds
    # env.render(wait=None) # render window closes when any button is pressed (when the render window is focused)


if __name__ == '__main__':
    jsp = np.array([
        [
            [0, 1, 2, 3],  # job 0 (engineer’s hammer)
            [0, 2, 1, 3],  # job 1  (Nine Man Morris)
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4],  # task durations of job 1
        ]

    ])
    random_action_loop(jsp_instance=jsp)
