import inquirer
import numpy as np
import time

from jsp_env.src.graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from jsp_env.src.graph_jsp_env.disjunctive_graph_logger import log

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

    env = DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        scaling_divisor=40,
        perform_left_shift_if_possible=True,
        action_mode='task',
    )

    done = False
    log.info("each task/node corresponds to an action")

    while not done:
        env.render(
            show=["gantt_console", "gantt_window", "graph_console", "graph_window"],
            # ,stack='vertically'
        )
        questions = [
            inquirer.List(
                "task",
                message="Which task should be scheduled next?",
                choices=[
                    (f"Task {task_id}", task_id)
                    for task_id, bol in enumerate(env.valid_action_mask(), start=1)
                    if bol
                ],
            ),
        ]
        action = inquirer.prompt(questions)["task"] - 1  # note task are index 1 in the viz, but index 0 in action space
        print(action)
        n_state, reward, done, info = env.step(action)
        print(n_state)

        # note: gantt_window and graph_window use a lot of resources

    log.info(f"the JSP is completely scheduled.")
    log.info(f"makespan: {info['makespan']}")
    log.info("press any key to close the window (while the window is focused).")
    # env.render(wait=None)  # wait for keyboard input before closing the render window
    env.render(
        wait=None,
        show=["graph_window", "gantt_window"],
        # stack='vertically'
    )