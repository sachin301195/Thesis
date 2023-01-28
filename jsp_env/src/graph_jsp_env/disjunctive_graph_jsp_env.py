import gym
import numpy as np
import networkx as nx
import pandas as pd
import json

import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import List, Union

from jsp_env.src.graph_jsp_env.disjunctive_graph_jsp_visualizer import DisjunctiveGraphJspVisualizer

# from graph_jsp_env.disjunctive_graph_jsp_visualizer import DisjunctiveGraphJspVisualizer
from jsp_env.src.graph_jsp_env.disjunctive_graph_logger import log


class DisjunctiveGraphJspEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self, env_config: dict):
        # Note: None-fields will be populated in the 'load_instance' method
        self.size = None
        self.n_jobs = None
        self.n_machines = None
        self.total_tasks_without_dummies = None
        self.total_tasks = None
        self.src_task = None
        self.sink_task = None
        self.longest_processing_time = None
        self.observation_space_shape = None
        self.scaling_divisor = None
        self.horizon = None
        self.machine_colors = None
        self.G = None
        self.machine_routes = None
        self.task_to_duration_mapping = None
        self.task_to_machine_mapping = None
        self.start = None
        self.task_status = None
        self.info = None
        self.not_valid = None
        self.sum_op = None
        self.jsp = env_config["jsp"]

        self.env_config = env_config
        self.reward_version = env_config["reward_version"]
        self.scale_reward = env_config["scale_reward"]
        self.time_length = 0

        # observation settings
        self.normalize_observation_space = env_config["normalize_observation_space"]
        self.flat_observation_space = env_config["flat_observation_space"]
        self.dtype = env_config["dtype"]

        # action setting
        self.perform_left_shift_if_possible = env_config["perform_left_shift_if_possible"]
        if env_config["action_mode"] not in ['task', 'job']:
            raise ValueError(f"only 'task' and 'job' are valid arguments for 'action_mode'. {env_config['action_mode']}"
                             f" is not.")
        self.action_mode = env_config["action_mode"]

        if env_config['env_transform'] not in [None, 'mask']:
            raise ValueError(f"only `None` and 'mask' are valid arguments for 'env_transform'. "
                             f"{env_config['env_transform']} is not.")
        self.env_transform = env_config['env_transform']

        # rendering settings
        self.c_map = "rainbow"
        self.default_visualisations = ["gantt_console", "gantt_window", "graph_console", "graph_window"]
        visualizer_kwargs = {}
        self.visualizer = DisjunctiveGraphJspVisualizer(**visualizer_kwargs)

        # values for dummy tasks nedded for the graph structure
        self.dummy_task_machine = -1
        self.dummy_task_job = -1
        self.dummy_task_color = "tab:gray"

        self.verbose = env_config['verbose']

        jps_instance, self.opt_value = self.jsp

        if self.scale_reward:
            if env_config["scaling_divisor"] == 0:
                scaling_divisor = self.opt_value
            else:
                scaling_divisor = env_config["scaling_divisor"]
        else:
            scaling_divisor = None

        if jps_instance is not None:
            self.load_instance(jsp_instance=jps_instance, scaling_divisor=scaling_divisor)

    def load_instance(self, jsp_instance: np.ndarray, *, scaling_divisor: float = None) -> None:
        _, n_jobs, n_machines = jsp_instance.shape
        self.size = (n_jobs, n_machines)
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.total_tasks_without_dummies = n_jobs * n_machines
        self.total_tasks = n_jobs * n_machines + 2  # 2 dummy tasks: start, finish
        self.src_task = 0
        self.sink_task = self.total_tasks - 1

        self.longest_processing_time = jsp_instance[1].max()

        # self.task_to_machine_mapping = np.zeros(shape=(self.total_tasks_without_dummies, 1), dtype=int)
        # self.task_to_duration_mapping = np.zeros(shape=(self.total_tasks_without_dummies, 1), dtype=self.dtype)

        if self.action_mode == 'task':
            self.action_space = gym.spaces.Discrete(self.total_tasks_without_dummies)
        else:  # action mode 'job'
            self.action_space = gym.spaces.Discrete(self.n_jobs)

        self.observation_space_shape = (self.total_tasks_without_dummies,
                                        self.total_tasks_without_dummies + 5)

        if self.flat_observation_space:
            a, b = self.observation_space_shape
            self.observation_space_shape = (a * b,)

        if self.env_transform is None:
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0 if self.normalize_observation_space else jsp_instance.max(),
                shape=self.observation_space_shape,
                dtype=self.dtype
            )
        elif self.env_transform == 'mask':
            self.observation_space = gym.spaces.Dict({
                "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int32),
                "observations": gym.spaces.Box(
                    low=0.0,
                    high=1.0 if self.normalize_observation_space else jsp_instance.max(),
                    shape=self.observation_space_shape,
                    dtype=self.dtype)
            })
        else:
            raise NotImplementedError(f"'{self.env_transform}' is not supported.")

        if self.scale_reward and scaling_divisor:
            self.scaling_divisor = scaling_divisor
        elif self.scale_reward and not scaling_divisor:
            if self.verbose > 0:
                log.warning(
                    "defaulting scaling_divisor to an naive lower bound. You might consider setting 'scaling_divisor'"
                    " to a lower bound calculated by a suitable heuristic or Google Or tools for better performance.")
            self.scaling_divisor = np.sum(jsp_instance[1], axis=1).min()  # shortest job

        # naive upper bound
        # may be useful for wrappers
        # the term 'horizon' is inspired by the Google OR Tools terminology
        self.horizon = jsp_instance[1].max() * n_machines * n_jobs

        # generate colors for machines
        c_map = plt.cm.get_cmap(self.c_map)  # select the desired cmap
        arr = np.linspace(0, 1, n_machines, dtype=self.dtype)  # create a list with numbers from 0 to 1 with n items
        self.machine_colors = {m_id: c_map(val) for m_id, val in enumerate(arr)}

        # jsp representation as directed graph
        self.G = nx.DiGraph()
        # 'routes' of the machines. indicates in which order a machine processes tasks
        self.machine_routes = {m_id: np.array([], dtype=int) for m_id in range(n_machines)}

        #
        # setting up the graph
        #

        # src node
        self.G.add_node(
            self.src_task,
            pos=(-2, int(-n_jobs * 0.5)),
            duration=0,
            machine=self.dummy_task_machine,
            scheduled=True,
            color=self.dummy_task_color,
            job=self.dummy_task_job,
            start_time=0,
            finish_time=0
        )

        # add nodes in grid format
        task_id = 0
        machine_order = jsp_instance[0]
        processing_times = jsp_instance[1]
        for i in range(n_jobs):
            for j in range(n_machines):
                task_id += 1  # start from task id 1, 0 is dummy starting task
                m_id = machine_order[i, j]  # machine id
                dur = processing_times[i, j]  # duration of the task

                self.G.add_node(
                    task_id,
                    pos=(j, -i),
                    color=self.machine_colors[m_id],
                    duration=dur,
                    scheduled=False,
                    machine=m_id,
                    job=i,
                    start_time=None,
                    finish_time=None
                )

                if j == 0:  # first task in a job
                    self.G.add_edge(
                        self.src_task, task_id,
                        job_edge=True,
                        weight=self.G.nodes[self.src_task]['duration'],
                        nweight=-self.G.nodes[self.src_task]['duration']
                    )
                elif j == n_machines - 1:  # last task of a job
                    self.G.add_edge(
                        task_id - 1, task_id,
                        job_edge=True,
                        weight=self.G.nodes[task_id - 1]['duration'],
                        nweight=-self.G.nodes[task_id - 1]['duration']
                    )
                else:
                    self.G.add_edge(
                        task_id - 1, task_id,
                        job_edge=True,
                        weight=self.G.nodes[task_id - 1]['duration'],
                        nweight=-self.G.nodes[task_id - 1]['duration']
                    )
        # add sink task at the end to avoid permutation in the adj matrix.
        # the rows and cols correspond to the order the nodes were added not the index/value of the node.
        self.G.add_node(
            self.sink_task,
            pos=(n_machines + 1, int(-n_jobs * 0.5)),
            color=self.dummy_task_color,
            duration=0,
            machine=self.dummy_task_machine,
            job=self.dummy_task_job,
            scheduled=True,
            start_time=None,
            finish_time=None
        )
        # add edges from last tasks in job to sink
        for task_id in range(n_machines, self.total_tasks, n_machines):
            self.G.add_edge(
                task_id, self.sink_task,
                job_edge=True,
                weight=self.G.nodes[task_id]['duration']
            )

    def reset(self):
        # remove machine edges/routes
        self.start = True
        self.not_valid = False
        self.sum_op = 1
        self.info = {"finish_time": -1, "makespan": 0}
        if self.scale_reward:
            if self.env_config["scaling_divisor"] == 0:
                scaling_divisor = self.opt_value
            else:
                scaling_divisor = self.env_config["scaling_divisor"]
        else:
            scaling_divisor = None

        # jps_instance, self.opt_value = self.jsp
        #
        # if jps_instance is not None:
        #     self.load_instance(jsp_instance=jps_instance, scaling_divisor=scaling_divisor)
        log.info(f"Running the instance: \n {self.jsp[0]} \n with Optimal value: {self.opt_value}")

        machine_edges = [(from_, to_) for from_, to_, data_dict in self.G.edges(data=True) if not data_dict["job_edge"]]
        self.G.remove_edges_from(machine_edges)

        # reset machine routes dict
        self.machine_routes = {m_id: np.array([]) for m_id in range(self.n_machines)}

        # remove scheduled flags, reset start_time and finish_time
        for i in range(1, self.total_tasks_without_dummies + 1):
            node = self.G.nodes[i]
            node["scheduled"] = False
            node["start_time"] = None,
            node["finish_time"] = None

        return self._state_array(-1)

    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        self.start = False
        self.info = {
            'action': action
        }

        # add arc in graph
        if self.action_mode == 'task':
            task_id = action + 1

            if self.verbose > 1:
                log.info(f"handling action={action} (Task {task_id})")
            self.info = {
                "finish_time": -1,
                **self._schedule_task(task_id=task_id)
            }
        else:  # case for self.action_mode == 'job'
            task_mask = self.valid_action_mask(action_mode='task')
            job_mask = np.array_split(task_mask, self.n_jobs)[action]

            if True not in job_mask:
                if self.verbose > 0:
                    log.info(f"job {action} is already completely scheduled. Ignoring it.")
                self.info["valid_action"] = False
            else:
                task_id = 1 + action * self.n_machines + np.argmax(job_mask)
                if self.verbose > 1:
                    log.info(f"handling job={action} (Task {task_id})")
                self.info = {
                    "finish_time": -1,
                    **self._schedule_task(task_id=task_id)
                }

        # check if done
        min_length = min([len(route) for m_id, route in self.machine_routes.items()])
        done = min_length == self.n_jobs

        if self.info["finish_time"] == -1:
            self.not_valid = True
        else:
            self.not_valid = False
            self.time_length = max(self.time_length, self.info["finish_time"])
            self.sum_op += self.info["finish_time"] - self.info["start_time"]

        if done:
            try:
                # by construction a cycle should never happen
                # add cycle check just to be sure
                cycle = nx.find_cycle(self.G, source=self.src_task)
                log.critical(f"CYCLE DETECTED cycle: {cycle}")
                raise RuntimeError(f"CYCLE DETECTED cycle: {cycle}")
            except nx.exception.NetworkXNoCycle:
                pass

            makespan = nx.dag_longest_path_length(self.G)
            reward = - makespan / self.scaling_divisor if self.scale_reward else - makespan

            self.info["makespan"] = makespan
            self.info["optimal_value"] = self.opt_value
            self.info["gantt_df"] = self.network_as_dataframe()
            if self.verbose > 0:
                log.info(f"makespan: {makespan}, return: {reward:.2f}")
                log.info(f"optimal value: {self.opt_value}")

        reward = self._calculate_reward(done)

        self.info["scaling_divisor"] = self.scaling_divisor

        state = self._state_array(action)
        return state, reward, done, self.info

    def _calculate_reward(self, done):
        if self.reward_version == 'B':
            reward = - (self.time_length / self.scaling_divisor if self.scale_reward else - self.time_length) * \
                     (not self.not_valid) - 5 * self.not_valid
        elif self.reward_version == "C":
            try:
                reward = -(((self.time_length * self.n_machines)/self.sum_op) * (not self.not_valid))-5 * self.not_valid
            except ZeroDivisionError:
                pass
        elif self.reward_version == "D":
            reward = (self.time_length - self.info["finish_time"]) * (not self.not_valid) - 5 * self.not_valid
        else:
            reward = - (self.info["makespan"] / self.scaling_divisor if self.scale_reward else self.info["makespan"]) \
                     * done - 5 * self.not_valid

        return reward

    def render(self, mode="human", show: List[str] = None, **render_kwargs) -> Union[None, np.ndarray]:
        df = None
        colors = None

        if mode not in ["human", "rgb_array", "console"]:
            raise ValueError(f"mode '{mode}' is not defined. allowed modes are: 'human' and 'rgb_array'.")

        if show is None:
            if mode == "rgb_array":
                show = [s for s in self.default_visualisations if "window" in s]
            elif mode == "console":
                show = [s for s in self.default_visualisations if "console" in s]
            else:
                show = self.default_visualisations

        if "gantt_console" in show or "gantt_window" in show:
            df = self.network_as_dataframe()
            colors = {f"Machine {m_id}": (r, g, b) for m_id, (r, g, b, a) in self.machine_colors.items()}

        if "graph_console" in show:
            self.visualizer.graph_console(self.G, shape=self.size, colors=colors)
        if "gantt_console" in show:
            self.visualizer.gantt_chart_console(df=df, colors=colors)

        if "graph_window" in show:
            if "gantt_window" in show:
                if mode == "human":
                    self.visualizer.render_graph_and_gant_in_window(G=self.G, df=df, colors=colors, **render_kwargs)
                elif mode == "rgb_array":
                    return self.visualizer.gantt_and_graph_vis_as_rgb_array(G=self.G, df=df, colors=colors)
            else:
                if mode == "human":
                    self.visualizer.render_graph_in_window(G=self.G, **render_kwargs)
                elif mode == "rgb_array":
                    return self.visualizer.graph_rgb_array(G=self.G)

        elif "gantt_window" in show:
            if mode == "human":
                self.visualizer.render_gantt_in_window(df=df, colors=colors, **render_kwargs)
            elif mode == "rgb_array":
                return self.visualizer.gantt_chart_rgb_array(df=df, colors=colors)

    def _schedule_task(self, task_id: int) -> dict:
        node = self.G.nodes[task_id]

        if node["scheduled"]:
            if self.verbose > 0:
                log.info(f"task {task_id} is already scheduled. ignoring it.")
            return {
                "valid_action": False,
                "node_id": task_id,
            }

        m_id = node["machine"]

        prev_task_in_job_id, _ = list(self.G.in_edges(task_id))[0]
        prev_job_node = self.G.nodes[prev_task_in_job_id]

        if not prev_job_node["scheduled"]:
            if self.verbose > 1:
                log.info(f"the previous task (T{prev_task_in_job_id}) in the job is not scheduled yet. "
                         f"Not scheduling task T{task_id} to avoid cycles in the graph.")
            return {
                "valid_action": False,
                "node_id": task_id,
            }

        len_m_routes = len(self.machine_routes[m_id])
        if len_m_routes:

            if self.perform_left_shift_if_possible:

                j_lower_bound_st = prev_job_node["finish_time"]
                duration = node["duration"]
                j_lower_bound_ft = j_lower_bound_st + duration

                # check if task can be scheduled between src and first task
                m_first = self.machine_routes[m_id][0]
                first_task_on_machine_st = self.G.nodes[m_first]["start_time"]

                if j_lower_bound_ft <= first_task_on_machine_st:
                    # schedule task as first node on machine
                    # self.render(show=["gantt_console"])
                    self.info = self._insert_at_index_0(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
                    self.G.add_edge(
                        task_id, m_first,
                        job_edge=False,
                        weight=duration
                    )
                    # self.render(show=["gantt_console"])
                    return self.info
                elif len_m_routes == 1:
                    return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)

                # check if task can be scheduled between two tasks
                for i, (m_prev, m_next) in enumerate(zip(self.machine_routes[m_id], self.machine_routes[m_id][1:])):
                    m_temp_prev_ft = self.G.nodes[m_prev]["finish_time"]
                    m_temp_next_st = self.G.nodes[m_next]["start_time"]

                    if j_lower_bound_ft > m_temp_next_st:
                        continue

                    m_gap = m_temp_next_st - m_temp_prev_ft
                    if m_gap < duration:
                        continue

                    # at this point the task can fit in between two already scheduled task
                    # self.render(show=["gantt_console"])

                    # remove the edge from m_temp_prev to m_temp_next
                    replaced_edge_data = self.G.get_edge_data(m_prev, m_next)
                    st = max(j_lower_bound_st, m_temp_prev_ft)
                    ft = st + duration
                    node["start_time"] = st
                    node["finish_time"] = ft
                    node["scheduled"] = True

                    # from previous task to the task to schedule
                    self.G.add_edge(
                        m_prev, task_id,
                        job_edge=replaced_edge_data['job_edge'],
                        weight=replaced_edge_data['weight']
                    )
                    # from the task to schedule to the next
                    self.G.add_edge(
                        task_id, m_next,
                        job_edge=False,
                        weight=duration
                    )
                    # remove replaced edge
                    self.G.remove_edge(m_prev, m_next)
                    # insert task at the corresponding place in the machine routes list
                    self.machine_routes[m_id] = np.insert(self.machine_routes[m_id], i + 1, task_id)

                    if self.verbose > 1:
                        log.info(f"scheduled task {task_id} on machine {m_id} between task {m_prev:.0f} "
                                 f"and task {m_next:.0f}")
                    # self.render(show=["gantt_console"])
                    return {
                        "start_time": st,
                        "finish_time": ft,
                        "node_id": task_id,
                        "valid_action": True,
                        "scheduling_method": 'left_shift',
                        "left_shift": 1,
                    }
                else:
                    return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)

            else:
                return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)

        else:
            return self._insert_at_index_0(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)

    def _append_at_the_end(self, task_id: int, node: dict, prev_job_node: dict, m_id: int) -> dict:
        prev_m_task = self.machine_routes[m_id][-1]
        prev_m_node = self.G.nodes[prev_m_task]
        self.G.add_edge(
            prev_m_task, task_id,
            job_edge=False,
            weight=int(prev_m_node['duration'])
        )
        self.machine_routes[m_id] = np.append(self.machine_routes[m_id], task_id)
        st = max(prev_job_node["finish_time"], prev_m_node["finish_time"])
        ft = st + node["duration"]
        node["start_time"] = st
        node["finish_time"] = ft
        node["scheduled"] = True
        # return additional info
        return {
            "start_time": st,
            "finish_time": ft,
            "node_id": task_id,
            "valid_action": True,
            "scheduling_method": '_append_at_the_end',
            "left_shift": 0,
        }

    def _insert_at_index_0(self, task_id: int, node: dict, prev_job_node: dict, m_id: int) -> dict:
        self.machine_routes[m_id] = np.insert(self.machine_routes[m_id], 0, task_id)
        st = prev_job_node["finish_time"]
        ft = st + node["duration"]
        node["start_time"] = st
        node["finish_time"] = ft
        node["scheduled"] = True
        # return additional info
        return {
            "start_time": st,
            "finish_time": ft,
            "node_id": task_id,
            "valid_action": True,
            "scheduling_method": '_insert_at_index_0',
            "left_shift": 0,
        }

    def _state_array(self, task_id) -> np.ndarray:
        adj = nx.to_numpy_array(self.G)[1:-1, 1:].astype(dtype=int)  # remove dummy tasks
        # print(adj)
        if self.start:
            self.task_to_machine_mapping = np.zeros(shape=(self.total_tasks_without_dummies, 1), dtype=int)
            self.task_to_duration_mapping = np.zeros(shape=(self.total_tasks_without_dummies, 1), dtype=self.dtype)

            for task_id, data in self.G.nodes(data=True):
                if task_id == self.src_task or task_id == self.sink_task:
                    continue
                else:
                    # index shift because of the removed dummy tasks
                    self.task_to_machine_mapping[task_id - 1] = data["machine"] + 1
                    self.task_to_duration_mapping[task_id - 1] = data["duration"]

            self.task_status = np.zeros((self.total_tasks_without_dummies, 2), dtype=self.dtype)

        if task_id >= 0 and self.info["finish_time"] > 0:
            self.task_status[task_id][1] = 1.0
            self.task_status[task_id][0] = self.info["finish_time"]

        if self.normalize_observation_space:
            # one hot encoding for task to machine mapping
            if self.start:
                # self.task_to_machine_mapping = self.task_to_machine_mapping.astype(int).ravel()
                # n_values = np.max(self.task_to_machine_mapping) + 1
                # self.task_to_machine_mapping = np.eye(n_values)[self.task_to_machine_mapping]
                self.task_to_machine_mapping = self.task_to_machine_mapping / self.n_machines
                self.task_to_duration_mapping = self.task_to_duration_mapping / self.longest_processing_time
            # normalize
            adj = adj / self.longest_processing_time  # note: adj matrix contains weights
            if task_id >= 0 and self.info["finish_time"] > 0:
                self.task_status[task_id][0] /= self.scaling_divisor
            # merge arrays
            res = np.concatenate((adj, self.task_to_machine_mapping, self.task_to_duration_mapping, self.task_status),
                                 axis=1, dtype=self.dtype)
        else:
            res = np.concatenate((adj, self.task_to_machine_mapping, self.task_to_duration_mapping, self.task_status),
                                 axis=1, dtype=self.dtype)

        if self.flat_observation_space:
            # falter observation
            res = np.ravel(res).astype(self.dtype)

        if self.env_transform == 'mask':
            res = OrderedDict({
                "action_mask": np.array(self.valid_action_mask()).astype(np.int32),
                "observations": res
            })

        return res

    def network_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'Task': f'Job {data["job"]}',
                'Start': data["start_time"],
                'Finish': data["finish_time"],
                'Resource': f'Machine {data["machine"]}'
            }
            for task_id, data in self.G.nodes(data=True)
            if data["job"] != -1 and data["finish_time"] is not None
        ])

    def valid_action_mask(self, action_mode: str = None) -> List[bool]:
        if action_mode is None:
            action_mode = self.action_mode

        if action_mode == 'task':
            mask = [False] * self.total_tasks_without_dummies
            for task_id in range(1, self.total_tasks_without_dummies + 1):
                node = self.G.nodes[task_id]

                if node["scheduled"]:
                    continue

                prev_task_in_job_id, _ = list(self.G.in_edges(task_id))[0]
                prev_job_node = self.G.nodes[prev_task_in_job_id]

                if not prev_job_node["scheduled"]:
                    continue

                mask[task_id - 1] = True

            if True not in mask:
                if self.verbose >= 1:
                    log.warning("no action options remaining")
                if not self.env_transform == 'mask':
                    raise RuntimeError("something went wrong")  # todo: remove error?
            return mask
        elif action_mode == 'job':
            task_mask = self.valid_action_mask(action_mode='task')
            masks_per_job = np.array_split(task_mask, self.n_jobs)
            return [True in job_mask for job_mask in masks_per_job]
        else:
            raise ValueError(f"only 'task' and 'job' are valid arguments for 'action_mode'. {action_mode} is not.")
