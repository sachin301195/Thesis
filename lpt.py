import numpy as np
import json
from evaluate import evaluate_instance


def lpt(jsp, opt_value):
    _, num_mc, num_jobs = jsp.shape
    machines_ = np.array(jsp[0])
    tmp = np.zeros((num_jobs, num_mc + 1), dtype=int)
    tmp[:, :-1] = machines_
    machines_ = tmp

    durations_ = np.array(jsp[1])
    tmp = np.zeros((num_jobs, num_mc + 1), dtype=int)
    tmp[:, :-1] = durations_
    durations_ = tmp

    indices = np.zeros([num_jobs], dtype=int)

    # Internal variables
    previousTaskReadyTime = np.zeros([num_jobs], dtype=int)
    machineReadyTime = np.zeros([num_mc], dtype=int)

    placements = [[] for _ in range(num_mc)]
    time = 0
    final_duration = np.zeros(num_mc, dtype=int)
    while (not np.array_equal(indices, np.ones([num_jobs], dtype=int) * num_mc)):

        machines_Idx = machines_[range(num_jobs), indices]
        durations_Idx = durations_[range(num_jobs), indices]

        # 1: Check previous Task and machine availability
        mask = np.zeros([num_jobs], dtype=bool)

        for j in range(num_jobs):

            if previousTaskReadyTime[j] == 0 and machineReadyTime[machines_Idx[j]] == 0 and indices[j] < num_mc:
                mask[j] = True

        # 2: Competition SPT

        for m in range(num_mc):

            job = None
            duration = 0

            for j in range(num_jobs):

                if machines_Idx[j] == m and durations_Idx[j] > duration and mask[j]:
                    job = j
                    duration = durations_Idx[j]
                if duration > 0:
                    final_duration[m] = duration
                else:
                    final_duration[m] = 0

            if job != None:
                placements[m].append([job, indices[job]])
                # final_duration = durations_[job, indices[job]]

                previousTaskReadyTime[job] += durations_Idx[job]
                machineReadyTime[m] += durations_Idx[job]

                indices[job] += 1

        time += 1

        previousTaskReadyTime = np.maximum(previousTaskReadyTime - 1, np.zeros([num_jobs], dtype=int))
        machineReadyTime = np.maximum(machineReadyTime - 1, np.zeros([num_mc], dtype=int))

    makespan = time + np.max(final_duration) - 1

    return placements, makespan


if __name__ == "__main__":
    opt_value_list = []
    makespan_list = []
    for i in range(100):
        jsp, opt_value = evaluate_instance("8x8", i)
        placements, makespan = lpt(jsp, opt_value)
        opt_value_list.append(opt_value)
        makespan_list.append(makespan)
        if i == 99:
            print(opt_value, makespan)

    opt_value_avg = sum(opt_value_list) / 100
    makespan_avg = sum(makespan_list) / 100
    gap = (opt_value_avg - makespan_avg) / opt_value_avg * 100
    print(opt_value_avg, makespan_avg, gap)
