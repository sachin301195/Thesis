import numpy as np
from jsp_env.src.JssEnv import JssEnv

instance_path = r"./instances/trail"
env = JssEnv(
    env_config={
        "instance_path": instance_path
    }
)

score = 0
step = 0
obs = env.reset()
done = False
while not done:
    print(env.legal_actions)
    action = np.random.choice(len(env.legal_actions), 1)[0]
    print(action)
    state, reward, done, info = env.step(action)
    score += reward
    if done:
        print(info)
        print(score)
        env.render()
