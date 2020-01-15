from baselines.envs import TorchEnv, const

MAX_EPISODE_LEN = 200

if __name__ == "__main__":
    env = TorchEnv(const.SPARSE_HALF_CHEETAH, MAX_EPISODE_LEN)

    s = env.reset()

    for _ in range(MAX_EPISODE_LEN):
        a = env.sample_action()
        s, r, d = env.step(a)
        print(r)
        env.render()
        if d:
            break
    
    env.close()
