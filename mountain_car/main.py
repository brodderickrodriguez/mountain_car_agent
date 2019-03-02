import numpy as np
import matplotlib.pyplot as plt
import logging

from mountain_car import ReadWrite as RW
from mountain_car import Configuration as Config


def get_all_state_strings():
    return [str(i).zfill(Config.NUM_PERCEPTRONS) for i in range(Config.NUM_STATES)]


def init_bins():
    return np.array([np.linspace(-1.2, 0.6, Config.NUM_BINS), np.linspace(-0.07, 0.07, Config.NUM_BINS)])


# def init_q():
#     return {s: {a: 0.0 for a in range(Config.env.action_space.n)} for s in get_all_state_strings()}


def init_q():
    q = {}
    for s in get_all_state_strings():
        q.update({s: {0: 0, 1: 0, 2: 0}})
    return q


def observation_to_state_string(observation, target_bins):
    state = np.array([np.digitize(observation[i], target_bins[i]) for i in range(Config.NUM_PERCEPTRONS)])
    return ''.join(str(int(e)) for e in state)


def run_single_episode(target_bins, q_func, epsilon, is_training=True, render=False):
    observation = Config.env.reset()
    done, step_count, total_reward = False, 0, 0
    state = observation_to_state_string(observation=observation, target_bins=target_bins)
    action = Config.env.action_space.sample()

    while not done:
        if render:
            Config.env.render()

        step_count += 1
        observation, reward, done, _ = Config.env.step(action=action)
        total_reward += reward

        next_state = observation_to_state_string(observation=observation,
                                                 target_bins=target_bins)

        next_action = Config.ra.next_action(q=q_func,
                                            next_state=next_state,
                                            epsilon=epsilon,
                                            is_training=is_training)

        if is_training:
            q_func[state][action] += Config.ra.get_new_q_value(q=q_func,
                                                               state=state,
                                                               action=action,
                                                               next_reward=reward,
                                                               next_state=next_state,
                                                               next_action=next_action)
            # logging.info('updated state {s} action {a} with value {v}'
            #              .format(s=state,
            #                      a=action,
            #                      v=q_func[state][action]))

        state, action = next_state, next_action

    Config.env.close()
    return total_reward, step_count, q_func


def run_n_episodes(target_bins, n, q_func):
    episode_lengths, episode_rewards = [], []

    for episode in range(n):
        # if episode % (n / 10) == 0:
        print('\tepisode # ', episode)

        epsilon = 2 / np.sqrt(episode + 1)
        # epsilon = 1 / 5
        episode_reward, episode_length, q_func = run_single_episode(target_bins=target_bins,
                                                                    q_func=q_func,
                                                                    epsilon=epsilon)
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)

    return episode_rewards, episode_lengths, q_func


def smart_tune_params():
    bins = init_bins()
    alphas = [i * 0.01 + 0.01 for i in range(10)]
    gammas = [i * 0.01 + 0.9 for i in range(10)]
    best = (np.inf, 0, 0)

    for alpha in alphas:
        Config.ALPHA = alpha
        for gamma in gammas:
            Config.GAMMA = gamma
            all_run_steps = []
            print('\n\nA={a} G={g}'.format(a=alpha, g=gamma))

            for i in range(5):
                q = init_q()
                _, _, q = run_n_episodes(target_bins=bins, n=100, q_func=q)
                reward, steps, q = run_single_episode(bins, q, epsilon=0, is_training=False, render=False)
                all_run_steps.append((steps, alpha, gamma))

            avg_step = sum([all_run_steps[i][0] for i in range(len(all_run_steps))]) / len(all_run_steps)
            print('avg is ', avg_step)

            if avg_step < best[0]:
                best = (avg_step, alpha, gamma)

    print('best score of {s} achieved with alpha {a} and gamma {g}'.format(s=best[0], a=best[1], g=best[2]))


def collect_data(q_func_filename):
    rewards, steps = [], []
    q = RW.read_q_func(q_func_filename)
    bins = init_bins()

    for i in range(20):
        reward, step, q = run_single_episode(bins, q, epsilon=0, is_training=False, render=False)
        rewards.append(reward)
        steps.append(step)

    print(rewards)
    print(steps)

    std_dev = np.std(rewards)
    mean = np.mean(rewards)

    print(std_dev, mean)
    print(np.mean(steps))

    plt.plot(rewards)
    # plt.plot(steps)
    plt.show()


def simple_train(q_func_filename):
    bins = init_bins()
    RUNS, EPISODES, all_rewards = 20, 20, []

    for run in range(RUNS):
        print('run is ', run)
        q = init_q()
        episode_rewards, step_counts, q = run_n_episodes(target_bins=bins, n=EPISODES, q_func=q)
        all_rewards.append(episode_rewards)

    reshape_reward = [[row[i] for row in all_rewards] for i in range(len(all_rewards[0]))]
    averages = [np.mean(run_results) for run_results in reshape_reward]
    stddev = [np.std(episode_rewards) for episode_rewards in reshape_reward]

    xs = np.arange(0, len(averages))
    print(averages, len(averages))
    print(xs, len(xs))

    plt.xticks(xs)
    plt.ylabel('return')
    plt.xlabel('episode')
    plt.errorbar(x=xs, y=averages, yerr=stddev)
    plt.plot(averages)
    plt.show()


if __name__ == '__main__':
    # ** set up logging **
    logging.basicConfig(filename='out.log', level=logging.DEBUG, filemode='w')
    logging.info('beginning RL')
    logging.disable(True)

    q_fname = 'q_func_' + str(Config.ra.algorithm_sequence) + '.json'

    # smart_tune_params()
    # collect_data(q_fname)
    simple_train(q_fname)
