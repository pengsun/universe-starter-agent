from envs import create_env
from gym import wrappers
import numpy as np
import time
import argparse


def avg_err(a):
    avg = a.mean()
    trials = len(a)
    if trials == 1:
        err = 0.0
    else:
        err = np.std(a) / (np.sqrt(trials) - 1)
    return avg, err


def evaluate_loop(env, network, max_episodes, args):
    sleep_time = args.sleep_time
    render = args.render
    verbose = args.verbose

    last_state = env.reset()
    last_features = network.get_initial_features()
    n_episode, step = 0, 0
    episode_reward = np.zeros((max_episodes,), dtype='float32')
    episode_length = np.zeros((max_episodes,), dtype='float32')

    print('evaluating for {} episodes...'.format(max_episodes))
    while n_episode < max_episodes:
        fetched = network.act(last_state, *last_features)
        action, features = fetched[0], fetched[2:]

        state, reward, terminal, _ = env.step(action.argmax())
        if render:
            env.render()

        episode_reward[n_episode] += reward
        if verbose:
            print("#step = {}, action = {}".format(step, action.argmax()))
            print("reward = {}".format(reward))

        if terminal:
            last_state = env.reset()
            last_features = network.get_initial_features()

            print("#episode = {}, #step = {}, reward sum = {}".format(n_episode, step, episode_reward[n_episode]))
            episode_length[n_episode] = step
            step = 0
            n_episode += 1
        else:
            last_state = state
            last_features = features

            step += 1
            time.sleep(sleep_time)

    print('evaluation done.')
    s_avg, s_err = avg_err(episode_reward)
    print('scores = {} +- {}'.format(s_avg, s_err))
    l_avg, l_err = avg_err(episode_length)
    print('episode length = {} +- {}'.format(l_avg, l_err))


def main(args):
    env_id = args.env_id
    max_episodes = args.max_episodes
    ckpt_dir = args.ckpt_dir
    output_dir = args.output_dir

    # env
    env = create_env(env_id, 0, 1)
    if len(output_dir) > 0:
        env = wrappers.Monitor(env, output_dir)
    if args.render:
        env.render()

    # work-around to the nasty env.render() failing issue when working with tensorflow
    # see https://github.com/openai/gym/issues/418
    import tensorflow as tf
    from model import Convx2LSTMActorCritic

    # model
    sess = tf.Session()
    with tf.variable_scope("global"):
        network = Convx2LSTMActorCritic(env.observation_space.shape, env.action_space.n)
    init = tf.global_variables_initializer()
    sess.run(init)

    # load model parameters
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
        raise Exception('cannot find checkpoint path')

    # run evaluating
    with sess.as_default():
        evaluate_loop(env, network, max_episodes, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env-id', default="BreakoutDeterministic-v3", help='Environment id')
    parser.add_argument('--ckpt-dir', default="save/breakout/train", help='Checkpoint directory path')
    parser.add_argument('--output-dir', default="/tmp/myexp", help='Output directory path')
    parser.add_argument('--max-episodes', default=2, type=int, help='Number of episodes to evaluate')
    parser.add_argument('--sleep-time', default=0.0, type=float, help='sleeping time')
    parser.add_argument('--render', action='store_true', help='render screen')
    parser.add_argument('--verbose', action='store_true', help='verbose')

    args = parser.parse_args()
    main(args=args)