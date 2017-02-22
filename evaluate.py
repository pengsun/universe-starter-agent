from envs import create_env
import numpy as np
import time
import argparse


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
        action = fetched[0]

        state, reward, terminal, _ = env.step(action.argmax())
        if render:
            env.render()

        episode_reward[n_episode] += reward
        if verbose:
            print("#step = {}, action = {}".format(step, action.argmax()))
            print("reward = {}".format(reward))

        if terminal:
            print("#episode = {}, #step = {}, reward sum = {}".format(n_episode, step, episode_reward[n_episode]))
            episode_length[n_episode] = step
            env.reset()
            step = 0
            n_episode += 1
        else:
            step += 1
            time.sleep(sleep_time)

    print('evaluation done.')
    print('avg score = {}'.format(episode_reward.mean()))
    print('avg episode length = {}'.format(episode_length.mean()))


def main(args):
    env_id = args.env_id
    ckpt_dir = args.ckpt_dir
    max_episodes = args.max_episodes

    # env
    env = create_env(env_id, 0, 1)
    if args.render:
        env.render()

    # work-around to the nasty env.render() failing issue when working with tensorflow
    # see https://github.com/openai/gym/issues/418
    import tensorflow as tf
    from model import Convx2LSTMActorCritic

    # model
    with tf.variable_scope("global"):
        network = Convx2LSTMActorCritic(env.observation_space.shape, env.action_space.n)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    # load model parameters
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
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
    parser.add_argument('--max-episodes', default=2, type=int, help='Number of episodes to evaluate')
    parser.add_argument('--sleep-time', default=0.0, type=float, help='sleeping time')
    parser.add_argument('--render', action='store_true', help='render screen')
    parser.add_argument('--verbose', action='store_true', help='verbose')

    args = parser.parse_args()
    main(args=args)