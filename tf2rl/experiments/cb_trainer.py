import os
import time
import rospy
import logging
import argparse

import numpy as np
import tensorflow as tf
from gym.spaces import Box
import optuna

from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, restore_replay_buffer, save_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer


if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


class Trainer:
    """
    Trainer class for off-policy reinforce learning

    Command Line Args:

        * ``--max-steps`` (int): The maximum steps for training. The default is ``int(1e6)``
        * ``--episode-max-steps`` (int): The maximum steps for an episode. The default is ``int(1e3)``
        * ``--n-experiments`` (int): Number of experiments. The default is ``1``
        * ``--show-progress``: Call ``render`` function during training
        * ``--save-model-interval`` (int): Interval to save model. The default is ``int(1e4)``
        * ``--save-summary-interval`` (int): Interval to save summary. The default is ``int(1e3)``
        * ``--model-dir`` (str): Directory to restore model.
        * ``--dir-suffix`` (str): Suffix for directory that stores results.
        * ``--normalize-obs``: Whether normalize observation
        * ``--logdir`` (str): Output directory name. The default is ``"results"``
        * ``--evaluate``: Whether evaluate trained model
        * ``--test-interval`` (int): Interval to evaluate trained model. The default is ``int(1e4)``
        * ``--show-test-progress``: Call ``render`` function during evaluation.
        * ``--test-episodes`` (int): Number of episodes at test. The default is ``5``
        * ``--save-test-path``: Save trajectories of evaluation.
        * ``--show-test-images``: Show input images to neural networks when an episode finishes
        * ``--save-test-movie``: Save rendering results.
        * ``--use-prioritized-rb``: Use prioritized experience replay
        * ``--use-nstep-rb``: Use Nstep experience replay
        * ``--n-step`` (int): Number of steps for nstep experience reward. The default is ``4``
        * ``--logging-level`` (DEBUG, INFO, WARNING): Choose logging level. The default is ``INFO``
    """

    def __init__(
            self,
            policy,
            env,
            args,
            seed=0,
            test_env=None,
            teacher_policy=False,
            save_best_policy=False,
            trial=None,
            exp=None):
        """
        Initialize Trainer class

        Args:
            policy: Policy to be trained
            env (gym.Env): Environment for train
            args (Namespace or dict): config parameters specified with command line
            test_env (gym.Env): Environment for test.
        """
        if isinstance(args, dict):
            _args = args
            args = policy.__class__.get_argument(Trainer.get_argument())
            args = args.parse_args([])
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")
        
        tf.random.set_seed(seed)

        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env
        self._teacher_policy = teacher_policy
        print("Teaching mode", teacher_policy)
        print("Save best policy mode", save_best_policy)
        self._save_best_policy = save_best_policy
        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(
                shape=env.observation_space.shape)

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            time_format='S.%f',
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)

        if trial is not None:
            index = self._output_dir.find("3d")
            new_output = self._output_dir[:index+2] + "-trial_" +str(trial.number) + self._output_dir[index+2:]
            self._output_dir = new_output
        if exp is not None:
            index = self._output_dir.find("3d")
            new_output = self._output_dir[:index+2] + exp + self._output_dir[index+2:]
            self._output_dir = new_output
            
        if self._model_dir is None:
            self.replay_buffer_path = self._output_dir + '/replay_buffer.pkl'
        else:
            self.replay_buffer_path = self._model_dir + '/replay_buffer.pkl'

        if args.evaluate:
            assert args.model_dir is not None
        self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

        # setup optuna optimization
        self.trial = trial

    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=self._output_dir, max_to_keep=5)
        
        if model_dir is not None:
            assert os.path.isdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

    def __call__(self):
        """
        Execute training
        """
        if self._evaluate:
            self.evaluate_policy_continuously()

        best_test_score = -np.inf

        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        # if os.path.exists(self.replay_buffer_path):
        #     print("Restoring reply buffer")
        #     replay_buffer.add(**restore_replay_buffer(self.replay_buffer_path))

        obs = self._env.reset()

        teaching_mode = False

        actual_episode_steps = 0
        total_agent_control_time = 0.0

        total_cumulative_reward = 0
        # To evaluate the ability of the policy to learn, we will measure the difference between
        # the average return value of the first half of the training 
        # and the average return value of the second half
        learning_range_first_half = 0
        learning_range_second_half = 0

        while total_steps < self._max_steps:

            # Call the teacher policy here
            if self._teacher_policy and teaching_mode:
                action = np.ones(self._env.n_actions)
                action[:6] *= 1.0 # Slow motion

                ## Fix policy
                # action[2] = -0.75
                
                # Two step policy
                xy_error = obs[:2]
                # print(round(np.linalg.norm(xy_error), 4))
                if np.linalg.norm(xy_error) < .02:
                    action[2] = -0.0
                    action[6:] *= -0.5 # Half compliance
                else:
                    action[2] = -0.9
                    action[8] = 1.0 # High compliance
                    action[6:] *= 0.75

            else:
                if total_steps < self._policy.n_warmup:
                    action = self._env.action_space.sample()
                else:
                    action = self._policy.get_action(obs)

            st = rospy.get_time()
            next_obs, reward, done, info = self._env.step(action)
            total_agent_control_time += rospy.get_time() - st

            if self._show_progress:
                self._env.render()
            episode_steps += 1
            episode_return += reward if not teaching_mode else 0
            total_steps += 1
            
            actual_episode_steps += 1 if not teaching_mode else 0
            
            tf.summary.experimental.set_step(total_steps)

            done_flag = done
            if (hasattr(self._env, "_max_episode_steps") and
                    episode_steps == self._env._max_episode_steps):
                done_flag = False
            replay_buffer.add(obs=obs, act=action,
                              next_obs=next_obs, rew=reward, done=done_flag)
            obs = next_obs

            collision = info.get("collision", False)
            success = info.get("success", False)
            dist = info.get("dist", 0)
            force = info.get("force", 0)
            jerk = info.get("jerk", 0)
            vel = info.get("vel", 0)
            cumulated_reward_details = info.get("cumulated_reward_details", np.zeros(3))
            r_dist = cumulated_reward_details[0]
            r_force = cumulated_reward_details[1]
            r_jerk = cumulated_reward_details[2]
            r_vel = cumulated_reward_details[3]
            w_dist = obs[-4]
            w_force = obs[-5]
            w_jerk = obs[-6]

            if self._teacher_policy:
                if collision and not teaching_mode: # start teaching mode on collision
                    teaching_mode = True
                    print('\033[36m' + "*** TEACHING MODE ON***" + '\033[0m')
                elif teaching_mode and (collision or done): # stop if there is another collision or if the task is completed when in teaching mode
                    teaching_mode = False
                    print('\033[36m' + "*** TEACHING MODE OFF***" + '\033[0m')

            if (done and not teaching_mode) or episode_steps == self._episode_max_steps:
                n_episode += 1
                total_episode_time = time.perf_counter() - episode_start_time
                policy_time = (total_episode_time - total_agent_control_time) / episode_steps
                time_per_step = (total_agent_control_time / episode_steps)
                self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f} dt {5:5.2f}".format(
                    n_episode, total_steps, actual_episode_steps, episode_return, policy_time, time_per_step))
                self._detailed_log(n_episode, total_steps, episode_steps, episode_return)
                tf.summary.scalar(name="Common/training_return", data=episode_return)
                tf.summary.scalar(name="Common/training_episode_length", data=actual_episode_steps)
                tf.summary.scalar(name="Common/computation_time", data=policy_time+time_per_step)

                if collision:
                    performance_metric = -self._episode_max_steps * 2
                elif success:
                    performance_metric = self._episode_max_steps - actual_episode_steps
                else:
                    performance_metric = -self._episode_max_steps

                tf.summary.scalar(name="Common/performance_metric", data=performance_metric)

                # publish to TF reward the information about the distance, the force and the jerkiness
                tf.summary.scalar(name="Common/dist", data=dist)
                tf.summary.scalar(name="Common/vel", data=vel)
                tf.summary.scalar(name="Common/force", data=force)
                tf.summary.scalar(name="Common/jerk", data=jerk)                
                tf.summary.scalar(name="Common/w_dist", data=w_dist)
                tf.summary.scalar(name="Common/w_force", data=w_force)
                tf.summary.scalar(name="Common/w_jerk", data=w_jerk)                
                tf.summary.scalar(name="Common/r_dist", data=r_dist)
                tf.summary.scalar(name="Common/r_force", data=r_force)
                tf.summary.scalar(name="Common/r_jerk", data=r_jerk)
                tf.summary.scalar(name="Common/r_vel", data=r_vel)

                
                obs = self._env.reset()
                
                # Update policy if defined to do so
                if self._policy.update_interval == 0:
                    self.update_policy(replay_buffer, save_summary=True)
                replay_buffer.on_episode_end()
                # Save replay buffer
                # save_replay_buffer(replay_buffer, self.replay_buffer_path)

                total_cumulative_reward += episode_return
                if total_steps < self._max_steps / 2 : learning_range_first_half += episode_return
                else : learning_range_second_half += episode_return

                # Send intermediate value of the current training episode to the current optuna trial
                if self.trial is not None : 
                    self.trial.report(episode_return, n_episode)
                    if self.trial.should_prune():
                        print("[PRUNED]")
                        raise optuna.TrialPruned()

                episode_steps = 0
                episode_return = 0
                actual_episode_steps = 0
                episode_start_time = time.perf_counter()
                total_agent_control_time = 0.0

            elif self._policy.update_interval != 0 and total_steps % self._policy.update_interval == 0:
                self.update_policy(replay_buffer, save_summary=(total_steps % self._save_summary_interval == 0))

            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._test_interval == 0:
                print('=============== TESTING POLICY =================')
                avg_test_return, avg_test_steps, success_rate = self.evaluate_policy(total_steps)
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))
                tf.summary.scalar(
                    name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(
                    name="Common/average_test_episode_length", data=avg_test_steps)
                tf.summary.scalar(name="Common/fps", data=1./(policy_time+time_per_step))
                tf.summary.scalar(name="Common/agent_hz", data=1./time_per_step)
                tf.summary.scalar(name="Common/success_rate", data=success_rate)
                print('=============== END OF TESTING =================')

                if self._save_best_policy:
                    test_score = avg_test_return + (success_rate * 100)
                    if best_test_score < test_score:
                        print('*** Saving New Best Policy ***')
                        self.checkpoint_manager.save()
                        best_test_score = test_score        
                
                # Start a new episode
                obs = self._env.reset()

            if not self._save_best_policy and total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()

        # self.checkpoint_manager.save(999)
        # Measuring the difference
        learning_range = learning_range_second_half - learning_range_first_half
        tf.summary.flush()

        return total_cumulative_reward/n_episode, learning_range/n_episode

    def update_policy(self, replay_buffer, save_summary=False):
        samples = replay_buffer.sample(self._policy.batch_size)
        with tf.summary.record_if(save_summary):
            self._policy.train(
                samples["obs"], samples["act"], samples["next_obs"],
                samples["rew"], np.array(samples["done"], dtype=np.float32),
                None if not self._use_prioritized_rb else samples["weights"])
        if self._use_prioritized_rb:
            td_error = self._policy.compute_td_error(
                samples["obs"], samples["act"], samples["next_obs"],
                samples["rew"], np.array(samples["done"], dtype=np.float32))
            replay_buffer.update_priorities(
                samples["indexes"], np.abs(td_error) + 1e-6)

    def evaluate_policy_continuously(self):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)

        self.evaluate_policy(total_steps=0)
        while True:
            latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
            if self._latest_path_ckpt != latest_path_ckpt:
                self._latest_path_ckpt = latest_path_ckpt
                self._checkpoint.restore(self._latest_path_ckpt)
                self.logger.info("Restored {}".format(self._latest_path_ckpt))
            self.evaluate_policy(total_steps=0)

    def evaluate_policy(self, total_steps):
        tf.summary.experimental.set_step(total_steps)
        if self._normalize_obs:
            self._test_env.normalizer.set_params(
                *self._env.normalizer.get_params())
        avg_test_return = 0.
        avg_test_steps = 0
        successes = 0.
        collisions = 0
        performance_metric = 0.
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            avg_test_steps += 1
            for j in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)
                next_obs, reward, done, info = self._test_env.step(action)
                if info.get("success", False):
                    successes += 1
                avg_test_steps += 1
                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=action,
                                      next_obs=next_obs, rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if done:
                    break
            if info.get("collision", False):
                collisions += 1
                performance_metric += -self._episode_max_steps * 2
            elif info.get("success", False):
                performance_metric += self._episode_max_steps - j
            else:
                performance_metric += -self._episode_max_steps
            print('Test episode {0: 3} steps {1: 4} return {2:8.2f}'.format(i+1, j, episode_return))
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)
            if self._save_test_path:
                save_path(replay_buffer._encode_sample(np.arange(self._episode_max_steps)),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return
        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images,)
        tf.summary.scalar(name="Common/test_performance_metric", data=performance_metric/self._test_episodes)
        tf.summary.scalar(name="Common/test_collisions", data=collisions)
        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes, successes / self._test_episodes

    def _detailed_log(self, n_episode, total_steps, episode_steps, episode_return):
        logfile = self._output_dir + '/detailed_log.npy'
        try:
            tmp = np.load(logfile, allow_pickle=True).tolist()
            tmp.append([n_episode, total_steps, episode_steps, episode_return])
            np.save(logfile, tmp)
        except FileNotFoundError:
            np.save(logfile, [[n_episode, total_steps, episode_steps, episode_return]])
        pass

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = (args.episode_max_steps
                                   if args.episode_max_steps is not None
                                   else args.max_steps)
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._evaluate = args.evaluate
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        """
        Create or update argument parser for command line program

        Args:
            parser (argparse.ArgParser, optional): argument parser

        Returns:
            argparse.ArgParser: argument parser
        """
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(1e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        parser.add_argument('--normalize-obs', action='store_true',
                            help='Normalize observation')
        parser.add_argument('--logdir', type=str, default='results',
                            help='Output directory')
        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(1e4),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        return parser
