# Custom Agent for F/T sensor data
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.distributions.diagonal_gaussian import DiagonalGaussian
from tf2rl.policies.wave_model import build_wavenet_model
from tf2rl.policies.tcn import compiled_tcn, TCN

class WaveFTActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action,
                 units=[256, 256], hidden_activation="relu",
                 fix_std=False, const_std=0.1,
                 state_independent_std=True,
                 squash=False, name='GaussianPolicy'):
        super().__init__(name=name)
        self.action_dim = action_dim
        self.dist = DiagonalGaussian(dim=action_dim)
        self._fix_std = fix_std
        self._const_std = const_std
        self._max_action = max_action
        self._squash = squash
        self._state_independent_std = state_independent_std

        self.x_l1 = Dense(128, name="x_L1", activation=hidden_activation)
        self.x_l2 = Dense(128, name="x_L2", activation=hidden_activation)
        self.x_features = Dense(32, name="x_features")#64

        # self.ft_tcn = TCN(return_sequences=False)
        # self.ft_l2 = Dense(units[1], name="ft_L2", activation=hidden_activation)

        self.ft_net = compiled_tcn(return_sequences=False,
                         num_feat=6,
                         num_classes=0,
                         nb_filters=9,#6
                         kernel_size=3,#6
                         dilations=[2 ** i for i in range(4)],
                         nb_stacks=1,
                         max_len=12,
                         use_skip_connections=False,
                         regression=True,
                         output_len=32,#64
                         dropout_rate=0)

        self.l1 = Dense(units[0], name="L1", activation=hidden_activation)
        self.l2 = Dense(units[1], name="L2", activation=hidden_activation)

        self.out_mean = Dense(action_dim, name="L_mean")
        if not self._fix_std:
            if self._state_independent_std:
                self.out_log_std = tf.Variable(
                    initial_value=-0.5*np.ones(action_dim, dtype=np.float32),
                    dtype=tf.float32, name="logstd")
            else:
                self.out_log_std = Dense(
                    action_dim, name="L_sigma")

        self(tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def _compute_dist(self, states):
        """
        Compute multivariate normal distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs mean and standard deviation to compute the distribution
        :return (Dict): Multivariate normal distribution
        """
        # tmp = 26 #14 actions
        tmp = 6 + 6 + 3 + self.action_dim # position + velocity + actions
        x_features = self.x_l1(states[:, :tmp])
        x_features = self.x_l2(x_features)
        x_features = self.x_features(x_features)

        ft_len = int((states.shape[1] - tmp) / 6)

        ft_state = tf.reshape(states[:, tmp:],(-1, ft_len, 6))
        ft_features = self.ft_net(ft_state)


        features = self.l1(tf.keras.layers.Concatenate()([x_features, ft_features]))
        features = self.l2(features)
        mean = self.out_mean(features)
        if self._fix_std:
            log_std = tf.ones_like(mean) * tf.math.log(self._const_std)
        else:
            if self._state_independent_std:
                log_std = tf.tile(
                    input=tf.expand_dims(self.out_log_std, axis=0),
                    multiples=[mean.shape[0], 1])
            else:
                log_std = self.out_log_std(features)
                log_std = tf.clip_by_value(
                    log_std, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        return {"mean": mean, "log_std": log_std}

    def call(self, states, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        param = self._compute_dist(states)
        if test:
            raw_actions = param["mean"]
        else:
            raw_actions = self.dist.sample(param)
        logp_pis = self.dist.log_likelihood(raw_actions, param)

        actions = raw_actions

        if self._squash:
            actions = tf.tanh(raw_actions)
            logp_pis = self._squash_correction(logp_pis, actions)

        return actions * self._max_action, logp_pis, param

    def compute_log_probs(self, states, actions):
        actions /= self._max_action
        param = self._compute_dist(states)
        logp_pis = self.dist.log_likelihood(actions, param)
        if self._squash:
            logp_pis = self._squash_correction(logp_pis, actions)
        return logp_pis

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)

    def _squash_correction(self, logp_pis, actions):
        # assert_op = tf.Assert(tf.less_equal(tf.reduce_max(actions), 1.), [actions])
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        # with tf.control_dependencies([assert_op]):
        diff = tf.reduce_sum(
            tf.math.log(1. - actions ** 2 + self.EPS), axis=1)
        return logp_pis - diff