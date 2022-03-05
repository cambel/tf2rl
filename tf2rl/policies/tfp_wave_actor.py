import numpy as np
import rospy
import tensorflow as tf
import tensorflow_probability as tfp

from tf2rl.policies.tcn import compiled_tcn
layers = tf.keras.layers

class WaveFTActor(tf.keras.Model):
    LOG_STD_CAP_MAX = -2  # np.e**-1 = 0.3679 || np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, units=(256, 256),
                 hidden_activation="relu", state_independent_std=False,
                 squash=False, name='gaussian_policy'):
        super().__init__(name=name)

        self._state_independent_std = state_independent_std
        self._squash = squash
        self.action_dim = action_dim
        self.base_layers = []

        prefix = "ur3e_gym"
        self.wrench_hist_size = rospy.get_param(prefix + "/wrench_hist_size", 12)
        self.ft_size = rospy.get_param(prefix + "/ft_size", 6)

        self.x_l1 = layers.Dense(128, name="x_L1", activation=hidden_activation)
        self.x_l2 = layers.Dense(128, name="x_L2", activation=hidden_activation)
        self.x_features = layers.Dense(32, name="x_features")#64

        self.ft_net = compiled_tcn(return_sequences=False,
                         num_feat=self.ft_size,
                         num_classes=0,
                         nb_filters=9,#6
                         kernel_size=3,#6
                         dilations=[2 ** i for i in range(4)],
                         nb_stacks=1,
                         max_len=self.wrench_hist_size,
                         use_skip_connections=False,
                         regression=True,
                         output_len=32,#64
                         dropout_rate=0)

        for unit in units:
            self.base_layers.append(layers.Dense(unit, activation=hidden_activation))

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        if self._state_independent_std:
            self.out_logstd = tf.Variable(
                initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                dtype=tf.float32, name="L_logstd")
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd")

        self._max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)

    def _compute_dist(self, states):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.MultivariateNormalDiag
                Multivariate normal distribution object whose mean and
                standard deviation is output of a neural network
        """
        wrench_hist_size = self.wrench_hist_size * self.ft_size

        # everything except the ft data
        x_features = self.x_l1(states[:, :-wrench_hist_size])
        x_features = self.x_l2(x_features)
        x_features = self.x_features(x_features)


        ft_state = tf.reshape(states[:, -wrench_hist_size:],(-1, self.wrench_hist_size, self.ft_size))
        ft_features = self.ft_net(ft_state)

        features = tf.keras.layers.Concatenate()([x_features, ft_features])

        for cur_layer in self.base_layers:
            features = cur_layer(features)

        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))

    def call(self, states, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        dist = self._compute_dist(states)
        if test:
            raw_actions = dist.mean()
        else:
            raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        if self._squash:
            actions = tf.tanh(raw_actions)
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=1)
            log_pis -= diff
        else:
            actions = raw_actions

        actions = actions * self._max_action
        return actions, log_pis

    def compute_log_probs(self, states, actions):
        raw_actions = actions / self._max_action
        dist = self._compute_dist(states)
        logp_pis = dist.log_prob(raw_actions)
        return logp_pis

    def compute_entropy(self, states):
        dist = self._compute_dist(states)
        return dist.entropy()
