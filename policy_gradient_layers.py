"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network
Uses tf.layers to build the neural network

"""

""" # below code to import tf>=2.0 as tf<2.0
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
tf.disable_eager_execution()
"""
import tensorflow as tf
import numpy as np
# from tensorflow.python.framework import ops

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        self.cost_history = []

        self.sess = tf.Session() # tf: init training by calling tf.Session()

        # TODO somehow optionally move to GPU
        gpu  tf.config.experimental.list_physical_devices("GPU")
        print("Is there a GPU available: "),

        self.build_network()
        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)


    def profile_ops(self):
        
        self.run_meta = tf.RunMetadata()
    
        opts_params = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        opts_flops = tf.profiler.ProfileOptionBuilder.float_operation()
        opts_flops['output'] = "none" # suppress outputs to stdout as I print manually; delete this file
        opts_flops["run_meta_path"] = True  # FIXME make this work
        flop_info = tf.profiler.profile(self.sess.graph, cmd="op", options=opts_flops)
        param_info = tf.profiler.profile(self.sess.graph, cmd="op", options=opts_params)

        return flop_info.total_float_ops.real, param_info.total_parameters.real


    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)
        self.episode_actions.append(a)


    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        # Reshape observation to (1, num_features)
        observation = observation[np.newaxis, :]

        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode
        self.sess.run(self.train_op, feed_dict={
             self.X: np.vstack(self.episode_observations), # shape [ examples, number of inputs]
             self.Y: np.array(self.episode_actions), # shape [actions, ]
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            print("Model saved in file: %s" % save_path)

        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards



    def build_network(self):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [None, self.n_x], name="X")
            self.Y = tf.placeholder(tf.int32, [None], name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        A1 = tf.layers.dense(
            inputs=self.X,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        A2 = tf.layers.dense(
            inputs=A1,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        # fc3
        Z3 = tf.layers.dense(
            inputs=A2,
            units=self.n_y,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3'
        )

        # Softmax outputs
        self.outputs_softmax = tf.nn.softmax(Z3, name='A3')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Z3, labels=self.Y)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()