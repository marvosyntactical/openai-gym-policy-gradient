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
import numpy as np

import tensorflow as tf
k = tf.keras
_K = k.backend
PROFILER = tf.profiler.experimental
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

        # TODO somehow optionally move to GPU
        gpus = tf.config.experimental.list_physical_devices("GPU") 
        print(f"Is there a GPU available: {gpus if len(gpus) else 'No'}"),

        self.build_network() # sets self.model to keras model
        assert hasattr(self, "model")

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.create_file_writer(
                "logs/", max_queue=None, flush_millis=None, filename_suffix=None, name="PGkeras"
        )

        self.start_profile_ops()

        # XXX port saving logic to tf=2.3
        """
        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)
        """

    def start_profile_ops(self, logdir="flops/"):
        opts = PROFILER.ProfilerOptions(
            host_tracer_level=2,
            python_tracer_level=0, 
            device_tracer_level=1 
        )
        PROFILER.start(
            logdir, opts
        )

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
        prob_weights = self.model.predict(observation)

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def learn(self):
        # Discount and normalize episode reward
        self.discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        obs = np.vstack(self.episode_observations) # shape batch x states 
        acts = np.array(self.episode_actions) # shape actions x 1

        # Train on episode
        self.model.fit(obs, acts)

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        # XXX port saving logic 
        """
        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            print("Model saved in file: %s" % save_path)
        """

        return self.discounted_episode_rewards_norm

    def discount_and_norm_rewards(self):
        # discounted reward calc 
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def reward_guided_loss(self, y_true, y_pred):
        """Args: labels, softmax probs"""

        loss = _K.categorical_crossentropy(y_true, y_pred)

        return _K.mean(loss*self.discounted_episode_rewards_norm)

    def build_network(self):

        model = k.Sequential()
        model.add(k.layers.InputLayer(batch_input_shape=(None, self.n_x)))
        model.add(k.layers.Dense(10, activation="relu"))
        model.add(k.layers.Dense(10, activation="relu"))
        model.add(k.layers.Dense(self.n_y, activation="softmax"))

        # training logic

        adam = tf.optimizers.Adam(self.lr)
        # adam.minimize(self.reward_guided_loss, model)

        model.compile(loss=self.reward_guided_loss, optimizer=adam, metrics=['mae'])

        self.model = model
        


    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()