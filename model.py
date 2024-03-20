import numpy as np
import tensorflow as tf

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, clip_ratio=0.2, gamma=0.99, epsilon=0.1, critic_coef=0.5, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.epsilon = epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        
        # Initialize policy network and critic network
        self.policy_network = self._build_policy_network()
        self.critic_network = self._build_critic_network()
        
        # Initialize optimizer for both networks
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def _build_policy_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model
    
    def _build_critic_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def choose_action(self, state):
        action_probs = self.policy_network.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action
    
    def train(self, states, actions, advantages, discounted_rewards):
        with tf.GradientTape() as policy_tape, tf.GradientTape() as critic_tape:
            probs = self.policy_network(states, training=True)
            action_masks = tf.one_hot(actions, self.action_dim)
            chosen_action_probs = tf.reduce_sum(probs * action_masks, axis=1)

            old_probs = chosen_action_probs  # Assuming discrete actions

            ratios = tf.exp(tf.math.log(chosen_action_probs + 1e-10) - tf.math.log(old_probs + 1e-10))
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            values = self.critic_network(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(discounted_rewards - values))

            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
            entropy_loss = -tf.reduce_mean(entropy)

            total_loss = policy_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

        policy_gradients = policy_tape.gradient(total_loss, self.policy_network.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic_network.trainable_variables)

        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_network.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))
