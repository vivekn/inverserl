import numpy as np
from gridworlds import *

class IRLModel:
    def __init__(self, nstates, nactions,
        nrewards, nfeatures, T, gamma, state_features):

        self.nstates = nstates
        self.nactions = nactions
        self.nfeatures = nfeatures # num features for state observations
        self.T = T # transition model
        self.gamma = gamma
        # Weights for the linear reward functions
        self.Theta = np.random.rand(nrewards, nfeatures)
        # Weights for the reward transition functions
        self.omega = np.random.rand(nrewards, nrewards)
        # Probabilities for the initial state distribution
        self.nu = np.ones((1, nstates)) / nstates
        # Probabilities for the initial reward function distribution
        self.sigma = np.ones((1, nrewards)) / nrewards
        # function that returns features for a state
        self.state_features = state_features

    def learn(self, trajectories, tolerance, max_iters):
        """
        Learns parameters by using EM.
        A trajectory is a sequence of (state, action) tuples
        """
        self.trajectories = trajectories
        curr_likelihood = self.training_log_likelihood()
        last_likelihood = curr_likelihood - 1e9
        iter = 0
        while (iter < max_iters and
            (abs(curr_likelihood - last_likelihood) > tolerance)):
            iter = iter + 1
            # Maximize parameters simultaneously
            nu = self.maximize_nu()
            sigma = self.maximize_sigma()
            Theta = self.maximize_reward_weights()
            omega = self.maximize_reward_transitions()
            # Set parameters
            self.nu, self.sigma, self.Theta, self.omega = nu, sigma, Theta, omega
            # Compute likelihoods
            last_likelihood = curr_likelihood
            curr_likelihood = self.training_log_likelihood()

    def maximize_nu(self):
        """
        TODO: Maximize the initial state probabilities according to expert trajectories
        @ Daniel/Karthik
        """
        pass

    def maximize_sigma(self):
        """
        TODO: Maximize the initial reward function probabilities according to expert trajectories
        @ Daniel/Karthik
        """
        pass

    def maximize_reward_weights(self):
        """
        TODO: Find the optimal set of weights for the reward functions using gradient ascent
        @ Karthik
        """
        pass

    def maximize_reward_transitions(self):
        """
        TODO: Find the optimal set of weights for the reward transitions
        @ Daniel
        """
        pass

    def training_log_likelihood(self):
        """
        TODO: Returns log likelihood for the training trajectories
        based on the current parameters of the model
        """
        return 0.0

    def validate(self, trajectories):
        return sum(self.viterbi_log_likelihood(t) for t in trajectories)

    def viterbi_log_likelihood(self, trajectory):
        """
        Compute log likelihood of a test trajectory using Viterbi
        and the learned parameters of the model
        @ Frances
        """
        pass


