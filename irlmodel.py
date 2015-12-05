import numpy as np
from gridworlds import *

class IRLModel:
    def __init__(self, nstates, nactions,
        nrewards, nfeatures, T, gamma, state_features):

        self.nstates = nstates
        self.nactions = nactions
        self.nfeatures = nfeatures # num features for state observations
        self.T = T # transition model
        self.Q = np.zeros((nstates, nactions)) # Q-value function (NOT the EM Q fn)
        self.nrewards = nrewards
        self.gamma = gamma
        # Weights for the linear reward functions
        self.Theta = np.random.rand(nrewards, nfeatures)
        # Weights for the reward transition functions
        self.omega = np.random.rand(nrewards, nrewards, nfeatures)
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
        the first time step should be a dummy (state, action) pair
        """
        self.trajectories = trajectories

        self.BWLearn = BaumWelch(trajectories, self)



        curr_likelihood = self.training_log_likelihood()
        last_likelihood = curr_likelihood - 1e9
        iter = 0
        while (iter < max_iters and
            (abs(curr_likelihood - last_likelihood) > tolerance)):
            iter = iter + 1
            # Maximize parameters simultaneously
            self.BWLearn.update()

            nu = self.maximize_nu()
            sigma = self.maximize_sigma()
            Theta = self.maximize_reward_weights()
            omega = self.maximize_reward_transitions()
            # Set parameters
            self.nu, self.sigma, self.Theta, self.omega = nu, sigma, Theta, omega
            # Compute likelihoods
            last_likelihood = curr_likelihood
            curr_likelihood = self.training_log_likelihood()

    def tau(self, rtheta1, rtheta2, state):
        """
        Transition function between reward functions.
        """
        state = self.state_features(state)
        num = 0.0
        if rtheta1 == rtheta2:
            num = 1
        else:
            num = np.exp(np.dot(
                self.omega[rtheta1, rtheta2]), state)

        selftransition = np.exp(np.dot(
                self.omega[rtheta1, rtheta1]), state)
        den = (np.sum
            (np.exp(np.dot(self.omega[rtheta1], state))) - selftransition) + 1
        return num / den

    def policy(self, rtheta, state, action):
        """
        TODO: Implement Q-learning @Karthik
        """
        return 1.0 / self.nactions


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
        Returns log likelihood for the training trajectories
        based on the current parameters of the model
        """
        L = 0.0
        for n, traj in enumerate(self.trajectories):
            init_state_prob = np.log(self.nu(traj[1, 0]))
            first_rew_prob = 0.0
            for r in xrange(self.nrewards):
                first_rew_prob += (self.BWLearn.ri_given_seq(n, 1, r) *
                    np.log(self.sigma[r]))
            policy_prob = 0.0
            for t in xrange(1, len(traj)):
                for r in xrange(self.nrewards):
                    policy_prob += (self.BWLearn.ri_given_seq(n, t, r) *
                        np.log(self.policy(r, traj[t, 0], traj[t, 1])))
            reward_transition_prob = 0.0
            for t in xrange(1, len(traj)):
                for rprev in xrange(self.nrewards):
                    for r in xrange(self.nrewards):
                        reward_transition_prob += (self.BWLearn.ri_given_seq2(
                            n, t, rprev, r) * np.log(self.tau(rprev, r, state)))
            transition_prob = 0.0
            for t in xrange(2, len(traj)):
                sprev = traj[t-1, 0]
                aprev = traj[t-1, 1]
                s = traj[t, 0]
                transition_prob += np.log(self.T(sprev, aprev, s))
            L += (init_state_prob + first_rew_prob + policy_prob +
                    reward_transition_prob + transition_prob)

        return L

    def validate(self, trajectories):
        return sum(self.viterbi_log_likelihood(t) for t in trajectories)

    def viterbi_log_likelihood(self, trajectory):
        """
        Compute log likelihood of a test trajectory using Viterbi
        and the learned parameters of the model
        @ Frances
        """
        pass


