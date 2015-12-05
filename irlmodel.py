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

        # Initialize data structures
        self.alpha = [np.zeros((traj.shape)) for traj in trajectories]
        self.beta = [np.zeros((traj.shape)) for traj in trajectories]
        self.seq_probs = np.zeros((1, len(trajectories)))



        curr_likelihood = self.training_log_likelihood()
        last_likelihood = curr_likelihood - 1e9
        iter = 0
        while (iter < max_iters and
            (abs(curr_likelihood - last_likelihood) > tolerance)):
            iter = iter + 1
            # Maximize parameters simultaneously
            self.baum_welch()

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

    def baum_welch(self):
        """
        Computes forward and backward probabilities
        for each trajectory using the current model parameters.
        """

        ntraj = len(self.alpha)

        for n in xrange(ntraj):
            traj = self.trajectories[n]

            # alpha recursion

            for r in xrange(self.nrewards):
                self.alpha[n][(0, r)] = self.sigma[r]
            tmax = self.alpha[n].shape[0]

            # for t = 1
            for r in xrange(self.nrewards):
                for rprev in xrange(self.nrewards):
                    self.alpha[n][(1, r)] += (self.sigma[traj[(1, 0)]] *
                        self.tau(r, rprev, traj[(1, 0)]) *
                        self.policy(r, traj[(1, 0)], traj[(1, 1)]))

            # for t = 2 to n
            for t in xrange(2, tmax):
                s = traj[(t, 0)]
                sprev = traj[(t-1, 0)]
                a = traj[(t, 1)]
                aprev = traj[(t-1, 1)]
                for r in xrange(self.nrewards):
                    for rprev in xrange(self.nrewards):
                        self.alpha[n][(t, r)] += (self.T(sprev, aprev, s) *
                            self.tau(r, rprev, s) *
                            self.policy(r, s, a) *
                            self.alpha[n][(t-1, rprev)])


            # beta recursion
            for r in xrange(self.nrewards):
                self.beta[n][(tmax-1, r)] = 1

            # for t = n-2 to 1
            for t in xrange(tmax-2, 0, -1):
                s = traj[(t, 0)]
                snext = traj[(t+1, 0)]
                a = traj[(t, 1)]
                anext = traj[(t+1, 1)]
                for r in xrange(self.nrewards):
                    for rnext in xrange(self.nrewards):
                        self.beta[n][(t, r)] += (self.T(s, a, snext) *
                            self.tau(rnext, r, snext) *
                            self.policy(rnext, snext, anext) *
                            self.beta[n][(t+1, rnext)])

            # for t = 0
            for r in xrange(self.nrewards):
                for rnext in xrange(self.nrewards):
                    self.beta[n][(0, r)] += (self.sigma[traj[(1, 0)]] *
                        self.tau(r, rnext, traj[(1, 0)]) *
                        self.policy(r, traj[(1, 0)], traj[(1, 1)]) *
                        self.beta[n][1, rnext])

            # Compute sequence probabilities
            self.seq_probs[n] = np.sum(self.alpha[n][tmax, :])

    def ri_given_seq(self, seq, time, rtheta):
        return (self.alpha[seq][(time, rtheta)] * self.beta[seq][(time, rtheta)] /
                    self.seq_probs[seq])





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
        init_state_prob = np.sum(np.log(self.nu))
        #TODO Implement Baum Welch algorithm : init_reward_prob =


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


