import numpy as np
import collections
from baum_welch import BaumWelch
from numba import jit

class IRLModel:
    def __init__(self, nstates, nactions,
        nrewards, nfeatures, T, stateTransition, gamma, delta, state_features,
        dynamic_features=None):

        self.nstates = nstates
        self.nactions = nactions
        self.nfeatures = nfeatures # num features for state observations
        self.T = T # transition model
        self.stateTransition = stateTransition
        self.Q = np.zeros([nrewards,nstates, nactions]) # Q-value function (NOT the EM Q fn)
        self.nrewards = nrewards
        self.gamma = gamma
        self.boltzmann = 0.5
        # Weights for the linear reward functions
        self.Theta = np.random.rand(nrewards, nfeatures)
        # Weights for the reward transition functions
        self.omega = np.random.rand(nrewards, nrewards, nfeatures)
        # Probabilities for the initial state distribution
        self.nu = np.ones((1, nstates)) / nstates
        # Probabilities for the initial reward function distribution
        self.sigma = np.ones((1, nrewards)) / nrewards
        # function that returns features for a state
        self.delta = delta

        self.tau = np.zeros(self.nrewards, self.nrewards, self.nstates)

        self.state_features = state_features
        self.dynamic_features = dynamic_features
        if dynamic_features:
            self.ndynfeatures = dynamic_features.shape[1]
        else:
            self.ndynfeatures = 0

        self.initial_state_prob =  collections.defaultdict(int)

        #Initialize policy
        self.policy = np.zeros(nrewards, nstates, nactions)
        for r in xrange(nrewards):
            self.policy[r], _ = self.gradient_pi(r)

    def learn(self, trajectories, tolerance, max_iters):
        """
        Learns parameters by using EM.
        A trajectory is a sequence of (state, action) tuples
        the first time step should be a dummy (state, action) pair
        """
        self.trajectories = trajectories
        self.ntrajectories = len(trajectories)
        self.BWLearn = BaumWelch(trajectories, self)
        self.max_iters = max_iters
        self.tolerance = tolerance
        curr_likelihood = self.training_log_likelihood()
        last_likelihood = curr_likelihood - 1e9
        iter = 0

        self.nu = self.maximize_nu()

        while (iter < max_iters and
            (abs(curr_likelihood - last_likelihood) > tolerance)):
            iter = iter + 1
            # Maximize parameters simultaneously
            omega = None
            self.BWLearn.update()
            if self.ndynfeatures > 0:
                self.precompute_tau_dynamic()
                omega = self.maximize_reward_transitions()

            sigma = self.maximize_sigma()
            Theta = self.maximize_reward_weights()
            if self.ndynfeatures == 0:
                self.precompute_tau_static()

            # Set parameters
            self.sigma, self.Theta = sigma, Theta
            if omega:
                self.omega = omega
            # Compute likelihoods
            last_likelihood = curr_likelihood
            curr_likelihood = self.training_log_likelihood()

    def tau_helper(self, rtheta1, rtheta2, s):
        """
        Transition function between reward functions.
        """

        state = self.dynamic_features[s]
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

    def precompute_tau_dynamic(self):
        for s in xrange(self.nstates):
            for r1 in xrange(self.nrewards):
                for r2 in xrange(self.nrewards):
                    self.tau[r1, r2, s] = self.tau_helper(r1, r2, s)

    @jit
    def precompute_tau_static(self):
        """
        This is very slow, need to optimize it
        """
        for s in xrange(self.nstates):
            for r1 in xrange(self.nrewards):
                for r2 in xrange(self.nrewards):
                    numerator = 0
                    denominator = 0
                    for n in xrange(len(self.trajectories)):
                        Tn = len(self.trajectories[n])
                        for t in xrange(1, Tn):
                            numerator += self.BWLearn.ri_given_seq2(n, t, r1, r2)
                    for r3 in xrange(self.nrewards):
                        for n in xrange(len(self.trajectories)):
                            Tn = len(self.trajectories[n])
                            for t in xrange(1, Tn):
                                denominator += self.BWLearn.ri_given_seq2(n, t, r1, r3)
                    self.tau[r1, r2, s] = numerator / denominator

    def pi(self,rtheta,traj,time):

        numerator = np.exp(self.boltzmann*self.Q(rtheta,traj[time][0],traj[time][1]))

        denominator = 0
        for a in self.nactions:
            denominator+= np.exp(self.boltzmann*self.Q(rtheta,traj[time][0],traj[time][1]))

        return numerator/denominator



    def maximize_nu(self):
        """
        Maximize the initial state probabilities according to expert trajectories
        """
        nstates = self.nstates
        trajectories = self.trajectories
        ntrajectories = self.ntrajectories
        nu = np.zeros(nstates)
        for s in xrange(nstates):
            for j in trajectories:
                if j[1][0] == s:
                    nu[s]+=1.0/ntrajectories

        return nu


    def maximize_sigma(self):
        """
        Maximize the initial reward function probabilities according to expert trajectories
        """
        sigma = np.zeros(self.nstates)

        curr_prob = 0
        probSum = 0

        for r in self.nrewards:
            for traj in self.trajectories:
                probSum+=self.BWLearn.ri_given_seq(traj,0,self.Theta[r])

            curr_prob = probSum/len(self.trajectories)

            sigma[r] = curr_prob

        return sigma


    def maximize_reward_weights(self, max_iters=100, tolerance = 0.01):
        """
        Find the optimal set of weights for the reward functions using gradient ascent
        """
        curr_magnitude = 0
        last_magnitude = 1e9
        iter = 0
        Theta = np.copy(self.Theta)
        policy = np.zeros(self.nrewards, self.nstates, self.nactions)

        while (iter < max_iters and
            (abs(curr_magnitude - last_magnitude) > tolerance)):
            iter = iter + 1

            for r in xrange(self.nrewards):
                # Q learning
                pi, gradPi = self.gradient_pi(r)
                policy[r] = pi
                gradTheta = np.zeros(self.nfeatures)
                for n, traj in enumerate(self.trajectories):
                    Tn = len(traj)
                    for t in xrange(1, Tn):
                        s = traj[t, 0]
                        a = traj[t, 1]
                        prob = self.BWLearn.ri_given_seq(n, t, r)
                        gradTheta += prob * gradPi[s, a] / pi[s, a]

                # Set parameters
                Theta[r] = Theta[r] + self.delta * gradTheta

            # Compute magnitudes
            last_magnitude = curr_magnitude
            curr_magnitude = np.sum(np.abs(Theta))
        self.policy = policy



    def gradient_pi(self, rtheta, iters=100):
        """
        Returns pi(s, a) matrix for reward function rtheta.
        Also returns the gradient of pi, uses a Q learning like
        method to compute the gradient.
        """
        pi = np.zeros((self.nstates, self.nactions))
        gradPi = np.zeros((self.nstates, self.nactions, self.nfeatures))

        # Initialize values
        V = np.dot(self.state_features, self.Theta[rtheta])
        Q = np.tile(V, self.nactions)
        gradV = np.copy(self.state_features)
        gradZ = np.zeros(self.nstates, self.nfeatures)

        for iter in xrange(iters):
            Q = (np.tile(np.dot(self.state_features, self.Theta[rtheta]), self.nactions) +
                    self.gamma * np.tensordot(self.T, V))
            #gradQ s*f*a tensor
            gradQ = (np.swapaxes(np.tile(self.state_features, self.nactions), 1, 2)
                        + self.gamma * np.tensordot(self.T, self.state_features))

<<<<<<< HEAD
        omega = self.omega

        currOmegaCoordinate = 1
        lastOmegaCoordinate = 0 #what are good valyes for intial omegas?

        for r1 in nrewards:
            for r2 in nrewards:
                bigSum = 0
                while (iter < max_iters and (abs(currOmegaCoordinate-lastOmegaCoordinate) > tolerance)):
                    iter = iter + 1
                    lastOmegaCoordinate = currOmegaCoordinate
                    for traj in range(len(self.trajectories)):
                        Tn = len(self.trajectories[traj])
                        smallSum = 0
                        for t in range(Tn):
                            smallerSum = 0
                            for r in nrewards:
                                prob = this.BWLearn.ri_given_seq2(traj,t,self.Theta[r1],self.Theta[r2])
                                tau = tau(self.Theta[r1],self.Theta[r2],traj,t)
                                dTau = 
                                smallerSum+=prob*dTau/tau
                            smallSum+=smallerSum
                        bigSum+=smallSum
                    currOmegaCoordinate = lastOmegaCoordinate + self.delta*bigSum
                omega[r1][r2] = #TODO


=======
            Z = np.sum(np.exp(self.boltzmann * Q), 1)
            for s in xrange(self.nstates):
                gradZ[s] = self.boltzmann * np.dot(np.exp(self.boltzmann * Q[s]), gradQ[s])
>>>>>>> 905489d4a881f62dbbdd1016722144ad5e70a252

            for s in xrange(self.nstates):
                pi[s] = np.exp(self.boltzmann * Q[s]) / Z[s]

            for s in xrange(self.nstates):
                for a in xrange(self.nactions):
                    temp = np.exp(self.boltzmann*Q[s, a])
                    gradPi[s, a] = (self.boltzmann * Z[s] * temp * gradQ[s, a] - temp * gradZ[s]) / (Z[s] ** 2)

            V = pi * Q
            for s in xrange(self.nstates):
                gradV[s] = np.dot(Q[s], gradPi[s, a]) + np.dot(pi[s], gradQ[s, a])

        return (pi, gradPi)


    def maximize_reward_transitions(self):
        """
        TODO: Find the optimal set of weights for the reward transitions
        @ Daniel
        """


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
                        np.log(self.policy[r, traj[t, 0], traj[t, 1]]))
            reward_transition_prob = 0.0
            for t in xrange(2, len(traj)):
                state = traj[t-1, 0]
                for rprev in xrange(self.nrewards):
                    for r in xrange(self.nrewards):
                        reward_transition_prob += (self.BWLearn.ri_given_seq2(
                            n, t, rprev, r) * np.log(self.tau(rprev, r, state)))
            transition_prob = 0.0
            for t in xrange(2, len(traj)):
                sprev = traj[t-1, 0]
                aprev = traj[t-1, 1]
                s = traj[t, 0]
                transition_prob += np.log(self.T[sprev, aprev, s])
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


