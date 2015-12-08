import numpy as np
from gridworlds import *
import collections
import scipy.constants as C

class IRLModel:
    def __init__(self, nstates, nactions,
        nrewards, nfeatures, T, gamma, state_features):

        self.nstates = nstates
        self.nactions = nactions
        self.nfeatures = nfeatures # num features for state observations
        self.T = T # transition model
        self.Q = np.zeros([nrewards,nstates, nactions]) # Q-value function (NOT the EM Q fn)
        self.nrewards = nrewards
        self.gamma = gamma
        self.bolztmann = C.k
        # Weights for the linear reward functions
        self.Theta = np.random.rand(nrewards, nfeatures)
        # Weights for the reward transition functions
        self.omega = np.random.rand(nrewards, nrewards, nfeatures)
        # Probabilities for the initial state distribution
        self.nu = np.ones((1, nstates)) / nstates
        # Probabilities for the initial reward function distribution
        self.sigma = np.ones((1, nrewards)) / nrewards
        # function that returns features for a state

        self.states = states

        self.actions = actions

        self.state_features = state_features
        self.initial_state_prob =  collections.defaultdict(int)

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

    def policy(self, rtheta, traj, time):
        """
        TODO: Implement Q-learning @Karthik
        """

        thetaPhi = np.dot(rtheta,state_features[states[traj][time]])

        bigSum = 0

        for s in nstates:
            transition = T[states[traj][time]][actions[traj][time]][s]
            smallSum = 0
            for a in nactions:
                smallSum+=self.Q(rtheta,s,a)*pi(rtheta,s,a)
            bigSum+=transition*smallSum

        Q(rtheta,states[traj][time],actions[traj][time]) = thetaPhi + self.gamma*bigSum

        self.Q = Q

        return Q(rthera,states[traj][time],action[traj][time])

    def pi(self,rtheta,traj,time):

        numerator = np.exp(self.bolztmann*self.Q(rtheta,states[traj][time],actions[traj][time]))

        denominator = 0
        for a in nactions:
            denominator+= np.exp(self.bolztmann*Q(rtheta,states[traj][time],actions[traj][time]))

        return numerator/denominator



    def maximize_nu(self):
        """
        TODO: Maximize the initial state probabilities according to expert trajectories
        @ Daniel/Karthik
        """
        nstates = self.nstates
        trajectories = self.trajectories
        ntrajectories = self.ntrajectories
        nu = collections.defaultdict(int)
        for s in nstates:
            for j in trajectories:
                if j[0][0]= s:
                    nu[s]+=1/ntrajectories

        return nu


    def maximize_sigma(self):
        """
        TODO: Maximize the initial reward function probabilities according to expert trajectories
        @ Daniel/Karthik
        """
        sigma = collections.defaultdict()

        curr_prob = 0
        probSum = 0
        
        for r in nrewards:
            for traj in self.trajectories:
                probSum+=this.BWLearn.ri_given_seq(traj,0,theta[r])

            curr_prob = probSum/ntrajectories

            sigma[s] = curr_prob

        return sigma
        

    def maximize_reward_weights(self):
        """
        TODO: Find the optimal set of weights for the reward functions using gradient ascent
        @ Karthik
        """

        theta = self.Theta

        currThetaCoordinate = 1 #what are good values for initial thetas?
        lastThetaCoordinate = 0

        for r in nrewards:
            bigSum = 0
            while (iter < max_iters and (abs(currThetaCoordinate-lastThetaCoordinate) > tolerance)):
                iter = iter + 1
                lastThetaCoordinate = currThetaCoordinate
                for traj in range(len(self.trajectories)):
                    Tn = len(self.trajectories[traj])
                    smallSum = 0
                    for t in range(Tn):
                        prob = this.BWLearn.ri_given_seq(traj,t,theta[r])
                        dQ = np.sum(self.state_features[states[traj][t]])
                        dPi = np.exp(self.bolztmann*Q(states[traj][t],actions[traj][t]))*self.bolztmann*dQ
                        for aprime in nactions:
                            sumAct+=np.exp(self.bolztmann*Q(states[traj][t],aprime))*self.bolztmann
                        dPi = dPi/sumAct
                        Pi = pi(theta[r],traj,t)
                        smallSum+= prob*dPi/Pi
                    bigSum+=smallSum
                currThetaCoordinate = lastThetaCoordinate + self.delta*bigSum

            theta[r] = currThetaCoordinate

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


