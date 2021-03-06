import numpy as np
import collections
import random
from baum_welch import BaumWelch
import time
np.seterr(all='raise')


def stable_boltzmann(beta, numerator, denominator):
    """
    Computes exp(beta * num) / sum_i exp(beta * den_i)
    """
    minx = min(numerator, denominator.min())
    new_num = beta * (numerator - minx)
    new_den = beta * (denominator - minx)
    return np.exp(new_num) / np.sum(np.exp(new_den))


class IRLModel:
    def __init__(self, nstates, nactions,
        nrewards, nfeatures, T, gamma, delta, state_features,
        dynamic_features=None, ignore_tau=False):

        self.nstates = nstates
        self.nactions = nactions
        self.nfeatures = nfeatures # num features for state observations
        self.T = T # transition model
        self.Q = np.zeros([nrewards,nstates, nactions]) # Q-value function (NOT the EM Q fn)
        self.nrewards = nrewards
        self.gamma = gamma
        self.boltzmann = 0.25
        # Weights for the linear reward functions
        self.Theta = np.random.rand(nrewards, nfeatures) - 0.5
        # Weights for the reward transition functions
        # Probabilities for the initial state distribution
        self.nu = np.ones(nstates) / nstates
        # Probabilities for the initial reward function distribution
        self.sigma = np.ones(nrewards) / nrewards
        # function that returns features for a state
        self.delta = delta
        self.static_train = False
        self.ignore_tau = ignore_tau

        self.tau = np.random.rand(self.nrewards, self.nrewards, self.nstates)
        for r in xrange(self.nrewards):
            for s in xrange(self.nstates):
                self.tau[r, :, s] /= np.sum(self.tau[r, :, s])

        self.state_features = state_features
        self.dynamic_features = dynamic_features
        if dynamic_features != None:
            self.ndynfeatures = dynamic_features.shape[1]
            self.omega = np.random.rand(nrewards, nrewards, self.ndynfeatures) - 0.5
        else:
            self.ndynfeatures = 0

        #Initialize policy
        self.policy = np.zeros((nrewards, nstates, nactions))
        for r in xrange(nrewards):
            self.policy[r], _ = self.gradient_pi(self.Theta[r])


    def set_parameters(self, nu, T, sigma, theta, omega=None,tau=None):
        """
        Sets true parameters of the model
        (for trajectory generation)
        """
        self.nu = nu
        self.T = T
        self.sigma = sigma
        if omega != None:
            self.omega = omega
        self.Theta = theta
        if tau != None:
            self.tau = tau
            self.static_train = True
            self.normalize_tau()

        for r in xrange(self.nrewards):
            pi, _ = self.gradient_pi(self.Theta[r])
            self.policy[r] = pi

    def set_tau(self, tau):
        self.tau = tau
        self.normalize_tau()
        self.static_train = True

    def test(self, test_traj):
        testBW = BaumWelch(test_traj, self)
        testBW.update()
        k =  np.mean(testBW.seq_probs)
        print k
        return k

    def normalize_tau(self):
        """
        Make transitions smooth
        """
        eps = 1e-8
        for r in xrange(self.nrewards):
            for s in xrange(self.nstates):
                self.tau[r, :, s] += eps
                self.tau[r, :, s] /= np.sum(self.tau[r, :, s])

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
        iter = 0

        self.nu = self.maximize_nu()


        self.BWLearn.update()
        curr_likelihood = self.training_log_likelihood()
        last_likelihood = curr_likelihood - 1e9

        while (iter < max_iters and
            (abs(curr_likelihood - last_likelihood) > tolerance)):
            iter = iter + 1
            print iter
            # Maximize parameters simultaneously
            omega = None
            self.BWLearn.update()

            if self.ndynfeatures > 0:
                self.precompute_tau_dynamic()
                omega = self.maximize_reward_transitions()

            sigma =  self.maximize_sigma()

            Theta = self.maximize_reward_weights()
            if self.ndynfeatures == 0:
                self.precompute_tau_static()

            # Set parameters
            self.sigma, self.Theta = sigma, Theta
            if omega != None:
                self.omega = omega
            # Compute likelihoods

            last_likelihood = curr_likelihood
            curr_likelihood = self.training_log_likelihood()

        print "EM done"

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
                self.omega[rtheta1, rtheta2], state))

        selftransition = np.exp(np.dot(
                self.omega[rtheta1, rtheta1], state))

        den = (np.sum
                (np.exp(np.dot(self.omega[rtheta1], state))) - selftransition) + 1

        return (num/den)

    def precompute_tau_dynamic(self):
        if self.static_train:
            return
        for s in xrange(self.nstates):
            for r1 in xrange(self.nrewards):
                for r2 in xrange(self.nrewards):
                    self.tau[r1, r2, s] = self.tau_helper(r1, r2, s)
        #self.normalize_tau()


    def precompute_tau_static(self):
        """
        This is very slow, need to optimize it
        """
        if self.static_train:
            return
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
                    if denominator > 0:
                        self.tau[r1, r2, s] = numerator / denominator



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
        sigma = np.copy(self.sigma)

        for r in xrange(self.nrewards):
            probSum = 0
            for n, traj in enumerate(self.trajectories):
                probSum+=self.BWLearn.ri_given_seq(n,0,r)

            sigma[r] = probSum
        S = np.sum(sigma)
        return sigma / S


    def maximize_reward_weights(self, max_iters=20, tolerance = 0.01):
        """
        Find the optimal set of weights for the reward functions using gradient ascent
        """
        curr_magnitude = 0
        last_magnitude = 1e9
        iter = 0
        Theta = np.copy(self.Theta)
        policy = np.zeros((self.nrewards, self.nstates, self.nactions))

        while (iter < max_iters):# and
           # (abs(curr_magnitude - last_magnitude) > tolerance)):
            iter = iter + 1

            for r in xrange(self.nrewards):
                # Q learning
                pi, gradPi = self.gradient_pi(Theta[r])
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
        return Theta



    def gradient_pi(self, theta_r, iters=20):
        """
        Returns pi(s, a) matrix for reward function rtheta.
        Also returns the gradient of pi, uses a Q learning like
        method to compute the gradient.
        """
        pi = np.zeros((self.nstates, self.nactions))
        gradPi = np.zeros((self.nstates, self.nactions, self.nfeatures))

        # Initialize values
        V = np.dot(self.state_features, theta_r)
        Q = np.tile(V.T, [self.nactions, 1]).T
        gradV = np.copy(self.state_features)
        gradQ = np.zeros((self.nstates, self.nactions, self.nfeatures))

        SB = lambda n, d: stable_boltzmann(self.boltzmann, n, d)

        for iter in xrange(iters):
            r_s = np.dot(self.state_features, theta_r)
            for s in xrange(self.nstates):
                for a in xrange(self.nactions):
                    Q[s, a] = r_s[s] + self.gamma * np.dot(self.T[s, a], V).T


            for s in xrange(self.nstates):
                for a in xrange(self.nactions):
                    gradQ[s, a] = self.state_features[s] + self.gamma * np.dot(self.T[s, a], gradV)

            for s in xrange(self.nstates):
                for a in xrange(self.nactions):
                    pi[s, a] = SB(Q[s, a], Q[s])

            for s in xrange(self.nstates):
                for a in xrange(self.nactions):
                    term1 = self.boltzmann * SB(Q[s, a], Q[s]) * gradQ[s, a]
                    term3 = 0
                    for a2 in xrange(self.nactions):
                        term3 += SB(Q[s, a2], Q[s]) * gradQ[s, a2]
                    term2 = self.boltzmann * SB(Q[s, a], Q[s]) * term3
                    gradPi[s, a] = term1 - term2

            V = np.sum(pi*Q, 1)

            for s in xrange(self.nstates):
                gradV[s] = np.dot(Q[s], gradPi[s]) + np.dot(pi[s], gradQ[s])



        return (pi, gradPi)

    def gradient_tau(self, iters=100):

        gradTau = np.zeros((self.nrewards,self.nrewards,self.nstates,self.ndynfeatures))

        for r1 in xrange(self.nrewards):
            for r2 in xrange(self.nrewards):
                for s in xrange(self.nstates):
                    for f in xrange(self.ndynfeatures):
                        if r1==r2:
                            gradTau[r1][r2][s][f] = 0
                        else:
                            num = np.exp(np.dot(self.omega[r1][r2],self.dynamic_features[s]))*self.dynamic_features[s, f]
                            #import pdb; pdb.set_trace()
                            selftransition = np.exp(np.dot(self.omega[r1, r1], self.dynamic_features[s]))
                            den = (np.sum(np.exp(np.dot(self.omega[r1], s))) - selftransition) + 1
                            gradTau[r1][r2][s][f] = num/den

        return gradTau


    def maximize_reward_transitions(self, delta=1e-5, max_iters=20, tolerance = 0.01):
        """
        TODO: Find the optimal set of weights for the reward transitions
        @ Daniel
        """

        omega = np.copy(self.omega)

        curr_magnitude = 0
        last_magnitude = -1e9
        iter = 0
        timeout = time.time() + 10

        while (iter < max_iters):# and (abs(curr_magnitude-last_magnitude) > tolerance and time.time()<timeout)):
            iter = iter + 1
            dTau = self.gradient_tau()
            for r1 in xrange(self.nrewards):
                for r2 in xrange(self.nrewards):
                    bigSum = 0
                    for traj in range(len(self.trajectories)):
                        Tn = len(self.trajectories[traj])
                        smallSum = 0
                        for t in range(Tn):
                            smallerSum = 0
                            for r in xrange(self.nrewards):
                                prob = self.BWLearn.ri_given_seq2(traj,t,r1,r2)
                                tau = self.tau_helper(r1,r2,self.trajectories[traj][t, 0])
                                smallerSum+=prob*dTau[r1,r2,self.trajectories[traj][t,0],:]/tau
                            smallSum+=smallerSum
                        bigSum+=smallSum
                    omega[r1][r2] +=delta*bigSum


            last_magnitude = curr_magnitude
            curr_magnitude = np.sum(np.abs(omega))

        return omega


    def training_log_likelihood(self):
        """
        Returns log likelihood for the training trajectories
        based on the current parameters of the model
        """
        L = 0.0
        nu_ctr = 0
        sigma_ctr = 0
        policy_ctr = 0
        reward_ctr = 0
        transition_ctr = 0
        for n, traj in enumerate(self.trajectories):
            #import pdb; pdb.set_trace()
            init_state_prob = np.log(self.nu[traj[1, 0]])
            first_rew_prob = 0.0
            for r in xrange(self.nrewards):
                #import pdb; pdb.set_trace()
                first_rew_prob += (self.BWLearn.ri_given_seq(n, 0, r) *
                    np.log(self.sigma[r]))
            policy_prob = 0.0
            for t in xrange(1, len(traj)):
                for r in xrange(self.nrewards):
                    policy_prob += (self.BWLearn.ri_given_seq(n, t, r) *
                        np.log(self.policy[r, traj[t, 0], traj[t, 1]]))
            reward_transition_prob = 0.0
            if not self.ignore_tau:
                for t in xrange(2, len(traj)):
                    state = traj[t-1, 0]
                    for rprev in xrange(self.nrewards):
                        for r in xrange(self.nrewards):
                            reward_transition_prob += (self.BWLearn.ri_given_seq2(
                                n, t, rprev, r) * np.log(self.tau[rprev, r, state]))
            transition_prob = 0.0
            for t in xrange(2, len(traj)):
                sprev = traj[t-1, 0]
                aprev = traj[t-1, 1]
                s = traj[t, 0]
                transition_prob += np.log(self.T[sprev, aprev, s])
            L += (init_state_prob + first_rew_prob + policy_prob +
                    reward_transition_prob + transition_prob)
            nu_ctr += init_state_prob
            sigma_ctr += first_rew_prob
            policy_ctr += policy_prob
            reward_ctr += reward_transition_prob
            transition_ctr += transition_prob

        print nu_ctr, sigma_ctr, policy_ctr, reward_ctr, transition_ctr
        print L

        return L

    def validate(self, trajectories):
        return sum(self.viterbi_log_likelihood(t) for t in trajectories)

    def viterbi_log_likelihood(self, trajectory):
        """
        Compute log likelihood of a test trajectory using Viterbi
        and the learned parameters of the model
        @ Frances
        """
        init_r_list = [0 for i in xrange(T+1)]
        T = len(trajectory)
        cur_max = -1
        cur_argmax = None
        for reward in self.Theta:
            res, r_list = viterbi_recursive(trajectory, reward, T, init_r_list)
            if res > cur_max:
                cur_max = res
                r_list[T] = reward
                cur_argmax = r_list

        return cur_argmax

    def viterbi_recursive(self, trajectory, r_theta, T, r_list):
        cur_max = -1
        cur_argmax = None

        if T == 1:
            nu_pi = self.nu(trajectory[1][0])*self.policy(trajectory[1][0], trajectory[1][1])
            for reward in self.Theta:
                res, r_list = self.sigma(r_theta)*tau_helper(reward, trajectory[1][0], r_theta)
                if res > cur_max:
                    cur_max = res
                    r_list[T] = reward
                    cur_argmax = r_list
            return nu_pi*cur_max, cur_argmax
        else:
            # t*pi
            t_pi = self.T(trajectory[T-1][0], trajectory[T-1][1], trajectory[T][0])*self.policy(r_theta, trajectory[T][0], trajectory[T][1])

            #max among rewards
            for reward in self.Theta:
                res, r_list = viterbi_recursive(trajectory, reward, T-1)*tau_helper(reward, trajectory[T][0], r_theta)
                if res > cur_max:
                    cur_max = res
                    r_list[T] = reward
                    cur_argmax = r_list
            return t_pi*cur_max, cur_argmax


def test():
    T = np.random.rand(2, 3, 2)
    for i in xrange(2):
        for j in xrange(3):
            s = np.sum(T[i, j])
            T[i, j] = T[i, j] / s
    state_features = np.random.rand(2, 3)
    dynamic_features = np.random.rand(2, 4)
    testModel = IRLModel(2, 3, 2, 3, T, 0.95, 0.01, state_features)
    trajectories = []
    for i in xrange(100):
        Tn = np.random.randint(5, 101)
        traj = np.zeros((Tn+1, 2))
        for t in xrange(1, Tn+1):
            traj[t, 0] = np.random.randint(0, 2)
            traj[t, 1] = np.random.randint(0, 2)
        trajectories.append(traj)
    testModel.learn(trajectories, 1e-3, 10)

if __name__ == "__main__":
    test()


