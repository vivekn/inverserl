import numpy as np

class BaumWelch(object):
    """
    Class for computing forward backward probabilities of
    the IRL graphical model

    Can be used with testing or training trajectories.
    """

    def __init__(self, trajectories, model):
        self.trajectories = trajectories
        self.model = model #IRLModel

        # Initialize data structures
        self.alpha = [np.zeros((traj.shape)) for traj in trajectories]
        self.beta = [np.zeros((traj.shape)) for traj in trajectories]
        self.seq_probs = np.zeros(len(trajectories))

    def update(self):
        """
        Computes forward and backward probabilities
        for each trajectory using the current model parameters.
        """
        model = self.model

        ntraj = len(self.alpha)

        for n in xrange(ntraj):
            traj = self.trajectories[n]


            # alpha recursion

            for r in xrange(model.nrewards):
                self.alpha[n][(0, r)] = model.sigma[r]
            tmax = self.alpha[n].shape[0]

            # for t = 1
            for r in xrange(model.nrewards):
                for rprev in xrange(model.nrewards):
                    self.alpha[n][(1, r)] += (model.sigma[traj[(1, 0)]] *
                        model.tau[r, rprev, traj[(1, 0)]] *
                        model.policy[r, traj[(1, 0)], traj[(1, 1)]])

            # for t = 2 to n
            for t in xrange(2, tmax):
                s = traj[(t, 0)]
                sprev = traj[(t-1, 0)]
                a = traj[(t, 1)]
                aprev = traj[(t-1, 1)]
                for r in xrange(model.nrewards):
                    for rprev in xrange(model.nrewards):
                        self.alpha[n][(t, r)] += (model.T[sprev, aprev, s] *
                            model.tau[r, rprev, s] *
                            model.policy[r, s, a] *
                            self.alpha[n][(t-1, rprev)])


            # self.beta recursion
            for r in xrange(model.nrewards):
                self.beta[n][(tmax-1, r)] = 1

            # for t = n-2 to 1
            for t in xrange(tmax-2, 0, -1):
                s = traj[(t, 0)]
                snext = traj[(t+1, 0)]
                a = traj[(t, 1)]
                anext = traj[(t+1, 1)]
                for r in xrange(model.nrewards):
                    for rnext in xrange(model.nrewards):
                        self.beta[n][(t, r)] += (model.T[s, a, snext] *
                            model.tau[rnext, r, snext] *
                            model.policy[rnext, snext, anext] *
                            self.beta[n][(t+1, rnext)])

            # for t = 0
            for r in xrange(model.nrewards):
                for rnext in xrange(model.nrewards):
                    self.beta[n][(0, r)] += (model.sigma[traj[(1, 0)]] *
                        model.tau[r, rnext, traj[(1, 0)]] *
                        model.policy[r, traj[(1, 0)], traj[(1, 1)]] *
                        self.beta[n][1, rnext])

            # Compute sequence probabilities
            self.seq_probs[n] = np.sum(self.alpha[n][tmax, :])

    def ri_given_seq(self, seq, time, rtheta):
        """
        Return P(R_i| S, A) for a particular seq
        """
        return (self.alpha[seq][(time, rtheta)] * self.beta[seq][(time, rtheta)] /
                    self.seq_probs[seq])

    def ri_given_seq2(self, seq, time, rthetaprev, rtheta):
        """
        Return P(R_{i-1}, R_i| S, A) for a particular seq
        """
        traj = self.trajectories[seq]
        s = traj[(time, 0)]
        sprev = traj[(time-1, 0)]
        a = traj[(time, 1)]
        aprev = traj[(time-1, 1)]
        model = self.model

        return (self.alpha[seq][(time-1, rthetaprev)] *
                self.beta[seq][(time, rtheta)] *
                model.T[sprev, aprev, s] * model.tau[rthetaprev, rtheta, s] *
                model.policy[rtheta, s, a] / self.seq_probs[seq])



