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
        self.logalpha = [np.zeros(traj.shape[0]) for traj in trajectories]
        self.beta = [np.zeros((traj.shape)) for traj in trajectories]
        self.logbeta = [np.zeros(traj.shape[0]) for traj in trajectories]
        self.seq_probs = np.zeros(len(trajectories))
        self.prsa = np.zeros((len(trajectories), self.model.nrewards))

    def reset(self):
        trajectories = self.trajectories
        self.alpha = [np.zeros((traj.shape)) for traj in trajectories]
        self.logalpha = [np.zeros(traj.shape[0]) for traj in trajectories]
        self.beta = [np.zeros((traj.shape)) for traj in trajectories]
        self.logbeta = [np.zeros(traj.shape[0]) for traj in trajectories]
        self.seq_probs = np.zeros(len(trajectories))
        self.prsa = np.zeros((len(trajectories), self.model.nrewards))
  
    def update(self):
        """
        Computes forward and backward probabilities
        for each trajectory using the current model parameters.
        """
        self.reset()
        model = self.model

        ntraj = len(self.alpha)

        if self.model.ignore_tau:
            for n in xrange(ntraj):
                self.seq_probs[n] = 0 
                for r in xrange(model.nrewards):
                    self.prsa[n, r] = 0
                    traj = self.trajectories[n]
                    tmax = self.alpha[n].shape[0]
                    for t in xrange(1, tmax):
                        s = traj[t, 0]
                        a = traj[t, 1]
                        self.prsa[n, r] += np.log(model.policy[r, s, a])
                    for t in xrange(1, tmax-1):
                        s = traj[t, 0]
                        a = traj[t, 1]
                        sn = traj[t+1, 0]
                        self.prsa[n, r] += np.log(model.T[s, a, sn])
                    self.prsa[n, r] += np.log(model.sigma[r]) + np.log(model.nu[traj[1, 0]])
                minx = self.prsa[n].min()
                self.seq_probs[n] = np.log(np.sum(np.exp(self.prsa[n] - minx))) + minx
            return

        for n in xrange(ntraj):
            traj = self.trajectories[n]


            # alpha recursion

            for r in xrange(model.nrewards):
                self.alpha[n][(0, r)] = model.sigma[r]

            asum = np.sum(self.alpha[n][0])
            self.logalpha[n][0] = np.log(asum)
            self.alpha[n][0] /= asum


            tmax = self.alpha[n].shape[0]

            # for t = 1
            for r in xrange(model.nrewards):
                for rprev in xrange(model.nrewards):
                    self.alpha[n][(1, r)] += (model.nu[traj[(1, 0)]] *
                        self.tau(r, rprev, traj[(1, 0)]) *
                        model.policy[r, traj[(1, 0)], traj[(1, 1)]] *
                        self.alpha[n][(0, rprev)])


            asum = np.sum(self.alpha[n][1])
            self.logalpha[n][1] = np.log(asum) + self.logalpha[n][0]
            self.alpha[n][1] /= asum


            # for t = 2 to n
            for t in xrange(2, tmax):
                s = traj[(t, 0)]
                sprev = traj[(t-1, 0)]
                a = traj[(t, 1)]
                aprev = traj[(t-1, 1)]
                for r in xrange(model.nrewards):
                    for rprev in xrange(model.nrewards):
                        #print model.T[sprev, aprev, s],model.tau[r, rprev, s],model.policy[r, s, a] ,self.alpha[n][(t-1, rprev)]
                        self.alpha[n][(t, r)] += (model.T[sprev, aprev, s] *
                            self.tau(r, rprev, s) *
                            model.policy[r, s, a] *
                            self.alpha[n][(t-1, rprev)])
                asum = np.sum(self.alpha[n][t])
                self.logalpha[n][t] = np.log(asum) + self.logalpha[n][t-1]
                self.alpha[n][t] /= asum


            # self.beta recursion
            for r in xrange(model.nrewards):
                self.beta[n][(tmax-1, r)] = 1

            bsum = np.sum(self.beta[n][tmax-1])
            self.logbeta[n][tmax-1] = np.log(bsum)
            self.beta[n][tmax-1] /= bsum

            # for t = n-2 to 1
            for t in xrange(tmax-2, 0, -1):
                s = traj[(t, 0)]
                snext = traj[(t+1, 0)]
                a = traj[(t, 1)]
                anext = traj[(t+1, 1)]
                for r in xrange(model.nrewards):
                    for rnext in xrange(model.nrewards):
                        self.beta[n][(t, r)] += (model.T[s, a, snext] *
                            self.tau(rnext, r, snext) *
                            model.policy[rnext, snext, anext] *
                            self.beta[n][(t+1, rnext)])
                bsum = np.sum(self.beta[n][t])
                self.logbeta[n][t] = np.log(bsum) + self.logbeta[n][t+1]
                self.beta[n][t] /= bsum

            # for t = 0
            for r in xrange(model.nrewards):
                for rnext in xrange(model.nrewards):
                    self.beta[n][(0, r)] += (model.nu[traj[(1, 0)]] *
                        self.tau(r, rnext, traj[(1, 0)]) *
                        model.policy[r, traj[(1, 0)], traj[(1, 1)]] *
                        self.beta[n][1, rnext])
            bsum = np.sum(self.beta[n][0])
            self.logbeta[n][0] = np.log(bsum) + self.logbeta[n][1]
            self.beta[n][0] /= bsum

            # Compute sequence probabilities
            self.seq_probs[n] = np.log(np.sum(self.alpha[n][tmax-1, :])) + self.logalpha[n][tmax-1]

    def ri_given_seq(self, seq, time, rtheta):
        """
        Return P(R_i| S, A) for a particular seq
        """
        if self.model.ignore_tau:
            return np.exp(self.prsa[seq, rtheta] - self.seq_probs[seq])

        return np.exp(np.log(self.alpha[seq][(time, rtheta)]) + self.logalpha[seq][time] +
                np.log(self.beta[seq][(time, rtheta)]) + self.logbeta[seq][time] -
                    self.seq_probs[seq])

    def ri_given_seq2(self, seq, time, rthetaprev, rtheta):
        """
        Return P(R_{i-1}, R_i| S, A) for a particular seq
        """

        # print self.seq_probs[seq]
        traj = self.trajectories[seq]
        s = traj[(time, 0)]
        sprev = traj[(time-1, 0)]
        a = traj[(time, 1)]
        aprev = traj[(time-1, 1)]
        model = self.model


        if (model.policy[rtheta, s, a] < 1e-10 or
            model.T[sprev, aprev, s] < 1e-10 or
            self.tau(rthetaprev, rtheta, s) < 1e-10):
            return 0.0

        return np.exp(np.log(self.alpha[seq][time-1, rthetaprev]) +
                (self.logalpha[seq][time-1] if (time > 0) else 0) +
                np.log(self.beta[seq][time, rtheta]) +
                self.logbeta[seq][time] +
                np.log(model.T[sprev, aprev, s]) +
                np.log(self.tau(rthetaprev, rtheta, s)) +
                np.log(model.policy[rtheta, s, a]) -
                self.seq_probs[seq])

    def tau(self, x, y, z):
        if self.model.ignore_tau:
            return 1.0
        else:
            return self.model.tau[x, y, z]


