import numpy as np
from irlmodel import IRLModel

def sample(probs):
    return np.random.choice(probs.shape[0], 1, p=probs)[0]

class Simulator:
    def __init__(self, model, nu, T, sigma, theta, omega=None, tau=None):
        self.model = model
        if omega != None:
            self.model.set_parameters(nu, T, sigma, theta, omega=omega)
            self.model.precompute_tau_dynamic()
        elif tau != None:
            self.model.set_parameters(nu, T, sigma, theta, tau=tau)
        else:
            assert False, "Invalid arguments"

    def next(self, state, reward):
        """
        Input = current state and reward function
        Output = next state, current action, next reward function
        """
        model = self.model
        action = sample(model.policy[reward][state])
        next_state = sample(model.T[state][action])
        next_reward = sample(model.tau[reward, :, next_state])
        return (next_state, action, next_reward)

    def trajectories(self, N, goal_state, tmax):
        all = []
        model = self.model
        for i in xrange(N):
            traj = [(0, 0)]
            reward = sample(model.sigma)
            state = sample(model.nu)
            reward = sample(model.tau[reward, :, state])

            while state != goal_state and len(traj) < tmax:
                next_state, action, next_reward = self.next(state, reward)
                traj.append((state, action))
                state, reward = next_state, next_reward

            all.append(np.array(traj))
        return all

def getT():
    """
    State mapping (x, y) -> 5 * x + y
    actions (down, up, right, left)
    """
    T = np.zeros((25, 4, 25))
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    for x in xrange(5):
        for y in xrange(5):
            i = 5*x + y
            reachable = [0]*4
            s = 0
            for j in xrange(4):
                if  0 <= (x+dx[j]) < 5 and 0 <= (y+dy[j]) < 5:
                    reachable[j] = 1
                    s += 1
            for a in xrange(4):
                for b in xrange(4):
                    if reachable[b]:
                        k = 5*(x+dx[b]) + (y+dy[b])
                        T[i][a][k] += 0.85 if a == b else 0.05
                    else:
                        T[i][a][i] += 0.85 if a == b else 0.05
    return T

def tau_EM():
    """
    MLIRL - Littman's paper - Benchmark
    """
    tau = np.zeros((2, 2, 25))

    for r1 in xrange(2):
        for s in xrange(25):
            tau[r1][r1][s] = 1.0
    return tau


def worldA():
    T = getT()
    state_features = np.zeros((25, 3))
    for x in xrange(5):
        for y in xrange(5):
            state_features[5*x + y] = np.array([0,1,0])

    for x in xrange(5):
        for y in xrange(5):
            if x == 0 or y == 0 or x == 4 or y == 4:
                state_features[5*x+y] = np.array([1,0,0])
    state_features[5*2+2] = np.array([0,0,0])
    state_features[5*2+4] = np.array([1,0,1])

    modelA = IRLModel(25, 4, 2, 3, T, 0.95, 0.1, state_features)
    modelA_IRL = IRLModel(25, 4, 2, 3, T, 0.95, 0.1, state_features)
    modelA_EM = IRLModel(25, 4, 2, 3, T, 0.95, 0.1, state_features)
    modelA_EM.set_tau(tau_EM())

    nu = np.zeros(25)
    nu[5*2+0] = 1.0
    sigma = np.array([0.5, 0.5])
    Theta = np.array([[0, 20, 30],
                      [20, 0, 30]])
    tau = np.zeros((2, 2, 25))

    for i in xrange(25):
        tau[0][0][i] = tau[1][1][i] = 1.0

    for i in [(5*2), (5*2+2)]:
        tau[0][1][i] = 0.7
        tau[0][0][i] = 0.3
        tau[1][0][i] = tau[1][1][i] = 0.5

    simA = Simulator(modelA, nu, T, sigma, Theta, tau=tau)
    trajectories = simA.trajectories(50, 2*5+4, 20)
    trajectories_test = simA.trajectories(4, 2*5+4, 20)
    modelA_IRL.learn(trajectories, 1e-3, 5)
    modelA_IRL.test(trajectories_test)
    print "MLIRL"
    modelA_EM.learn(trajectories, 1e-3, 5)
    modelA_EM.test(trajectories_test)
    print "Expert"
    modelA.test(trajectories_test)

def worldB():
    T = getT()
    state_features = np.zeros((25, 2))
    dynamic_features = np.zeros((25,2))

    for x in xrange(5):
        for y in xrange(5):
            state_features[5*x + y] = np.array([0,0])
            dynamic_features[5*x+ y] = np.array([0,1])

    state_features[5*4+0] = np.array([0,1])
    state_features[5*0+4] = np.array([1,0])

    dynamic_features[5*2+0] = np.array([1,1])
    dynamic_features[5*2+1] = np.array([1,1])
    dynamic_features[5*2+2] = np.array([1,1])
    dynamic_features[5*3+2] = np.array([1,1])
    dynamic_features[5*4+2] = np.array([1,1])


    modelB = IRLModel(25, 4, 2, 2, T, 0.95, 0.1, state_features, dynamic_features=dynamic_features)
    modelB_IRL = IRLModel(25, 4, 2, 2, T, 0.95, 0.1, state_features,dynamic_features=dynamic_features)
    modelB_EM = IRLModel(25, 4, 2, 2, T, 0.95, 0.1, state_features)
    modelB_EM.set_tau(tau_EM())


    nu = np.zeros(25)
    nu[5*2+0] = 1.0
    sigma = np.array([0.5, 0.5])
    Theta = np.array([[30,0],
                      [0, 30]])
    omega = np.zeros([2,2,2])
    omega[0,0] = np.array([-11,12])
    omega[1,0] = np.array([13,-12])

    simB = Simulator(modelB, nu, T, sigma, Theta, omega=omega)
    trajectories = simB.trajectories(50, 2*5+4, 60)
    modelB_IRL.learn(trajectories, 1e-3, 10)

    modelB_IRL.test(trajectories_test)
    print "MLIRL"
    modelB_EM.learn(trajectories, 1e-3, 5)
    modelB_EM.test(trajectories_test)
    print "Expert"
    modelB.test(trajectories_test)





worldB()
#worldA()


