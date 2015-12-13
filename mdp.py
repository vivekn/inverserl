from utils import *
import numpy
import pprint
import gridA
import gridB
import random

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, init, actlist, terminals, gamma=.9):
        update(self, init=init, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        abstract

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""
    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse() ## because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            return [(1.0, self.go(state, action))]
            # return [(0.8, self.go(state, action)),
            #         (0.1, self.go(state, turn_right(action))),
            #         (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states, state1, state)

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x,y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))

def value_iteration(mdp, epsilon=0.001):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
             return U

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])

def policy_iteration(mdp):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, random.choice(mdp.actions(s))) for s in mdp.states])
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a,s,U,mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def policy_evaluation(pi, U, mdp, k=500):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s] for (p, s1) in T(s, pi[s])])
    return U


#lambda = (v, sigma, thetas, tau_w)
#experts = expert trajectories: list of state: action dictionaries
def calculate_log_likelihood(lamda, experts):
    n_tot = len(experts)
    tot_sum = 0;
    for i in range(n_tot):
        tot_sum += traj_likelihood(experts[i])

    return tot_sum/n_tot

def traj_likelihood(big_lambda, expert):
    v = big_lambda['v']
    sigma = big_lambda['sigma']
    thetas = big_lambda['thetas']
    tau_w = big_lambda['tau_w']
    
    num_rewards = len(thetas)
    traj_length = len(expert)
    for i in range(traj_length):
        for theta in thetas:
            for theta_p in thetas:

            t()




# S=possible states 
# A=possible actions
# t=state transition function
# r_theta=reward function
# gamma=discount for future rewards
# v=initial state probabililties 
# sigma=initial reward function probabilities
# thetas=set of reward weights
# tau_w=reward transition function
def irl_mdp(S, A, t, r_theta, gamma, v, sigma, thetas, tau_w):
    rand1 = random.random()

if __name__ == '__main__':
      # my_mdp = GridMDP([[-0.04, -0.04, -0.04, +1],
    #                  [-0.04, None,  -0.04, -1],
    #                  [-0.04, -0.04, -0.04, -0.04]],
    #                 terminals=[(3, 2), (3, 1)],)
    # pi_my_mdp = policy_iteration(my_mdp)
    # pprint.pprint(pi_my_mdp)


    land_vals = [0, 20, 30]
    water_vals = [20, 0, 30]

    #gridA
    grid_a = gridA.generate_grid_A();

    #grid A land
    r_a_land = numpy.dot(grid_a, land_vals)
    pprint.pprint(r_a_land.tolist())
    MDP_a_land = GridMDP(r_a_land.tolist(),
                    terminals=[], init=(0, 2), gamma=.95)
    pi_a_land = policy_iteration(MDP_a_land)
    print "land policy:"
    pprint.pprint(pi_a_land)

    #grid A water
    r_a_water = numpy.dot(grid_a, water_vals)
    MDP_a_water = GridMDP(r_a_water.tolist(),
                    terminals=[], init=(0,2), gamma=.95)
    pi_a_water = policy_iteration(MDP_a_water)
    print "water policy:"
    pprint.pprint(pi_a_water)
    print "here"
