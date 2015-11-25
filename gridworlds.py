"""
Simulate grid worlds and generate expert trajectories
"""

class TransitionModel:
    """
    Template for an MDP transition model with stochastic actions.
    """
    def get_next_states(self, state, action):
        """
        Returns list of (next_state, probability) tuples
        """
        return []

    def get_prob(self, state, action, next_state):
        """
        returns T(s, a, s')
        """
        return 0.0

    def is_goal_state(self, state):
        return False

"""
TODO: create Transition Models for grid worlds A and B
"""

def generate_trajectories_world_a(n=1500):
    pass

def generate_trajectories_world_b(n=500):
    pass


