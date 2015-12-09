import pprint
import math
def generate_grid_B():
    num_rows = 5
    num_cols = 5
    gridA = [[[0 for k in xrange(2)] for j in xrange(num_cols)] for i in xrange(num_rows)]

    gridA[2][2] = [0, 0, 0] #(2,2) is an obstacle
    gridA[2][4] = [1, 0, 1] #(2,4) is set to land and destination

    pprint.pprint(gridA)
    return gridA

def expert():
    theta = [0, 20, 30] #reward preferring land over water
    theta_p = [20, 0, 30] #reward preferring water over land

#def calculate_log_likelihood(lambda, n_tot):
 #   log_sum = 0




if __name__ == '__main__':
    gridB = generate_grid_B()
