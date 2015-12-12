import pprint
import math
import mdp

def generate_grid_A():
    num_rows = 5
    num_cols = 5
    gridA = [[[0 for k in xrange(3)] for j in xrange(num_cols)] for i in xrange(num_rows)]
    
    for row in range(1, 4):
        for col in range(1, 4):
            gridA[row][col] = [0, 1, 0] #set middle 3x3 square to land

    for row in range(num_rows):
        gridA[row][0] = [1, 0, 0] #set col 0 to water
        gridA[row][4] = [1, 0, 0] #set col 4 to water
    
    for col in range(num_cols):
        gridA[0][col] = [1, 0, 0] #set row 0 to water
        gridA[4][col] = [1, 0, 0] #set row 4 to water

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
    gridA = generate_grid_A()
