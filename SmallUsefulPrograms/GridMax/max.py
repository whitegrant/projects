#Grant White
#August 2019


import numpy as np

def maximum(filename='grid.npy'):
    """Given an array, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.

    parameter: (str) name of 2-d NumPy array
    test with filename = 'grid.npy'
    """
    grid = np.load(filename)

    right_max = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])
    down_max = np.max(grid[:-3] * grid[1:-2] * grid[2:-1] * grid[3:])
    rdiag_max = np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:,3:])
    ldiag_max = np.max(grid[3:,:-3] * grid[2:-1,1:-2] * grid[1:-2,2:-1] * grid[:-3,3:])

    return max(right_max, down_max, rdiag_max, ldiag_max)


    """
    This is the naive way of finding the answer.
    It still works, it just takes a lot more code and time.
    """
    
    # grid = np.load("grid.npy")
    # dim = 20
    # greatest = grid[0, 0] * grid[0, 1] * grid[0, 2] * grid[0, 3]

    # #SIDE-BY-SIDE
    # for i in range(dim):
    #     for j in range(dim - 3):
    #         cur_val = grid[i, j] * grid[i, j+1] * grid[i, j+2] * grid[i, j+3]
    #         if cur_val > greatest:
    #             greatest = cur_val

    # #UP AND DOWN
    # for i in range(dim):
    #     for j in range(dim - 3):
    #         cur_val = grid[j, i] * grid[j+1, i] * grid[j+2, i] * grid[j+3, i]
    #         if cur_val > greatest:
    #             greatest = cur_val

    # #DIAGONAL (down and right)
    # for i in range(dim - 3):
    #     for j in range(dim - 3):
    #         cur_val = grid[i, j] * grid[i+1, j+1] * grid[i+2, j+2] * grid[i+3, j+3]
    #         if cur_val > greatest:
    #             greatest = cur_val

    # #DIAGONAL (up and right)
    # for i in range(dim - 3):
    #     for j in range(dim - 3):
    #         cur_val = grid[i+3, j] * grid[i+2, j+1] * grid[i+1, j+2] * grid[i, j+3]
    #         if cur_val > greatest:
    #             greatest = cur_val

    # return greatest
