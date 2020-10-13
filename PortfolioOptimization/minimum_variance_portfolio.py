#minimum_variance_portfolio.py
"""
Grant White
March 29, 2020
"""

import numpy as np
from cvxopt import matrix, solvers


def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization
    Finds the minimum variance portfolio with and without short-selling.
    Solves:
        minimize    (1/2)(x.T)Px + (q.T)x
        subject to  Gx <= h
                    Ax = b    

    Parameters:
        filename (str): The name of the portfolio data file.
            Each column is a differnt asset.
            Each row is a different year (or other unit of time)
            The values are the returns.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """

    #read in data
    data = np.loadtxt(filename)
    n = len(data[0]) - 1

    #create the covariance matrix
    P = np.cov(data[:,1:], rowvar=False)

    #expected rate of return
    mu = np.average(data[:,1:], axis=0)
    R = 1.13

    #matricies
    q = np.zeros(n)
    G = -np.eye(n)
    h = np.zeros(n)
    A = np.vstack((np.ones(n), mu))
    b = np.array((1, R))

    #convert to cvxopt matricies
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    #solve
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    sol_short = solvers.qp(P=P, q=q, A=A, b=b)

    #return optimal portfolio with and without short-selling
    return np.ravel(sol_short['x']), np.ravel(sol['x'])
