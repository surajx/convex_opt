import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

data = scipy.io.loadmat('asgn5q2.mat')
m, n = data['X'].shape
sigmaHat = np.cov(data['X'], rowvar=0)

print("Shape: ", m, n)

K = cvx.semidefinite(n)

# Plot properties.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Create figure.
plt.figure()

Pstars = []
logL = []
nonZeroCount = []

reg_coeff_lambdas = [0.001, 0.01, 0.1, 1, 2, 4, 5, 8, 10]


def off_diag_l1_penalty(M, M_dim):
    return cvx.sum_entries(cvx.abs(M)) - sum([(M)[i, i] for i in range(M_dim)])


def reg_term(arr, dim):
    return np.sum(np.abs(arr)) - sum([(arr)[i, i] for i in range(dim)])

for reg_coeff_lambda in reg_coeff_lambdas:

    obj = cvx.Minimize(-cvx.log_det(K) + cvx.trace(sigmaHat * K) +
                       (reg_coeff_lambda * off_diag_l1_penalty(K, n)))

    problem = cvx.Problem(obj)

    print("Reg. Coeff::Lambda:", reg_coeff_lambda)
    Pstar = problem.solve(
        solver=cvx.CVXOPT, kktsolver='robust', feastol=1.0e-6)
    print("Optimal value: ", Pstar)
    if problem.status != cvx.OPTIMAL:
        raise Exception('CVXPY Error: Not optimal solution')

    Kstar = K.value
    # print("Optimal point: ", Kstar)

    logLstar = Pstar - (reg_coeff_lambda * reg_term(Kstar, n))
    print("Optimal log-Likelihood:", logLstar)

    # print("Kstar:", Kstar)
    # print("isclose:", np.isclose(Kstar, np.zeros((n, n)), atol=1.0e-6))
    # print("Kstar[mask]:", Kstar[np.isclose(
    #     Kstar, np.zeros((n, n)), atol=1.0e-6)])
    # print("len(Kstar[mask]):", Kstar[np.isclose(
    #     Kstar, np.zeros((n, n)), atol=1.0e-6)].shape[1])

    nZeroCount = Kstar[np.isclose(
        Kstar, np.zeros((n, n)), atol=1.0e-6)].shape[1]
    print("#Non-Zeros in inverse co-variance matrix:", nZeroCount)

    Pstars.append(Pstar)
    logL.append(logLstar)
    nonZeroCount.append(nZeroCount)

plt.plot(reg_coeff_lambdas, Pstars)
plt.plot(reg_coeff_lambdas, logL)
plt.plot(reg_coeff_lambdas, nonZeroCount)

plt.show()
