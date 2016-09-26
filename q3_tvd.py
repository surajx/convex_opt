import cvxpy as cvx
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

data = scio.loadmat('asgn5q3.mat')
x_corr = data['x_corr']
n = len(data['x_corr'])

# plt.plot(data['x_corr'], linewidth=2, label="corrupted")

xHat = cvx.Variable(n)

reg_coeff_lambdas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]

D = np.zeros((n - 1, n))
np.fill_diagonal(D[:, 0:], -1)
np.fill_diagonal(D[:, 1:], 1)

for reg_coeff_lambda in reg_coeff_lambdas:

    obj = cvx.Minimize(cvx.norm(xHat - x_corr) ** 2 +
                       reg_coeff_lambda * cvx.norm(D * xHat, 1))

    problem = cvx.Problem(obj)

    print("Reg. Coeff::Lambda:", reg_coeff_lambda)
    Pstar = problem.solve(verbose=True, solver=cvx.SCS)
    if problem.status != cvx.OPTIMAL:
        raise Exception('CVXPY Error: Not optimal solution')
    print("Optimal value: ", Pstar)

    xHatStar = xHat.value
    # print("xHatStar:", xHatStar)

    # plt.plot(xHatStar, linewidth=2,
    #          label="recovered, λ=" + str(reg_coeff_lambda))

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
plt.show()
