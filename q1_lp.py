import cvxpy as cvx
import numpy as np

# Create optimization variables
# x \in R^4
x = cvx.Variable(4)

# Coefficient vector
c = np.array([1, 2, 3, 4])

obj = cvx.Minimize(c.T * x)

constraints = [
    np.array([1, 1, 1, 1]).T * x == 1,
    np.array([1, -1, 1, -1]).T * x == 0,
    x >= 0
]

problem = cvx.Problem(obj, constraints)

print("Optimal value: ", problem.solve())
print("Optimal point: ", x.value)
