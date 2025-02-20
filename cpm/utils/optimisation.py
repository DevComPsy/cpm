import numpy as np


class LinearConstraint:
    """
    This class is used to define linear constraints for the optimisation problem.

    Parameters
    ----------
    A : array_like
        Coefficients of the linear constraint.
    b : array_like
        Right-hand side of the linear constraint.
    type : str
        Type of the constraint. Can be either "ineq" for inequality or "eq" for equality.

    Examples
    --------
    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    constraint = LinearConstraint(A, b, type="ineq")

    x = np.array([0.5, 0.5])
    print("Constraint value:", constraint(x))
    print("Fulfills constraint:", constraint.fulfills(x))
    print("Constraint residual:", constraint.residual(x))
    """

    def __init__(self, A, b, type="ineq"):
        self.A = A
        self.b = b
        self.type = type

    def __call__(self, x):
        return np.dot(self.A, x) - self.b

    def fulfills(self, x):
        """
        Check if the constraint is fulfilled for the given parameter values.
        """
        if self.type == "ineq":
            return (self(x) >= 0).all()
        elif self.type == "eq":
            return (self(x) == 0).all()
        else:
            raise ValueError("Constraint type must be either 'ineq' or 'eq'.")

    def residual(self, x):
        """
        Calculate the residual of the constraint for the given parameter values.
        """
        if self.type == "ineq":
            return np.linalg.norm(np.minimum(0, self(x)))
        elif self.type == "eq":
            return np.linalg.norm(self(x))
        else:
            raise ValueError("Constraint type must be either 'ineq' or 'eq'.")
