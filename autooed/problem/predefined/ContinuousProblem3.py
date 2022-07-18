import numpy as np
from autooed.problem.problem import Problem

class ContinuousProblem3(Problem):
    '''
    Example 3, with specified number of design variables, objectives and constraints, also same bounds for all design variables
    NOTE: for constraint value (g), > 0 means violating constraint, <= 0 means satisfying constraint
    '''
    config = {
        'type': 'continuous',
        'n_var': 6,
        'n_obj': 2,
        'n_constr': 1,
        'var_lb': 0,
        'var_ub': 0.5,
    }

    def evaluate_objective(self, x):
        f1 = float (input (
            "ratios A-F {} sum {} Enter Tensile Str. [Mpa]: ".
             format(np.round(x,2), np.sum(np.round(x,2)))))
        f2 = float (input ("Enter Toughness in Mpa: "))
        return f1, f2

    def evaluate_constraint(self, x):
        print ('x', x)
        #x1, x2, x3, x4, x5, x6 = x[0], x[1], x[2], x[3], x[4], x[5]
        g1 = 1
        if sum(x) < 1.0:
            if (sum(x)+0.001 > 1.0):
                g1 = -1
                #print ("constraints are met for:", x)
        elif sum(x) > 1.0:
            if (sum(x)-0.001 < 1.0):
                g1 = -1
        #g2 = (x2 + x3 - 2) ** 2 - 1e-5 # x2 + x3 = 2
        print ("g1 ", g1)
        return g1
