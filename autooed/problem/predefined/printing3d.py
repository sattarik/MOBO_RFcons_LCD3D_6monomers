'''
DTLZ problem suite.
'''

import numpy as np
from pymoo.factory import get_reference_directions
from pymoo.problems.util import load_pareto_front_from_file

from autooed.problem.problem import Problem


class printing3d(Problem):

    config = {
        'type': 'continuous',
        'n_var': 5,
        'n_obj': 2,
        'n_constr': 2,
        'var_lb': [0, 0, 0, 0, 0],
        'var_ub': [1, 1, 1, 1, 1]
    }

    def __init__(self):
        super().__init__()
        self.k = self.n_var - self.n_obj + 1
    
    def obj_func(self, x_):
        f = []

        for i in range(0, self.n_obj):
            _f = float (input (
            "ratios A-F {} sum {} Enter objective {}: ".
             format(np.round(x,2), np.sum(np.round(x,2)), i)))
            _f_ = -_f
            f.append(_f)
          

        f = np.array(f)
        return f

    def evaluate_constraint(self, x_, RFclassifier):
        x1, x2, x3, x4, x5 = x_[0], x_[1], x_[2], x_[3], x_[4] 
        g1 = x1 + x2 + x3 + x4 + x5 -1
        x_ = x_.reshape(1, -1)
        g2 = -RFclassifier.predict_proba(x_)[0][1] + 0.5
        return g1, g2

class printing3d_dlp(printing3d):

    def _calc_pareto_front(self):
        ref_kwargs = dict(n_points=100) if self.n_obj == 2 else dict(n_partitions=15)
        ref_dirs = get_reference_directions('das-dennis', n_dim=self.n_obj, **ref_kwargs)
        return 0.5 * ref_dirs
   

    def evaluate_objective(self, x_, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            
            _f = float (input (
            "ratios A-F {} sum {} Enter objective {}: ".
             format(np.round(x_,2), np.sum(np.round(x_,2)), i)))
             
            _f = -_f

            f.append(_f)

        f = np.array(f)
        return f

    def evaluate_constraint(self, x_, RFclassifier):
        x1, x2, x3, x4, x5 = x_[0], x_[1], x_[2], x_[3], x_[4] 
        g1 = x1 + x2 + x3 + x4 + x5 -1
        x_ = x_.reshape(1, -1)
        g2 = -RFclassifier.predict_proba(x_)[0][1] + 0.5
        return g1, g2














