'''
3D DLP printing problem suite.
'''

import numpy as np
from pymoo.factory import get_reference_directions
from pymoo.problems.util import load_pareto_front_from_file

from autooed.problem.problem import Problem

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class printing3d(Problem):

    config = {
        'type': 'continuous',
        'n_var': 5,
        'n_obj': 2,
        'n_constr': 3,
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
    
    def evaluate_constraint(self, x_):
        x1, x2, x3, x4, x5 = x_[0], x_[1], x_[2], x_[3], x_[4] 
        g1 = x1 + x2 + x3 + x4 + x5 -1
        x_ = x_.reshape(1, -1)
        #print ('Printability accuracy on all data', RF_print.score(X_, Y))
        #print ('Tg accuracy on all data group 1 in range of [{}, {}] is: {}'.format(Tg_min, Tg_max, RF_Tg.score(X_, Tg_group)))
        g2 = -RF_print.predict_proba(x_)[0][1] + 0.7
        g3 = -RF_Tg.predict_proba(x_)[0][1] + 0.7
        return g1, g2, g3


class printing3d_dlp(printing3d):
    def _calc_pareto_front(self):
        ref_kwargs = dict(n_points=100) if self.n_obj == 2 else dict(n_partitions=15)
        ref_dirs = get_reference_directions('das-dennis', n_dim=self.n_obj, **ref_kwargs)
        return 0.5 * ref_dirs
   
    def evaluate_objective(self, x_, alpha=1):
        f = []
        objectives = ['Strength_Mpa', 'Toughness_MJ_m3']
        for i in range(0, self.n_obj):
            while True:
                try:
                    _f = float (input (
                            "ratios A-F {} sum {} Enter objective {}: ".
                            format(np.round(x_,2), np.sum(np.round(x_,2)), objectives[i])))
                except ValueError:
                    print ('the objective {} was not valid, try again'.format(objectives[i]))
                    continue
                else:
                    break
            _f = -_f
            f.append(_f)

        f = np.array(f)
        return f

    def evaluate_constraint(self, x_):
        x1, x2, x3, x4, x5 = x_[0], x_[1], x_[2], x_[3], x_[4] 
        g1 = x1 + x2 + x3 + x4 + x5 -1
        x_ = x_.reshape(1, -1)
        
        df = pd.read_csv('./printability_Tg.csv')
        Printability = np.asarray (df['Printability']).reshape(1,-1)
        Y0 = Printability.T
        Y = Y0
        Y = np.ravel(Y)
        #print ("samples for training Tg and printability", Y.shape)

        Tg = np.asarray (df['Tg']).reshape(1,-1).T
        #Tg[np.isnan(Tg)] = 200
        Tg_min = 10
        Tg_max = 60
        Tg_group = [1 if 10<i<60 else 0 for i in Tg]
        Tg_group = np.array(Tg_group)

        #X_ = df.to_numpy()
        A_Ratio = np.asarray (df['R1(HA)']).reshape(1,-1)
        B_Ratio = np.asarray (df['R2(IA)']).reshape(1,-1)
        C_Ratio = np.asarray (df['R3(NVP)']).reshape(1,-1)
        D_Ratio = np.asarray (df['R4(AA)']).reshape(1,-1)
        E_Ratio = np.asarray (df['R5(HEAA)']).reshape(1,-1)
        F_Ratio = np.asarray (df['R6(IBOA)']).reshape(1,-1)
        X0 = np.concatenate((A_Ratio.T, B_Ratio.T, C_Ratio.T, 
                             D_Ratio.T, E_Ratio.T, F_Ratio.T), axis=1)

        # load monomers descriptors
        df = pd.read_csv('monomers_info.csv')
        energy = np.array (-df['dft_sp_E_RB3LYP'])
        pol_area = np.array (df['polar_surface_area'])
        complexity = np.array (df['complexity'])
        HAMW = np.array (df['HAMW'])
        solubility = np.array (df['solubility_sqrt_MJperm3'])
        solubility_d = np.array (df['solubility_dipole'])
        solubility_h = np.array (df['solubility_h'])
        solubility_p = np.array (df['solubility_p'])
        X_energy = np.multiply (X0, energy)
        X_complexity = np.multiply (X0, complexity)
        X_HAMW = np.multiply (X0, HAMW)
        X_solubility_d = np.multiply (X0, solubility_d)
        X_solubility_h = np.multiply (X0, solubility_h)
        X_solubility_p = np.multiply (X0, solubility_p)

        X = np.concatenate ((X_energy, X_complexity, X_HAMW, 
                    X_solubility_d, X_solubility_h, X_solubility_p), axis=1)

        RF_print = RandomForestClassifier(max_depth=5, n_estimators=50, random_state=0, criterion='gini')
        RF_Tg =    RandomForestClassifier(max_depth=5, n_estimators=50, random_state=0, criterion='gini')

        RF_print.fit(X, Y)
        RF_Tg.fit(X, Tg_group)

        #print ('Printability accuracy on all data', RF_print.score(X, Y))
        #print ('Tg accuracy on all data group 1 in range of [{}, {}] is: {}'.format(Tg_min, Tg_max, RF_Tg.score(X, Tg_group)))
        x_ = np.append(x_, [1-np.sum (x_)])
        X_energy = np.multiply (x_, energy)
        X_complexity = np.multiply (x_, complexity)
        X_HAMW = np.multiply (x_, HAMW)
        X_solubility_d = np.multiply (x_, solubility_d)
        X_solubility_h = np.multiply (x_, solubility_h)
        X_solubility_p = np.multiply (x_, solubility_p)

        X = np.concatenate ((X_energy, X_complexity, X_HAMW, 
                    X_solubility_d, X_solubility_h, X_solubility_p), axis=0)
        X = X.reshape(1, -1)
        g2 = -RF_print.predict_proba(X)[0][1] + 0.7
        g3 = -RF_Tg.predict_proba(X)[0][1] + 0.7
        return g1, g2, g3

