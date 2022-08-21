#!/usr/bin/env python
import time
import os
import io

from random import seed
from random import randint

from argparse import ArgumentParser, Namespace
import yaml
from multiprocessing import cpu_count

import matplotlib.pyplot as plt

# default is to maximize the objectives
import time as time
import copy
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from scipy.stats import norm

# example of a gaussian process surrogate function
from math import sin
from math import pi
import numpy as np
from numpy import arange
from numpy import asarray
from numpy.random import normal
from numpy.random import uniform
from numpy.random import random
from numpy import cov
from numpy import mean
from numpy import std

from warnings import catch_warnings
from warnings import simplefilter


from autooed.utils.sampling import lhs
import random
#import xgboost as xgb
#from xgboost import XGBRegressor
#from xgboost import plot_tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import export_graphviz
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.inspection import partial_dependence, plot_partial_dependence
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# loocv to manually evaluate the performance of a random forest classifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

from scipy.stats import pearsonr as pearsonr
from scipy import ndimage, misc
import pickle
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from chainer_chemistry.datasets.molnet import get_molnet_dataset
# the package is in the same directory
# get Today's date from python!
from datetime import datetime
from autooed.utils.sampling import lhs
from autooed.problem import build_problem
from autooed.mobo import build_algorithm
from autooed.utils.seed import set_seed
from autooed.utils.initialization import generate_random_initial_samples, load_provided_initial_samples
from autooed.utils.plot import plot_performance_space, plot_performance_metric
from autooed.utils.plot import plot_performance_space_diffcolor
from argparse import ArgumentParser, Namespace
from arguments import get_args

#### preprocessing 
# printability as Y, Tg
df = pd.read_csv('Yuchao_20220816.csv')
Printability = np.asarray (df['Printability']).reshape(1,-1).T
Y0 = Printability
Y = np.where(Y0 == 'Y', 1, 0)
Tg = np.asarray (df['Tg']).reshape(1,-1).T
# put a very high value for Tg that are not printable, too brittle
Tg[np.isnan(Tg)] = 200
# group the Tg, we want it to be in range [10, 60]
Tg_group = [1 if 10<i<60 else 0 for i in Tg]
Tg_group = np.array(Tg_group)
# read the 2 objectives
toughness = np.asarray (df['Toughness(MJ/m3)']).reshape(1,-1).T
toughness[np.isnan(toughness)] = 0
strength = np.asarray (df['Tensile_Strength(MPa)']).reshape(1,-1).T
strength[np.isnan(strength)] = 0
# not using Tensile strain yet
strain = np.asarray (df['Tensile_Strain_percentage']).reshape(1,-1).T
strain[np.isnan(strain)] = 0

# read the ratios of 6 monomers.
A_Ratio = np.asarray (df['R1(HA)']).reshape(1,-1)
B_Ratio = np.asarray (df['R2(IA)']).reshape(1,-1)
C_Ratio = np.asarray (df['R3(NVP)']).reshape(1,-1)
D_Ratio = np.asarray (df['R4(AA)']).reshape(1,-1)
E_Ratio = np.asarray (df['R5(HEAA)']).reshape(1,-1)
F_Ratio = np.asarray (df['R6(IBOA)']).reshape(1,-1)
# did not consider F_Ratio, since we do not have it in optimization
X_ = np.concatenate((A_Ratio.T, B_Ratio.T, C_Ratio.T, D_Ratio.T, E_Ratio.T), axis=1)
X0 = np.concatenate((A_Ratio.T, B_Ratio.T, C_Ratio.T, D_Ratio.T, E_Ratio.T, F_Ratio.T), axis=1)

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
# multiply Ratios by their descriptors
X_energy = np.multiply (X0, energy)
#X_pol_area = np.multiply (X0, pol_area)
X_complexity = np.multiply (X0, complexity)
X_HAMW = np.multiply (X0, HAMW)
X_solubility_d = np.multiply (X0, solubility_d)
X_solubility_h = np.multiply (X0, solubility_h)
X_solubility_p = np.multiply (X0, solubility_p)
X = np.concatenate ((X_energy, X_complexity, X_HAMW, 
                    X_solubility_d, X_solubility_h, X_solubility_p), axis=1)

# got more information about input varialbe may reduce the accuracy for 
# few samples, but it is informative for new samples.
# The hyperparameters are fixed using one-leave-out in file "leavout_CV_RF_printability_Tg.ipynb"
RF_print = RandomForestClassifier(random_state=0, 
                                  max_depth = 5, 
                                  n_estimators = 50)
RF_print.fit(X, Y)
Yhat = RF_print.predict(X)
acc = accuracy_score(Y, Yhat)
print('Accuracy: %.3f' % acc)
#print (RF_print.get_params(deep=True))
RF_Tg = RandomForestClassifier(random_state=0, 
                                  max_depth = 5, 
                                  n_estimators = 50)
RF_Tg.fit(X, Tg_group)
Yhat = RF_Tg.predict(X)
acc = accuracy_score(Tg_group, Yhat)
print('Accuracy: %.3f' % acc)
#print (RF_Tg.get_params(deep=True))
# RF.n_estimators = int (5 * RF.n_estimators)
# RF2 = RF.fit(X_train[0:5,:], y_train[0:5])
# pred = RF2.predict_proba(X_train)
# print (RF2.score(X_train, y_train))
# print (RF2.score(X_test, y_test))

### Start the real optimization
# Change the default values for new argument
def get_general_args(args=None):
    '''
    General arguments: problem and algorithm description, experiment settings
    '''
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, default='printing3d_dlp', 
        help='optimization problem')
    parser.add_argument('--algo', type=str, default='tsemo',
        help='type of algorithm to use with some predefined arguments, or custom arguments')
    parser.add_argument('--seed', type=int, default=10,
        help='the specific seed to run')
    parser.add_argument('--batch-size', type=int, default=1, 
        help='size of the batch in optimization')
    parser.add_argument('--n-init-sample', type=int, default=0, 
        help='number of initial design samples')
    parser.add_argument('--n-total-sample', type=int, default=200, 
        help='number of total design samples (budget)')

    args, _ = parser.parse_known_args(args)
    return args


def get_surroagte_args(args=None):
    '''
    Arguments for fitting the surrogate model
    '''
    parser = ArgumentParser()

    parser.add_argument('--surrogate', type=str, 
        choices=['gp', 'nn', 'bnn'], default='gp', 
        help='type of the surrogate model')

    args, _ = parser.parse_known_args(args)
    return args


def get_acquisition_args(args=None):
    '''
    Arguments for acquisition function
    '''
    parser = ArgumentParser()

    parser.add_argument('--acquisition', type=str,  
        choices=['identity', 'pi', 'ei', 'ucb', 'ts'], default='ts', 
        help='type of the acquisition function')

    args, _ = parser.parse_known_args(args)
    return args


def get_solver_args(args=None):
    '''
    Arguments for multi-objective solver
    '''
    parser = ArgumentParser()

    # general solver
    parser.add_argument('--solver', type=str, 
        choices=['nsga2', 'moead', 'parego', 'discovery', 'ga', 'cmaes'], default='nsga2', 
        help='type of the multiobjective solver')
    parser.add_argument('--n-process', type=int, default=2,
        help='number of processes to be used for parallelization')

    args, _ = parser.parse_known_args(args)
    return args


def get_selection_args(args=None):
    '''
    Arguments for sample selection
    '''
    parser = ArgumentParser()

    parser.add_argument('--selection', type=str,
        choices=['direct', 'hvi', 'random', 'uncertainty'], default='hvi', 
        help='type of selection method for new batch')

    args, _ = parser.parse_known_args(args)
    return args

def get_args():
    '''
    Get arguments from all components
    You can specify args-path argument to directly load arguments from specified yaml file
    '''
    parser = ArgumentParser()
    parser.add_argument('--args-path', type=str, default=None,
        help='used for directly loading arguments from path of argument file')
    args, _ = parser.parse_known_args()

    if args.args_path is None:

        general_args = get_general_args()
        surroagte_args = get_surroagte_args()
        acquisition_args = get_acquisition_args()
        solver_args = get_solver_args()
        selection_args = get_selection_args()

        module_cfg = {
            'surrogate': vars(surroagte_args),
            'acquisition': vars(acquisition_args),
            'solver': vars(solver_args),
            'selection': vars(selection_args),
        }

    else:
        
        with open(args.args_path, 'r') as f:
            all_args = yaml.load(f)
        
        general_args = Namespace(**all_args['general'])
        module_cfg = all_args.copy()
        module_cfg.pop('general')

    return general_args, module_cfg


# load arguments
args, module_cfg = get_args()
print (args.seed)
# set random seed
set_seed(args.seed)

# build problem
problem = build_problem(args.problem)
print(problem)

# build algorithm
algorithm = build_algorithm(args.algo, problem, module_cfg)
print(algorithm)

# generate initial random samples
X = generate_random_initial_samples(problem, args.n_init_sample)
Y = np.array([problem.evaluate_objective(x) for x in X])
print ('read X', X.shape)
print ('read Y', Y.shape)

# read the initial samples, X is 6 Ratios, Y is two objective: 
# Strength, Toughness.
path = ['./Yuchao_20220816_X.csv', 
        './Yuchao_20220816_Y.csv']
X, Y = load_provided_initial_samples(path)
# we minimze the Objectives, so multiply -1
Y = -Y
print ('read X', X.shape)
print ('read Y', Y.shape)

X0 = X
Y0 = Y
# optimization
while len(X) < args.n_total_sample:
    start = time.time()
    # propose design samples
    X_next = algorithm.optimize(X, Y, X_busy=None, batch_size=2)
    print (X_next)
    print (time.time() - start)
    # evaluate proposed samples
    Y_next = np.array([problem.evaluate_objective(x) for x in X_next])
    # combine into dataset
    X = np.vstack([X, X_next])
    Y = np.vstack([Y, Y_next])
    for (x_next, y_next) in zip(X_next, Y_next):
        while True:
            try:
                printability_new = int (input (
                    "ratios A-F {} sum {} Enter Printability 0or1: ".
                     format(np.round(x_next, 2), np.sum(np.round(x_next, 2)))))
            except ValueError:
                print ("printability is not read correctly")
                continue
            else:
                break
        while True:
            try:
                 Tg_new = float (input (
                  "ratios A-F {} sum {} Enter Tg: ".
                 format(np.round(x_next,2), np.sum(np.round(x_next, 2)))))
            except ValueError:
                print ("Tg is not read correctly")
                continue
            else:
                break
        new_printability_Tg = [list(np.round(x_next,2)), [1-np.sum(np.round(x_next,2))], printability_new, Tg_new]
        new_sample = new_printability_Tg
        new_printability_Tg = str(new_printability_Tg)
        new_printability_Tg = new_printability_Tg.replace("[", "")
        new_printability_Tg = new_printability_Tg.replace("]", "")
        new_printability_Tg = new_printability_Tg + "\n"
        with open('printability_Tg.csv','a') as fd:
            fd.write(new_printability_Tg)
        new_sample.append(y_next[0])
        new_sample.append(y_next[1])
        new_sample = str(new_sample)
        new_sample = new_sample.replace("[", "")
        new_sample = new_sample.replace("]", "")
        new_sample = new_sample + "\n"
        with open('new_evaluated.csv','a') as fd:
            fd.write(new_sample)        
    print(f'{len(X)}/{args.n_total_sample} complete')
    print (time.time() - start)




# plot
Y_eval = Y[Y0.shape[0]:, :]
plot_performance_space_diffcolor(Y0=-Y0, Y_eval=-Y_eval)
plot_performance_metric(Y, problem.obj_type)


