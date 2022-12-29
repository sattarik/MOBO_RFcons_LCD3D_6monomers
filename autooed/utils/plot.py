import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from pymoo.factory import get_performance_indicator
import matplotlib as mpl
from autooed.utils.pareto import convert_minimization


def parallel_transform(Y):
    '''
    Transform performance values from cartesian to parallel coordinates
    '''
    Y = np.array(Y)
    return np.dstack([np.vstack([np.arange(Y.shape[1])] * len(Y)), Y])


def plot_performance_space(Y):
    '''
    '''
    Y = np.array(Y)
    assert Y.ndim == 2, f'Invalid shape {Y.shape} of objectives to plot'
    if Y.shape[1] == 1:
        plt.scatter(Y, [0] * len(Y), marker='x')
    elif Y.shape[1] == 2:
        plt.scatter(*Y.T)
    elif Y.shape[1] == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(*Y.T)
    elif Y.shape[1] > 3:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        segments = parallel_transform(Y)
        ax.add_collection(LineCollection(segments))
        ax.set_xlim(0, Y.shape[1] - 1)
        ax.set_ylim(np.min(Y), np.max(Y))
    else:
        raise Exception(f'Objectives with dimension {Y.shape[1]} is not supported')
    plt.title('Performance Space')
    plt.show()

def plot_performance_space_diffcolor(Y0, Y_eval):
    '''
    '''
    Y0 = np.array(Y0)
    Y_eval = np.array(Y_eval)
    assert Y_eval.ndim == 2, f'Invalid shape {Y.shape} of objectives to plot'
    if Y_eval.shape[1] == 1:
        plt.scatter(Y0, [0] * len(Y0), marker='x', color='red')
        plt.scatter(Y_eval, [0] * len(Y_eval), marker='x', color='blue')
    elif Y_eval.shape[1] == 2:
        plt.scatter(*Y0.T, color='red')
        plt.scatter(*Y_eval.T, color='blue')
    elif Y_eval.shape[1] == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(*Y0.T, color='red')
        ax.scatter(*Y_eval.T, color='blue')
    elif Y_eval.shape[1] > 3:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        segments = parallel_transform(Y_eval)
        ax.add_collection(LineCollection(segments))
        ax.set_xlim(0, Y_eval.shape[1] - 1)
        ax.set_ylim(np.min(Y_eval), np.max(Y_eval))
    else:
        raise Exception(f'Objectives with dimension {Y.shape[1]} is not supported')
    plt.title('Performance Space')
    plt.show()

def plot_performance_metric(Y, obj_type):
    '''
    '''
<<<<<<< HEAD
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.tick_params(direction='in', length=4, width=1.5, colors='black', grid_alpha=0, labelsize='1')
    ax.set_xlabel('Iterations', fontsize='18', fontname='Arial', fontweight='bold')
    ax.set_ylabel('Hypervolume', fontsize='18', fontname='Arial', fontweight='bold')
=======
    fig, ax = plt.subplots(figsize=(6,4))
>>>>>>> 97af2b00aa8533b72eea413eba3a91cd574e4aad
    if Y.shape[1] == 1:
        opt_list = []
        if obj_type == ['min']:
            opt_func = np.min
        elif obj_type == ['max']:
            opt_func == np.max
        else:
            raise Exception(f'Invalid objective type {obj_type}')
        for i in range(1, len(Y)):
            opt_list.append(opt_func(Y[:i]))
        plt.plot(np.arange(1, len(Y)), opt_list)
        plt.title('Optimum')
    elif Y.shape[1] > 1:
        Y = convert_minimization(Y, obj_type)
        ref_point = np.max(Y, axis=0)
        indicator = get_performance_indicator('hv', ref_point=ref_point)
        hv_list = []
        for i in range(1, len(Y)):
            hv = indicator.calc(Y[:i])
            hv_list.append(hv)
<<<<<<< HEAD
        
        plt.plot(np.arange(1, len(Y)), hv_list, linewidth=5)
        #plt.title('HVE0')
    else:
        raise Exception(f'Invalid objective dimension {Y.shape[1]}')
    mpl.rcParams['axes.linewidth'] = 2  
    ax.tick_params(axis='both', labelsize=15)
=======
        plt.plot(np.arange(1, len(Y)), hv_list, linewidth=5)
        plt.title('HVE')
    else:
        raise Exception(f'Invalid objective dimension {Y.shape[1]}')
    mpl.rcParams['axes.linewidth'] = 2
    ax.set_xlabel('Iterations', fontsize='18', fontname='Arial', fontweight='bold')
    ax.set_ylabel('Hypervolume', fontsize='18', fontname='Arial', fontweight='bold')   
    ax.tick_params(axis='both', labelsize=15)
    ax.xaxis.set_tick_params(width=5)
    ax.yaxis.set_tick_params(width=5)
>>>>>>> 97af2b00aa8533b72eea413eba3a91cd574e4aad
    plt.tight_layout()
    plt.savefig('HVE.png', dpi=300)
