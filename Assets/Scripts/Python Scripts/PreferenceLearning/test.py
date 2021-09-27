from scipy.optimize.optimize import OptimizeResult
from PreferenceLearning.kernels import Matern
from PreferenceLearning.posterior import Laplace
from PreferenceLearning.acquisitions import UCB
import numpy as np
import cv2
import os

from UIoptimizer import UIOptimizer
from PreferenceLearning.optimization import ProbitBayesianOptimization

def toolkit(img_b64, optimizer):
    X_arr = []

    for i in range(optimizer.num_panels):
        X = np.random.sample(size=(4, 3)) * 10
        X_arr.append(X)

    M = np.array([0, 1]).reshape(-1, 2)
   
    GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
                'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
                                        eta=0.01, tol=1e-3),
                'acquisition': UCB(kappa=2.576),
                'alpha': 1e-5,
                'random_state': None}

    gpr_opt = ProbitBayesianOptimization(optimizer, img_b64, X_arr, M, GP_params)
    bounds = {'x0': (optimizer.xl[0], optimizer.xu[0]), 'x1': (optimizer.xl[1], optimizer.xu[1]), 'x2': (optimizer.xl[2], optimizer.xu[2])}

    optimal_values, _, colors = gpr_opt.interactive_optimization(bounds=bounds, n_init=100, n_solve=10)
    print('--- Optimal values: ---')
    
    for val in optimal_values: 
        print(val)

    return optimal_values, colors


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_buffer_file = dir_path + "\\input_images\\context_img_buff_1629133512.log"
    img_meta_file = dir_path + "\\input_images\\context_img_1629133512.log"
    img_file = dir_path + "\\input_images\\context_img_1629133512.png"
    img = cv2.imread(img_file)
    f = open(img_buffer_file, 'r')
    byte_arr = bytes(f.read(), 'utf-8')

    with open(img_meta_file, 'r') as f:
        meta_data = f.read()

    img_dim = [504, 896]
    panel_dim = [(0.1, 0.15), (0.1, 0.1), (0.2, 0.1), (0.15, 0.1)]
    occlusion = False
    num_panels = 4

    colorfulness = 0.0
    edgeness = 0.0
    fitts_law = 0.0
    ce = 0.33
    muscle_act = 0.33
    rula = 0.33

    optimizer = UIOptimizer(byte_arr, meta_data, np.array(img_dim), np.array(panel_dim), num_panels, occlusion, 
                     colorfulness, edgeness, fitts_law, ce, muscle_act, rula)

    ### Preference learning w/ Bayesian Optimization ###
    ### 3D example
    X_arr = []

    for i in range(optimizer.num_panels):
        X = np.random.sample(size=(4, 3)) * 10
        X_arr.append(X)

    M = np.array([0, 1]).reshape(-1, 2)
   
    GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
                'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
                                        eta=0.01, tol=1e-3),
                'acquisition': UCB(kappa=2.576),
                'alpha': 1e-5,
                'random_state': None}

    gpr_opt = ProbitBayesianOptimization(optimizer, optimizer.img, X_arr, M, GP_params)
    bounds = {'x0': (optimizer.xl[0], optimizer.xu[0]), 'x1': (optimizer.xl[1], optimizer.xu[1]), 'x2': (optimizer.xl[2], optimizer.xu[2])}

    optimal_values, ret_img, colors = gpr_opt.interactive_optimization(bounds=bounds, n_init=100, n_solve=10)

    print('--- Optimal values: ---')
    for val in optimal_values: 
        print(val)
    
    ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Preference Learning output', ret_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()