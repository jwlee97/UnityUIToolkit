import pandas as pd
import numpy as np
from preference import ProbitPreferenceGP
from validations import check_x_m

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ProbitBayesianOptimization(ProbitPreferenceGP):
    """

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Training data consisting of numeric real positive values.

    M : array-like, shape = (n_samples, n_preferences)
        Target choices. A preference is an array of positive
        integers of shape = (2,). preference[0], r, is an index
        of X preferred over preference[1], c, which is an
        index of X.
    """
    def __init__(self, ui, img_path, gallery_size, X, M, GP_params={}):
        super().__init__(**GP_params)
        self.X_arr = X
        self.M = M
        self.ui = ui
        self.gallery_size = gallery_size
        self.img = plt.imread(img_path, 0)


    def plot_figures(self, iter, imgP, imgS):
        fig, ax = plt.subplots(1, 4)
        title = "Iteration " + str(iter+1)
        fig.suptitle(title, fontsize=16)

        ax[0].imshow(imgP)
        ax[0].set_title("Preference")

        for i, img in enumerate(imgS):
            ax[i+1].imshow(img)
            ax[i+1].set_title("Suggestion ")

        axP = plt.axes([0.1, 0.05, 0.16, 0.075])
        axS1 = plt.axes([0.3, 0.05, 0.16, 0.075])
        axS2 = plt.axes([0.5, 0.05, 0.16, 0.075])
        axS3 = plt.axes([0.7, 0.05, 0.16, 0.075])
        axQuit = plt.axes([0.9, 0.05, 0.16, 0.075])

        bP = Button(axP, 'Preference')
        bS1 = Button(axS1, 'Suggestion 1')
        bS2 = Button(axS2, 'Suggestion 2')
        bS3 = Button(axS3, 'Suggestion 3')

        bQuit = Button(axQuit, 'Quit')

        def on_click_p(event):
            global letter
            plt.close()
            letter = 'p'
            return letter

        def on_click_s1(event):
            global letter
            plt.close()
            letter = '1'
            return letter
        
        def on_click_s2(event):
            global letter
            plt.close()
            letter = '2'
            return letter
        
        def on_click_s3(event):
            global letter
            plt.close()
            letter = '3'
            return letter

        def on_quit(event):
            global letter
            plt.close()
            letter = 'Q'
            return letter

        bP.on_clicked(on_click_p)
        bS1.on_clicked(on_click_s1)
        bS2.on_clicked(on_click_s2)
        bS3.on_clicked(on_click_s3)
        bQuit.on_clicked(on_quit)
        plt.show()


    def get_img(self, i, w_coords, colors):
        img_copy = self.img.copy()
        w_coords = [w_coords[0]/100, w_coords[1]/100, w_coords[2]/100]
        uv_coords = self.ui.w2uv(w_coords)
        
        if i == 0 or len(colors) == 0:
            labelColor, _ = self.ui._color(uv_coords, i)
            ret_colors = self.ui.colorHarmony(labelColor, 93.6)

        if len(colors) > 0:
            ret_colors = colors

        min_x = int(uv_coords[0]-self.ui.panelDim[i][1]/2)
        max_x = int(uv_coords[0]+self.ui.panelDim[i][1]/2)
        min_y = int(uv_coords[1]-self.ui.panelDim[i][0]/2)
        max_y = int(uv_coords[1]+self.ui.panelDim[i][0]/2)
            
        cv2.rectangle(img_copy, (min_x, min_y), (max_x, max_y), ret_colors[i], -1)

        return img_copy, ret_colors


    def interactive_optimization(self, bounds, method="L-BFGS-B",
                                 n_init=1, n_solve=1, f_prior=None,
                                 max_iter=1e4, print_suggestion=True):
        """Bayesian optimization via preferences inputs.

        Parameters
        ----------
        bounds: dictionary
            Bounds of the search space for the acquisition function.

        method: str or callable, optional
            Type of solver.

        n_init: integer, optional
            Number of initialization points for the solver. Obtained
            by randomly sampling the acquisition function.

        n_solve: integer, optional
            The solver will be run n_solve times.
            Cannot be superior to n_init.

        f_prior: array-like, shape = (n_samples, 1), optional (default: None)
            Flat prior with mean zero is applied by default.

        max_iter: integer, optional (default: 1e4)
            Maximum number of iterations to be performed
            for the bayesian optimization.

        print_suggestion: Boolean, optional (default: True)
            If set to false, max_iter must be equal to 1.

        Returns
        -------
        optimal_values : array-like, shape = (n_features, )

        suggestion : array-like, shape = (n_features, )

        X : array-like, shape = (n_samples, n_features)
            Feature values in training data.

        M : array-like, shape = (n_samples - 1, 2)
            Target choices. A preference is an array of positive
            integers of shape = (2,). preference[0], r, is an index
            of X preferred over preference[1], c, which is an
            index of X.

        f_posterior : array-like, shape = (n_samples, 1)
            Posterior distribution of the  Gaussian Process.

        Examples
        --------
        >>> from GPro.kernels import Matern
        >>> from GPro.posterior import Laplace
        >>> from GPro.acquisitions import UCB
        >>> from GPro.optimization import ProbitBayesianOptimization
        >>> import numpy as np

        >>> GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
        ...      'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
        ...                             eta=0.01, tol=1e-3),
        ...      'acquisition': UCB(kappa=2.576),
        ...      'random_state': None}
        >>> X = np.random.sample(size=(2, 3)) * 10
        >>> M = np.array([0, 1]).reshape(-1, 2)
        >>> gpr_opt = ProbitBayesianOptimization(X, M, GP_params)
        >>> bounds = {'x0': (0, 10)}
        >>> console_opt = gpr_opt.interactive_optimization(bounds=bounds, n_solve=1,
        ...                                            n_init=100)
        >>> optimal_values, suggestion, X_post, M_post, f_post = console_opt
        >>> print('optimal values: ', optimal_values)

        >>> # Use posterior as prior
        >>> gpr_opt = ProbitBayesianOptimization(X_post, M_post, GP_params)
        >>> console_opt = gpr_opt.interactive_optimization(bounds=bounds, n_solve=1,
        ...                                            n_init=100,
        ...                                            f_prior=f_post)
        >>> optimal_values, suggestion, X_post, M_post, f_post = console_opt
        >>> print('optimal values: ', optimal_values)

        """

        if not max_iter:
            raise ValueError('max_iter must be superior to 0.')
        if not print_suggestion and max_iter > 1:
            raise ValueError('When print_suggestion is set to False, '
                             'max_iter must be set to 1.')
        
        X_arr = self.X_arr
        M = self.M
        features = list(bounds.keys())
        M_ind_cpt = M.shape[0] - 1

        f_prior_arr = []
        ret_colors = []
        df_arr = [None] * self.ui.num_panels
       
        for i in range(self.ui.num_panels):
            f_prior_arr.append(f_prior)
        
        pd.set_option('display.max_columns', None)
        iteration = 0

        while iteration < max_iter:
            for i in range(self.ui.num_panels):
                self.fit(X_arr[i], M, f_prior_arr[i])
                x_optim = self.bayesopt(bounds, method, n_init, n_solve)
                f_optim = self.predict(x_optim)
                f_prior_arr[i] = np.concatenate((self.posterior, f_optim))
                X_arr[i] = np.concatenate((X_arr[i], x_optim))
                
                # current preference index in X.
                M_ind_current = M[M.shape[0] - 1][0]
                
                # suggestion index in X.
                M_ind_proposal = M_ind_cpt + 2

                df = pd.DataFrame(data=np.concatenate((X_arr[i][[M_ind_current]],
                                                       X_arr[i][[M_ind_proposal]],
                                                       X_arr[i][[M_ind_proposal+1]],
                                                       X_arr[i][[M_ind_proposal+2]] )),
                                  columns=features,
                                  index=['preference', 'suggestion1', 'suggestion2', 'suggestion3'])
                
                df_arr[i] = df
            print(df_arr)
            
            if print_suggestion:
                for i, df in enumerate(df_arr):
                    imgP, colorP = self.get_img(i, df.loc['preference'].values, ret_colors)
                    imgS1, colorS1 = self.get_img(i, df.loc['suggestion1'].values, [])
                    imgS2, colorS2 = self.get_img(i, df.loc['suggestion2'].values, [])
                    imgS3, colorS3 = self.get_img(i, df.loc['suggestion3'].values, [])
            
                ret_img = imgP
                self.plot_figures(iteration, imgP, [imgS1, imgS2, imgS3])
                preference_input = letter

                print("User chooses: ", preference_input)

                if preference_input == 'Q':
                    break
                # left index is preferred over right index as a convention.
                elif preference_input == 'p':
                    new_pair = np.array([M_ind_current, M_ind_proposal])
                    ret_img = imgP
                    ret_colors = colorP
                elif preference_input == '1':
                    new_pair = np.array([M_ind_proposal, M_ind_current])
                    ret_img = imgS1
                    ret_colors = colorS1
                elif preference_input == '2':
                    new_pair = np.array([M_ind_proposal, M_ind_current])
                    ret_img = imgS2
                    ret_colors = colorS2
                elif preference_input == '3':
                    new_pair = np.array([M_ind_proposal, M_ind_current])
                    ret_img = imgS3
                    ret_colors = colorS3
                else:
                    break
                
                M = np.vstack((M, new_pair))
                M_ind_cpt += 1
                iteration += 1
            else:
                break

        pd.set_option('display.max_columns', 0)
        optimal_values = []

        for i, df in enumerate(df_arr):
            optimal_values.append(df.loc['preference'].values)
        
        f_posterior = f_prior_arr

        return optimal_values, X_arr, M, f_posterior, ret_img


    def function_optimization(self, f, bounds, max_iter=1,
                              method="L-BFGS-B", n_init=100, n_solve=1,
                              f_prior=None):
        """Bayesian optimization via function evaluation.

        Parameters
        ----------
        f: function object
            A function to be optimized.

        bounds: dictionary
            Bounds of the search space for the acquisition function.

        max_iter: integer, optional
            Maximum number of iterations to be performed
            for the bayesian optimization.

        method: str or callable, optional
            Type of solver.

        n_init: integer, optional
            Number of initialization points for the solver. Obtained
            by randomly sampling the acquisition function.

        n_solve: integer, optional
            The solver will be run n_solve times.
            Cannot be superior to n_init.

        f_prior : array-like, shape = (n_samples, 1), optional (default: None)
            Flat prior with mean zero is applied by default.

        Returns
        -------
        optimal_values : array-like, shape = (n_features, )

        X : array-like, shape = (n_samples, n_features)
            Feature values in training data.

        M : array-like, shape = (n_samples - 1, 2)
            Target choices. A preference is an array of positive
            integers of shape = (2,). preference[0], r, is an index
            of X preferred over preference[1], c, which is an
            index of X.

        f_posterior  : array-like, shape = (n_samples, 1)
            Posterior distribution of the  Gaussian Process.

        Examples
        --------
        >>> from GPro.kernels import Matern
        >>> from GPro.posterior import Laplace
        >>> from GPro.acquisitions import UCB
        >>> from GPro.optimization import ProbitBayesianOptimization
        >>> from scipy.stats import multivariate_normal
        >>> import numpy as np
        >>> from sklearn import datasets
        >>> import matplotlib.cm as cm
        >>> import matplotlib.pyplot as plt


        >>> # function optimization example.
        >>> def random_sample(n, d, bounds, random_state=None):
        >>>     # Uniform sampling given bounds.
        >>>     if random_state is None:
        >>>         random_state = np.random.randint(1e6)
        >>>     random_state = np.random.RandomState(random_state)
        >>>     sample = random_state.uniform(bounds[:, 0], bounds[:, 1],
        ...                                   size=(n, d))
        >>>     return sample


        >>> def sample_normal_params(n, d, bounds, scale_sigma=1, random_state=None):
        >>>     # Sample parameters of a multivariate normal distribution
        >>>     # sample centroids.
        >>>     mu = random_sample(n=n, d=d, bounds=np.array(list(bounds.values())),
        ...                        random_state=random_state)
        >>>     # sample covariance matrices.
        >>>     sigma = datasets.make_spd_matrix(d, random_state) * scale_sigma
        >>>     theta = {'mu': mu, 'sigma': sigma}
        >>>     return theta


        >>> d = 2
        >>> bounds = {'x' + str(i): (0, 10) for i in range(0, d)}
        >>> theta = sample_normal_params(n=1, d=d, bounds=bounds, scale_sigma=10, random_state=12)
        >>> f = lambda x: multivariate_normal.pdf(x, mean=theta['mu'][0], cov=theta['sigma'])
        >>> # X, M, init
        >>> X = random_sample(n=2, d=d, bounds=np.array(list(bounds.values())))
        >>> X = np.asarray(X, dtype='float64')
        >>> M = sorted(range(len(f(X))), key=lambda k: f(X)[k], reverse=True)
        >>> M = np.asarray([M], dtype='int8')
        >>> GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
        ...              'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
        ...                                     eta=0.01, tol=1e-3),
        ...              'acquisition': UCB(kappa=2.576),
        ...              'alpha': 1e-5,
        ...              'random_state': 2020}
        >>> gpr_opt = ProbitBayesianOptimization(X, M, GP_params)
        >>> function_opt = gpr_opt.function_optimization(f=f, bounds=bounds, max_iter=d*10,
        ...                                              n_init=1000, n_solve=1)

        >>> optimal_values, X_post, M_post, f_post = function_opt
        >>> print('optimal values: ', optimal_values)

        >>> # rmse
        >>> print('rmse: ', .5 * sum(np.sqrt((optimal_values - theta['mu'][0]) ** 2)))
        >>> # 2d plot
        >>> if d == 2:
        >>>     resolution = 10
        >>>     x_min, x_max = bounds['x0'][0], bounds['x0'][1]
        >>>     y_min, y_max = bounds['x1'][0], bounds['x1'][1]
        >>>     x = np.linspace(x_min, x_max, resolution)
        >>>     y = np.linspace(y_min, y_max, resolution)
        >>>     X, Y = np.meshgrid(x, y)
        >>>     grid = np.empty((resolution ** 2, 2))
        >>>     grid[:, 0] = X.flat
        >>>     grid[:, 1] = Y.flat
        >>>     Z = f(grid)
        >>>     plt.imshow(Z.reshape(-1, resolution), interpolation="bicubic",
        ...                origin="lower", cmap=cm.rainbow, extent=[x_min, x_max, y_min, y_max])
        >>>     plt.scatter(optimal_values[0], optimal_values[1], color='black', s=10)
        >>>     plt.title('Target function')
        >>>     plt.colorbar()
        >>>     plt.show()

        """

        X, M = check_x_m(self.X, self.M)
        new_pair = M[M.shape[0] - 1]
        for M_ind_cpt in range((M.shape[0] - 1), max_iter + (M.shape[0] - 1)):
            self.fit(X, M, f_prior)
            x_optim = self.bayesopt(bounds, method, n_init, n_solve)
            f_optim = self.predict(x_optim)
            f_prior = np.concatenate((self.posterior, f_optim))
            X = np.concatenate((X, x_optim))
            # current preference index in X.
            M_ind_current = M[M.shape[0] - 1][0]
            # suggestion index in X.
            M_ind_proposal = M_ind_cpt + 2
            new_pair = np.array([M_ind_current, M_ind_proposal])
            proposal = X[M_ind_proposal].reshape(1, -1)
            current = X[M_ind_current].reshape(1, -1)
            # minimize by convention.
            if f(current) < f(proposal):
                new_pair = np.array([M_ind_proposal, M_ind_current])
            M = np.vstack((M, new_pair))
        optimal_values = X[new_pair[0]]
        f_posterior = f_prior
        return optimal_values, X, M, f_posterior