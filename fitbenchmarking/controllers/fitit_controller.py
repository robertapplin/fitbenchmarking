import numpy as np

from fitbenchmarking.controllers.base_controller import Controller

from fitit import fit, minimizers, CostFunction, Minimizer, Model
from fitit.models import Gaussian
from fitit.cost_functions import ChiSquared
from fitit.framework.inspect import get_class_from_module


class FitItController(Controller):
    algorithm_check = {
        "all": [
            "GradientDescent",
            "GaussNewton",
            "LevenbergMarquardt",
        ],
        "ls": ["GaussNewton", "LevenbergMarquardt"],
        "deriv_free": [],
        "general": [
            "GradientDescent",
            "GaussNewton",
            "LevenbergMarquardt",
        ],
        "simplex": [],
        "trust_region": [],
        "levenberg-marquardt": ["LevenbergMarquardt"],
        "gauss_newton": ["GaussNewton"],
        "bfgs": [],
        "conjugate_gradient": [],
        "steepest_descent": [],
        "global_optimization": [],
        "MCMC": [],
    }

    jacobian_enabled_solvers = [
        "GradientDescent",
        "GaussNewton",
        "LevenbergMarquardt",
    ]

    support_for_bounds = False

    def __init__(self, cost_func):
        """
        Setup workspace, cost_function, ignore_invalid, and initialise vars
        used for temporary storage within the mantid controller

        :param cost_func: Cost function object selected from options.
        :type cost_func: subclass of
                :class:`~fitbenchmarking.cost_func.base_cost_func.CostFunc`
        """
        super().__init__(cost_func)

        self._fitit_model = None
        self._fitit_cost_function = None
        self._fitit_minimizer = None

        self._fit_range = None

        self._evaluator = None
        self._exit_code = None

    def setup(self) -> None:
        """
        Setup problem ready to run with FitIt.
        """
        if self.problem.start_x is not None and self.problem.end_x is not None:
            self._fit_range = [self.problem.start_x, self.problem.end_x]

        class DerivedModel(Model):
            def evaluate(model_self, x, parameters):
                return self.problem.eval_model(x=x, params=parameters)

            def evaluate_jacobian(model_self, x, parameters):
                print("HERE")
                print(self.problem.jacobian)
                return self.problem.jacobian(x, parameters)

            def number_of_parameters(model_self) -> int:
                return len(self.initial_params)


        class DerivedCF(CostFunction):
            def evaluate(cf_self, x, y, e, parameters, model):
                residuals = self.cost_func.eval_r(parameters, x=x, y=y, e=e)
                return sum(residuals**2)

            def derivative(cf_self, x, y, e, parameters, model):
                residual = y - self.problem.eval_model(x=x, params=parameters)
                jac_res = self.cost_func.jac_res(parameters, e=e)
                return 2 * jac_res.T @ residual

            def weight_matrix(self, x, e):
                assert e is not None, "Chi Squared requires y errors to be provided for the weighting"
                assert np.all(e > 0), "Chi Squared requires all errors to be positive to avoid division by zero"

                return np.diag(1.0 / e ** 2)

        self._fitit_model = Gaussian()#DerivedModel()
        self._fitit_cost_function = DerivedCF() # Not finished
        if self.minimizer is not None:
            self._fitit_minimizer = get_class_from_module(minimizers, Minimizer, self.minimizer)()

    def fit(self) -> None:
        """
        Run problem with FitIt.
        """
        evaluator, exit_code = fit(
            self.data_x,
            self.data_y,
            self._fitit_model,
            e=self.data_e,
            cost_function=self._fitit_cost_function,
            start_parameters=self.initial_params,
            minimizer=self._fitit_minimizer,
            fit_range=self._fit_range
        )

        self._evaluator = evaluator
        self._exit_code = exit_code

    def cleanup(self) -> None:
        """
        Convert the result to a numpy array and populate the variables results will be read from.
        """
        if self._exit_code.value in [0, 1, 7]:
            self.flag = self._exit_code.value
        else:
            self.flag = 2

        if self._evaluator is not None:
            self.final_params = self._evaluator.final_parameters()
