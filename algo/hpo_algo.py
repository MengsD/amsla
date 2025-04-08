#-------------------------------------------------------------------------------
# @description
#  To ensure the reproducibility of our experiments and avoid the impact of LLM 
#  stochasticity on performance, we preemptively ran the full set of LLM 
#  combinations across all evaluated tasks (cf. Table 3) and repeated five times. 
#  Based on these runs, all algorithms obtain the same average accuracy and cost 
#  of LLM combinations for each corresponding task.
#
#  HPO_MOO: Bayesian optimizaiton based search
#
# @author 
#  Mengsu Ding<dingmengsu.dms@alibaba-inc.com>
#
#-------------------------------------------------------------------------------



from algo.algo_base import Algo
from algo.context import Context
from algo.cost_estimator import CostEstimator

import optuna
import logging
from optuna.samplers import TPESampler
from optuna.trial import TrialState, FrozenTrial
from typing import List
from abc import ABC, abstractmethod
from optuna.distributions import FloatDistribution,CategoricalDistribution


def get_quality_constraint(trial: optuna.trial.FrozenTrial):
    return trial.user_attrs.get("_#acc_obj", (1,))

class HPO_MOO(Algo):
    def __init__(self, ctx: Context, num_trial: int):
        self.num_trial: int = num_trial
        self.trial_count: int = 0
        self.param_name_list: List[str] = []

        Algo.__init__(self, ctx)
        self.__set_param_names__(self.ctx.steps(), self.ctx.candidates())


    def create_initial_trial(self):
        initial_trial_list = []
        return initial_trial_list

    def add_constraints(self, trial, accuracy, cost):
        """ constraints_func in optuna"
            A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or they are pruned, but this behavior is
            subject to change in the future releases.
        """
            
        constraint_result = (self.ctx.get_accuracy_objective() - accuracy,)
        trial.set_user_attr("_#acc_obj", constraint_result)

    def create_objective(self, trial):
        model_list: List[str] = self._explore_params(trial, self.ctx.candidates())
        self.trial_count += 1
        self.add_trace(model_list)
        accuracy, cost = self.ctx.get_accuracy_and_cost(model_list)
        self.add_constraints(trial, accuracy, cost)
        logging.info(f"""Trial-#{self.trial_count-1}, acc: {accuracy}; cost: {cost} | models: {model_list}""")
        return accuracy, cost 

    def post_process(self, pareto_front: list[FrozenTrial]):
        if pareto_front is None or len(pareto_front) == 0:
            return None, None, None
        res_model_list = None
        res_acc = 0
        res_cost = float('inf')
        for pareto_point in pareto_front:
            model_list = list(pareto_point.params.values())
            accuracy = pareto_point.values[0]
            cost = pareto_point.values[1]
            print(f"post_process iter: model_list:{model_list} accuracy:{accuracy} cost:{cost}")
            if accuracy >= self.ctx.get_accuracy_objective():
                if res_cost > cost:
                    res_cost = cost
                    res_acc = accuracy
                    res_model_list = model_list
        print(f"post_process return: {res_model_list}, {res_acc}, {res_cost}")

        if accuracy >= self.ctx.get_accuracy_objective():
            return res_model_list, res_acc, res_cost
        else:
            return None, None, None
 
    def optimize(self) -> List[str]:
        self.sampler = TPESampler(
            multivariate=True,
            n_startup_trials=5,
            seed=42,
            constraints_func=get_quality_constraint 
        )

        study = optuna.create_study(directions=['maximize', 'minimize'],
                                    sampler=self.sampler
                                   )
    
        initial_trials = self.create_initial_trial()
        for each_trial in initial_trials:
            study.add_trial(each_trial)

        study.optimize(self.create_objective, n_trials=self.num_trial)

        # output the best
        #logging.info(f"best_value: {study.best_value}, best_params: {study.best_params}")
        logging.info(f"best_value: {study.best_trials}")
        logging.info(f"best_value: {study.best_trials}")
        print('Number of iterations: ', len(study.trials))

        return self.post_process(study.best_trials)


    def __set_param_names__(self, num_param: int, model_list: List[str]):
        """ Set up parameter names. """
        for idx in range(self.ctx.steps()):
            self.param_name_list.append(f"model_param_{idx}")
            print(f"Parameter:{self.param_name_list[idx]}, Candidates: {model_list}")

    def _explore_params(self, trial, model_list: List[str]) -> List[str]:
        local_trail_param: List[str] = []
        for param_idx in range(len(self.param_name_list)):
            param_value = trial.suggest_categorical(
                              self.param_name_list[param_idx],
                              model_list)
            local_trail_param.append(param_value)
        return local_trail_param

    def _is_finished(self) -> bool:
        return self.trial_count >= self.num_trial
