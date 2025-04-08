#-------------------------------------------------------------------------------
# @description
#  To ensure the reproducibility of our experiments and avoid the impact of LLM 
#  stochasticity on performance, we preemptively ran the full set of LLM 
#  combinations across all evaluated tasks (cf. Table 3) and repeated five times. 
#  Based on these runs, all algorithms obtain the same average accuracy and cost 
#  of LLM combinations for each corresponding task.
#
#  Context classes store task-specific context data. It calculates the default
#  values of a and theta (cf. our problem formulation) according to the 
#  profiling combinations with the same LLM.
#
# @author 
#  Mengsu Ding<dingmengsu.dms@alibaba-inc.com>
#
#-------------------------------------------------------------------------------


from abc import ABC, abstractmethod
from algo.data_map import DataMap
from typing import List
import pandas as pd

from algo.cost_estimator import CostEstimator
from algo.util import Config

import sys
         
class Context:
    def __init__(self, theta: float = None, lamda: int = 6, cost_est: bool=True, good_modle_enable: float = False):
        self._best_acc: float = 100.0
        self._best_theta: float = None
        if theta is not None:
            self._best_theta = theta
        self._lamda = lamda
        self.good_model_enable = good_modle_enable

        self._best_models = None
        self._best_cost = None
 
        # app-dependent arguments
        self.num_steps = 0
        self.model_list = None
        self.single_model_map: DataMap = None   # results of combinations using
                                                # the same model
        self.results_map: DataMap = None        # results of all combinations
        self.config = None
        self.estimator = None

        self.exec_mode = False  #  False: read exec data from file

        # inner debug
        self.cost_est = cost_est # if False, we use executed cost as estimated cost

 
    def set_app(self,
                num_steps: int,
                model_list: List[str],
                run_data_file: str = None,
                #D single_data_file: str = None,
                config_file: str = None,
               ):

        self.num_steps = num_steps
        self.model_list = model_list

        # for reproducibility, we preemptively ran all LLM combinations and read examined
        # results from files
        assert run_data_file is not None
        self.exec_mode = False
        self.results_map = DataMap(run_data_file)

        # model profiling: model combinations with the same model
        self.preprocess()
        default_theta = self._best_theta
        self._best_models, self._best_acc, self._best_theta = self.single_model_map.get_the_best_models_info()
        if default_theta is not None:
            self._best_theta = default_theta

        self._read_config(config_file)
        self._init_estimator(self.config)
        self._best_cost = self.calc_real_cost(self._best_models)

    def _read_config(self, config_file: str = None):
        """
            Load configuration data.
            Note: we simply set values currently.
                  We are going to load those data from files in the future.
        """
        self.config = Config(config_file)
        model_list: List[str] = [
                                  "gpt-4o",
                                  "qwen-max",
                                  "qwen-2.5",
                                  "gpt-4-turbo",
                                  "gpt-3.5",
                                 ]

        inprice_list: List[float]  = [0.0025, round(0.02/7.19, 4), round(0.004/7.19, 4), 0.01, 0.0010]
        outprice_list: List[float] = [0.01, round(0.06/7.19, 4), round(0.012/7.19, 4), 0.03, 0.002]

        # print(inprice_list, outprice_list)
        self.config.set_num_steps(self.num_steps)
        for index in range(len(model_list)):
            self.config.set_model_price(model_list[index], inprice_list[index],
                                 outprice_list[index])

    def _init_estimator(self, config: Config):
        self.estimator = CostEstimator(config)

        # estimate #tokens
        data_df = self.single_model_map.dataframe() 
        for index, row in data_df.iterrows():
            models = row['models']
            prompt_tokens = row['prompt_tokens']
            completion_tokens = row['completion_tokens']
            for step_idx in range(len(models)):
                self.estimator.update_model2token_dict(step_idx,
                                                       models[step_idx],
                                                       prompt_tokens[step_idx],
                                                       completion_tokens[step_idx])

    def preprocess(self):
        """ Model profiling: construct model combinations and
            collect runtime metrics, e.g., accuracy, #tokens.
            
            Currently, we directly loading profiling data from file, rather than
            executing user program because that takes long time
        """

        #1. construct model combinations using the same single model
        comb_list = []
        for each_model in self.model_list:
            cur_comb = []
            for index in range(self.num_steps):
                cur_comb.append(each_model) 
            comb_list.append(cur_comb)

        # for reproducibility, we preemptively ran all LLM combinations and read examined
        # results from files
        assert self.results_map is not None
        # construct model combinations using the same model and collect data
        self.single_model_map: DataMap = self.results_map.load_partial_map(comb_list)
        print("== single_map")
        self.single_model_map.print()

    def reset_the_best_without_relaxation(self):
        self._best_theta = 0

    def get_accuracy_objective(self) -> float:
        return round(self._best_acc - self._best_theta, 4)

    def get_accuracy_bound(self) -> float:
        return round(self._best_acc - self._lamda * self._best_theta, 4)

    def get_the_best_single_model(self) -> List[str]:
        return self._best_models

    def get_accuracy_and_cost(self, model_list: List[str]) -> tuple[float, float]:
        acc = 0.0
        intoken_list = None
        outtoken_list = None

        # for reproducibility, we preemptively ran all LLM combinations and read examined
        # results from files
        assert self.exec_mode is False
        acc, intoken_list, outtoken_list = self.results_map \
            .get_accuracy_and_tokens(model_list)
        cost = self.estimator.calc_cost(model_list, intoken_list, outtoken_list)
        return acc, cost
    
    def eval_accuracy(self, model_list: List[str]) -> float:
        res_acc, res_cost = self.get_accuracy_and_cost(model_list)
        return res_acc

    def calc_real_cost(self, model_list: List[str]) -> float:
        res_acc, res_cost = self.get_accuracy_and_cost(model_list)
        return res_cost 

    def estimate_cost(self, model_list: List[str]) -> float:
        if self.cost_est is False:
            return self.calc_real_cost(model_list)
        else:
            return self.estimator.estimate_cost_by_models(model_list)

    def steps(self) -> int:
        return self.num_steps
    
    def candidates(self) -> List[str]:
        return self.model_list

    def get_single_models_list(self) -> List[List[str]]:
        return self.single_model_map.get_models_list() 

    def get_good_model_list(self) -> List[str]:
        single_models_list = self.get_single_models_list()
        accuracy_bound = self.get_accuracy_bound()
        good_model_list = []
        for model_list in single_models_list:
            accuracy = self.eval_accuracy(model_list)
            if not self.good_model_enable or accuracy > accuracy_bound:
                good_model_list.append(model_list[0])
        return good_model_list

    def get_old_trials(self, data_map: DataMap = None) ->tuple[List[List[str]], List[float], List[float]]:
        models_list: List[List[str]] = []
        accuracy_list: List[float] = []
        cost_list: List[float] = []

        if data_map is None:
            data_map = self.single_model_map

        df = data_map.dataframe()
        for index, row in df.iterrows():
            model = row['models']
            acc = row['acc_mean']
            prompt_tokens = row['prompt_tokens']
            completion_tokens = row['completion_tokens']
            cost = self.estimator.calc_cost(model, prompt_tokens, completion_tokens)
            models_list.append(model)
            accuracy_list.append(acc)
            cost_list.append(cost)
        return models_list, accuracy_list, cost_list

    def benefit_to_best_single(self, acc: float, cost: float) \
        -> tuple[float, float]:
        """ Evaluate benefit to the best single models.

            Return:
                the accuracy improvements
                the cost savings
        """ 
        acc_improved: float = round(acc - self._best_acc, 1)
        cost_saving: float = round((self._best_cost-cost) / self._best_cost * 100, 1)
        return acc_improved, cost_saving

    def get_result_map(self, per_query_cost: bool = True):
        """
            Print accuracy and cost for all result combinations.
            If per_query_cost is True, then we calculate the average cost of a query.
        """
        result_df = pd.DataFrame(columns=["models","single","accuracy","cost"])
        data_map = self.results_map.dataframe()
        num_inputs = self.results_map.num_inputs
        for index, row in data_map.iterrows():
            model = row['models']
            acc = row['acc_mean']
            prompt_tokens = row['prompt_tokens']
            completion_tokens = row['completion_tokens']
            cost = self.estimator.calc_cost(model, prompt_tokens, completion_tokens)
            if per_query_cost is True:
                cost = cost / num_inputs
            single = 1 if all(x == model[0] for x in model) else 0
            result_df.loc[index] = [model, single, acc, cost]
        return result_df

    def get_num_inputs(self):
        assert self.results_map is not None
        return self.results_map.num_inputs


    def print(self):
        #self.results_map.print()
        print("------------------------------------------------")
        print("Prob. ")
        print(f"\t a={self._best_acc}, theta = {self._best_theta}")
        print(f"\t objective = {self.get_accuracy_objective()}")
        print(f"\t bound = {self.get_accuracy_bound()}")
        print(f"\t best single models: {self._best_models}, accuracy:{self._best_acc}, cost: {self._best_cost}")
        print(f"\t result map: ")
        models_list, accuracy_list, cost_list = self.get_old_trials(self.results_map)

        acc_space = 0
        cost_space = 0
        space_count = 0
        for index in range(len(models_list)):
            acc_improved, cost_saving = self.benefit_to_best_single(accuracy_list[index], cost_list[index])
            if accuracy_list[index] >= self.get_accuracy_objective():
                acc_space += acc_improved
                cost_space += cost_saving
                space_count += 1
                
            print(f"\t\t - {models_list[index]}, {accuracy_list[index]}, {cost_list[index]},\t\t acc_improved: {acc_improved}(%), cost_saving: {cost_saving}(%)")
        if space_count > 0:
            print(f"improve space# acc: {round(acc_space/space_count,2)}%, cost: {round(cost_space/space_count, 2)}%")
        print("------------------------------------------------")  
