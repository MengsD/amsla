#-------------------------------------------------------------------------------
# @description
#  Class CostEstimator is designed to estimate cost function.
#
#  UT cases are provided.
#
# @author 
#  Mengsu Ding<dingmengsu.dms@alibaba-inc.com>
#
#-------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import List
import unittest
import sys

from algo.util import Config


class CostEstimator:
    def __init__(self, config: Config):
        self.config = config
        self.model2token_dict: dict[tuple[int, str], dict[str, int]] = {}

    def estimate_cost_by_models(self, model_list: List[str]) -> float:
        cost: float = 0
        intoken_list, outtoken_list = self._get_estimated_tokens(model_list)
        return self.calc_cost(model_list, intoken_list, outtoken_list)

    def calc_cost(self, model_list: List[str], 
                             intoken_list: List[int],
                             outtoken_list: List[int]) -> float:
        cost: float = 0
        for step in range(0, len(model_list)):
            inprice_1k, outprice_1k = self.config.get_model_price(model_list[step])
            avg_intoken  = intoken_list[step]
            avg_outtoken = outtoken_list[step]
            cost += (avg_intoken*inprice_1k + avg_outtoken*outprice_1k)
        cost = round(cost / 1000, 2)
        return cost

    def update_model2token_dict(self,
                                step_id: int,
                                model: str,
                                num_est_intoken: int,
                                num_est_outtoken: int):
        # 1. create a new item
        entry = {
            'step_id': step_id,
            'model': model,
            'num_est_intoken': num_est_intoken,
            'num_est_outtoken': num_est_outtoken
        }

        #2. build index
        key = (entry['step_id'], entry['model'])
        self.model2token_dict[key] = entry
        # TODO: to support incremental update -> update existing models


    def _get_estimated_tokens(self, model_list: List[str]) \
        -> tuple[List[int], List[int]]:
        """ Get estimated input token and output token for all models.
            
            Return:
                input token list for all models
                output token list for all models
        """

        num_est_intokens = [0,0,0]
        num_est_outtokens = [0,0,0]
        for index in range(len(model_list)):
            found = self.model2token_dict.get((index, model_list[index]))
            if found is False:
                print(f"ERROR: some model(s) in {model_list} doesn't exist(s).")
                sys.exit()
            else:
                num_est_intokens[index] = found['num_est_intoken']
                num_est_outtokens[index] = found['num_est_outtoken']
        return num_est_intokens, num_est_outtokens


    def get_estimated_cost(self, model_list: List[str]) -> float:
        """ Get estimated total cost for given models in a llm-based pipeline. 
            The function should be called after calling estimate().

        Args:
            model_list: models to run in a llm-based pipeline
            
        Return:
            total cost
        """

        cost_per_record: float = 0
        assert self.config.num_steps is not None, "please set num_steps in config file"
        for step in range(0, self.config.num_steps):
            avg_intoken, avg_outtoken = self.metric_coll.get_estimated_token_size(step, model_list[step])
            inprice_1k, outprice_1k = self.config.get_model_price(model_list[step])
            cost_per_record += (avg_intoken*inprice_1k + avg_outtoken*outprice_1k)
        cost_per_record = cost_per_record / 1000
        return cost_per_record


    def not_greater_than_budget(self, model_list: List[str]) -> float:
        return self.get_estimated_cost(model_list) <= self.config.budget

    def not_greater_than_budget(self, cost: float) -> bool:
        return cost <= self.config.budget


# ----------------------------------------------------------------------------- 
# Unit Test Cases

class TestCostEstimator(unittest.TestCase):
    """ Init test cases for CostEstimator """

    def test_dict_read_and_search(self):
        #1. init
        model_list: List[str]      = [
                                      "gpt-4o",
                                      "qwen-max",
                                      "qwen-2.5",
                                     ]
        conf : Config = Config(None)
        inprice_list: List[float]  = [0.0025, round(0.02/7.19, 4), round(0.004/7.19, 4)]
        outprice_list: List[float] = [0.01, round(0.06/7.19, 4), round(0.012/7.19, 4)]
        for index in range(len(model_list)):
            conf.set_model_price(model_list[index], inprice_list[index],
                                 outprice_list[index])
        estimator = CostEstimator(conf)

        #2. build dict and search
        model_list: List[List[str]] = [
                                        ["gpt-4o", "gpt-4o", "gpt-4o"],
                                        ["qwen-max", "qwen-max", "qwen-max"],
                                        ["qwen-2.5", "qwen-2.5", "qwen-2.5"],
                                      ]
        prompt_tokens: List[int] = [
                                     [468246, 391507, 41736],
                                     [472813, 401285, 42642],
                                     [472813, 411527, 42839]
                                   ] 
                                     

        completion_tokens: List[int] = [
                                         [18430, 19174, 2717],
                                         [19818, 20091, 2973],
                                         [15249, 20037, 3178],
                                       ]

        final_cost_list=[0,0,0]
        for index in range(len(model_list)):
            for midx in range(3): 
                estimator.update_model2token_dict(midx,
                                                  model_list[index][midx],
                                                  prompt_tokens[index][midx],
                                                  completion_tokens[index][midx])
            final_cost_list[index] = estimator.estimate_cost_by_models(model_list[index])
        assert final_cost_list == [2.66, 2.92, 0.62]
         
 


if __name__ == '__main__':

    unittest.main()
