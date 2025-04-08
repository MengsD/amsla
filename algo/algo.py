#-------------------------------------------------------------------------------
# @description
#  To ensure the reproducibility of our experiments and avoid the impact of LLM 
#  stochasticity on performance, we preemptively ran the full set of LLM 
#  combinations across all evaluated tasks (cf. Table 3) and repeated five times. 
#  Based on these runs, all algorithms obtain the same average accuracy and cost 
#  of LLM combinations for each corresponding task.
#
#  @see TGS                 -- proposed Tree-based Greedy Search
#  @see FullEnumerationAlgo -- baseline, Brute Force Search
#  
#
# @author 
#  Mengsu Ding<dingmengsu.dms@alibaba-inc.com>
#
#-------------------------------------------------------------------------------




from algo.context import *
from algo.util import *
from algo.algo_base import *
import random
import sys
import copy

           
        

class FullEnumerationAlgo(Algo):
    """ Enumerate the whole space to find the optimal answer. """
    def __init__(self, ctx: Context):
        Algo.__init__(self, ctx)

    def get_the_best(self):
        best_single_model = self.ctx.get_the_best_single_model()
        res_model_list = list(best_single_model)
        res_acc, res_cost = self.ctx.get_accuracy_and_cost(res_model_list)

        def update(model_list: List[str], accuracy: float, cost: float):
            nonlocal res_model_list, res_cost, res_acc
            if accuracy >= self.ctx.get_accuracy_objective():
                if res_cost > cost:
                    res_cost = cost
                    res_acc = accuracy
                    res_model_list.clear()
                    res_model_list.extend(model_list)
            
        def check_equal(path):
            return all(x == path[0] for x in path)
        
        def backtrack(path):
            if len(path) == self.ctx.num_steps:
                self.add_trace(copy.deepcopy(path))
                if check_equal(path) is False:
                    acc, cost = self.ctx.get_accuracy_and_cost(path)
                    update(path, acc, cost)
                return
            for cur_model in self.ctx.candidates():
                path.append(cur_model)
                backtrack(path)
                path.pop()
        backtrack([])
        if res_acc >= self.ctx.get_accuracy_objective():
            return res_model_list, res_acc, res_cost
        return None, None, None

