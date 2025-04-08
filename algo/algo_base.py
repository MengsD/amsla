#-------------------------------------------------------------------------------
# @description
#   Algo Base Class

# @author 
#   Mengsu Ding<dingmengsu.dms@alibaba-inc.com>
#
#-------------------------------------------------------------------------------


from algo.context import *

class Algo(ABC):
    def __init__(self, ctx: Context):
        self.num_iterations = 0
        self.ctx = ctx
        self.history = [] # record history, for debug

    def get_num_iterations(self) -> int:
        return self.num_iterations

    def get_history_cost_usage(self, printh: bool=True) -> float:
        cost = 0
        for index in range(len(self.history)):
            if printh is True:
                cur_acc, cur_cost = self.ctx.get_accuracy_and_cost(self.history[index])
                cur_acc_improved, cost_saving = self.ctx.benefit_to_best_single(cur_acc, cur_cost)
                print(f"\thistory: {self.history[index]},\t acc: {cur_acc}, \tcost:{cur_cost}\t ecost:{self.ctx.estimate_cost(self.history[index])}\t acc_improved:{cur_acc_improved}%,\t cost_saving:{cost_saving}%")
            cost += self.ctx.calc_real_cost(self.history[index])
        extra_cost_ratio = round(cost/self.ctx._best_cost, 2)
        cost = round(cost,2)
        return extra_cost_ratio, cost
    
    def add_trace(self, model_list: List[str]) -> float:
        self.num_iterations += 1 
        self.history.append(model_list) 
 
