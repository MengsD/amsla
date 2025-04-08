#-------------------------------------------------------------------------------
# @description
#  To ensure the reproducibility of our experiments and avoid the impact of LLM 
#  stochasticity on performance, we preemptively ran the full set of LLM 
#  combinations across all evaluated tasks (cf. Table 3) and repeated five times. 
#  Based on these runs, all algorithms obtain the same average accuracy and cost 
#  of LLM combinations for each corresponding task.
#
#  TGS: Tree-based Greedy Search
#  
#
# @author 
#  Mengsu Ding<dingmengsu.dms@alibaba-inc.com>
#
#-------------------------------------------------------------------------------





from enum import Enum
import random
import math
import numpy as np

from algo.context import *
from algo.util import *
from algo.algo_base import *

DEBUG_LOG = False

class NodeState(Enum):
    CLOSED      = 1    # explored nodes, or pruned nodes with high confidence
    INACTIVE    = 2    # pruned nodes, with low confidence
    ACTIVE      = 3    # waiting to be selected


class Node:
    def __init__(self, 
                 model,
                 parent=None):
        self.model: list[str]  = model
        self.parent            = parent
        self.children          = []
        self.visit_count       = 0
        self.win_count         = 0
        self.is_expanded       = False
        self.state             = NodeState.ACTIVE
        self.eval_rate         = 0

        self.real_cost : float = None
        self.accuracy  : float = None

    def set_expanded(self):
        self.is_expanded = True

    def is_closed(self):
        return self.state == NodeState.CLOSED


    def state_backtrace(self):
        """ 
            To avoid selecting a path that fails to lead to active nodes, we have to update
            all nodes in the path according to its children.
            @see _uct_select, ucb_score

            This is triggered if the state of a node changes.
        """
        parent = self.parent
        if parent == None:
            return 0

        # if the node is updated to be ACTIVE (INACTIVE -> ACTIVE), then its parent should be ACTIVE
        if self.state == NodeState.ACTIVE:
            assert parent.state != NodeState.CLOSED, "illegal parent state: CLOSED"
            if parent.state == NodeState.INACTIVE:
                # activate parent
                parent.state = NodeState.ACTIVE
                return parent.state_backtrace()

        # if the node is updated to be INACTIVE (ACTIVE -> INACTIVE), then check whether to update its parent to be INACTIVE
        # when all children of the parent are INACTIVE or CLOSED, then change the parent to be INACTIVE
        elif self.state == NodeState.INACTIVE:
            assert parent.state == NodeState.ACTIVE, f"illegal parent state: {parent.state}"
            parent.update_state_with_children()

            assert parent.state != NodeState.CLOSED, f"illegal parent state: CLOSED"
            if parent.state == NodeState.INACTIVE:
                # parent from active to impossible, continue backtracing
                return parent.state_backtrace()
        else:
            # if the node is updated to be CLOSED (ACTIVE/INACTIVE -> CLOSED), then check whether to change the state of its parent
            assert parent.state != NodeState.CLOSED, "illegal parent state: CLOSED"
            if parent.update_state_with_children() is True:
                return parent.state_backtrace()


    def update_state_with_children(self) -> bool:
        """
            Update the state of the node according to states of children.
            If there is an active child, then the node is ACTIVE.
            If all children are CLOSED, then the node is CLOSED.
            Otherwise, the node is INACTIVE.

            Return:
                True if the node state is changed, otherwise return False
        """
        orig_state = self.state
        self.state = NodeState.CLOSED
        for child in self.children:
            if child.state == NodeState.INACTIVE:
                self.state = NodeState.INACTIVE
                continue
            elif child.state == NodeState.ACTIVE:
                self.state = NodeState.ACTIVE
                break
        return orig_state != self.state


    def is_expanded_leaf(self):
        return self.is_expanded and len(self.children) == 0

    def ucb_score(self, coef, parent_visit_count):
        if self.state != NodeState.ACTIVE:
            return 0
        if self.visit_count == 0:
            return float('inf')
        win_rate = self.win_count / self.visit_count
        exploration = coef * math.sqrt(math.log(parent_visit_count) / self.visit_count)
        return win_rate + exploration

    
    def dump(self, level=0):
        # if level > 4:
        #     print("too many level!")
        #     return
        print(" " * (level * 4) + self.to_str())
        for child in self.children:
            child.dump(level + 1)

    def to_str(self):
        node_str = f"Node# model:{self.model}, ecost: {self.ctx.estimate_cost(self.model)}, child: {self.children}, vc:{self.visit_count}, state: {self.state}, eval: {self.eval_rate}, wc:{self.win_count}, is_expanded:{self.is_expanded}"
        return node_str



class TGSAlgo(Algo):
    def __init__(self, ctx, 
                 explore_coef: float = math.sqrt(2)):
        Algo.__init__(self, ctx)
        self.ctx = ctx
        self.accuracy_objective = self.ctx.get_accuracy_objective()
        self.model_list = self.ctx.get_good_model_list()
        self.explore_coef = explore_coef

        self.node_map: dict[str, Node] = {}
        self.satisfied_list = []
        self.nodes_in_tree = []
        self.cost_reduce_cand_list = []

        self.replace_map = {}      # evaluate accuracy variance after replace
        self.cache_map = {}        # all expanded model_list
        self.impossible_list = {}  # impossible modle on a cetrain step
        for step in range(self.ctx.num_steps):
            self.impossible_list[step] = []

        self.optimal = None
        self.root = self._create_node(self.ctx.get_the_best_single_model())
        self._init_candidate_list() 


    def run(self):
        """ Run TGS Algorithm. It terminates automatically. """

        while self.to_stop() is False:
            node = self.select_node()

            if node.state == NodeState.ACTIVE and node.is_expanded_leaf() is False:
                node = self.examine_node(node)

                # update the optimal values found so far
                if self.is_satisfied(node):
                    if node.real_cost < self._get_minimum_cost():
                        self.optimal = node

                node = self.expand_node(node)
         
            self.update_node(node)

        if DEBUG_LOG:
            print("Finished!")
            self.root.dump()
        return self.get_best_combination()
       

    def to_stop(self) -> bool:
        if self._early_stop_by_acc() or self._early_stop_by_cost():
            return True
        return False


    def select_node(self) -> Node:
        """ 
            Greedily select a promising node. 
            Our two-phase greedy selection strategy is optimized to prioritize 
            exploring nodes that may reduce cost when there is an opportunity,
            since we have identified and pruned nodes that may decrease accuracy.
        """
        if len(self.cost_reduce_cand_list) > 0:
            min_cost_node = min(self.cost_reduce_cand_list, key=lambda node: self.ctx.estimate_cost(node.model))
            self.cost_reduce_cand_list.remove(min_cost_node)
            return min_cost_node

        else:
            node = self._select(self.root)
        return node


    def examine_node(self, node) -> Node:
        node.accuracy, node.real_cost = self.ctx.get_accuracy_and_cost(node.model)
        self._cache_and_update(node.model, node.accuracy, node.real_cost)
        self._update_action_trend(node.model)
        return node


    def expand_node(self, node):
        """ Expand the current node by replace i-th LLM """

        if DEBUG_LOG:
            print(f"####expand node={node.model}, accuracy={accuracy}, ecost={self.ctx.estimate_cost(node)}, neighbor_eval={neighbor_eval}") 

            print(f"node.accuracy: {node.accuracy}, self.accuracy_objective: {self.accuracy_objective}")

        replace_list = self._get_replace_list(node)
        for step in range(self.ctx.steps()):
            for model in replace_list:
                if model == node.model[step]:
                    continue
                new_model_list = list(node.model)
                new_model_list[step] = model
                if new_model_list in self.nodes_in_tree: # avoid repeated node
                    continue
                if self._check_cycle(node, new_model_list):
                    continue
                child = self._create_node(new_model_list)
                child.parent = node
                node.children.append(child)

                if self.is_satisfied(node) and child.state != NodeState.CLOSED:
                    self.cost_reduce_cand_list.append(child)

        node.set_expanded()
        if node.update_state_with_children() is True:
            node.state_backtrace()
        # print("-----tree after expand-----")
        # self.root.dump()
        return node


    def update_node(self, leaf):
        # update win_count and visit_count, for UCT select
        win = leaf.model in self.satisfied_list
        while leaf is not None:
            leaf.visit_count += 1
            if win:
                leaf.win_count += 1
            leaf = leaf.parent

        # update status
        self.update_states(leaf)


    def is_satisfied(self, node: Node) -> bool:
        """ Whether the given node satisfies condition. """
        assert node.accuracy is not None
        return node.accuracy >= self.accuracy_objective


    def get_best_combination(self):
        if self.optimal is None:
            return None, None, None
        else:
            return self.optimal.model, self.optimal.accuracy, self.optimal.real_cost

        
    def update_states(self, node):
        unexpanded_model_lists = self.get_unexpanded_model_lists()
        for model_list in unexpanded_model_lists:
            mnode = self._get_node(model_list)

            if self.node_state_update(mnode):
                mnode.state_backtrace()
                    
            # remove nodes with higher cost
            if mnode.is_closed():
                if mnode in self.cost_reduce_cand_list:
                    self.cost_reduce_cand_list.remove(mnode)
    

    def node_state_update(self, node: Node) -> bool:
        cur_traverse_state = node.state
        if self.is_promising(node.model):
            node.state = NodeState.ACTIVE
        else:
            node.state = NodeState.INACTIVE
        if self.ctx.estimate_cost(node.model) >= self._get_minimum_cost() or self._check_impossible_steps(node.model):
            node.state = NodeState.CLOSED
        node.eval_rate = self._eval_with_neighbor(node.model)
        return cur_traverse_state != node.state

    def _get_node(self, state: List[str]):
        res_node: Node = None
        model_key = ",".join(state)
        if model_key in self.node_map:
            res_node = self.node_map[model_key]
        return res_node

    def _create_node(self, model_list: List[str]):
        node = Node(model_list)
        model_key = ",".join(model_list)
        self.node_map[model_key] = node
        self.nodes_in_tree.append(model_list)
        self.node_state_update(node)
        return node

       
    def _get_minimum_cost(self) -> float:
        if self.optimal is None:
            return float('inf')
        else:
            return self.optimal.real_cost

    def _select(self, node):
        while node.is_expanded_leaf() is False:
            if node.is_expanded is False:
                return node
            else:
                node = self._uct_select(node)
        return node

    def _has_cycle_dfs(self, node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)

        for child in node.children:
            if child not in visited:
                if self._has_cycle_dfs(child, visited, rec_stack):
                    return True
            elif child in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    def _check_cycle(self, node, state):
        node_temp =  Node(state)
        node.children.append(node_temp)
        visited = set()
        rec_stack = set()
        has_cycle = self._has_cycle_dfs(node, visited, rec_stack)
        node.children.pop()
        return has_cycle

    def is_promising(self, model_list) -> bool:
        """ TGS optimistically regards a node as a promising node
            unless the number of negative neighbors exceeds twice
            that of positive neighbors. 
            @see (vote + 1) / (vote_total + 2)
        """

        is_promising_THRESHOLD = 0.33
        return self._eval_with_neighbor(model_list) >= is_promising_THRESHOLD

    def _uct_select(self, node):
        return max(node.children, key=lambda child: child.ucb_score(self.explore_coef, node.visit_count))

    def get_unexpanded_model_lists(self):
        ret = []
        for model_list in self.nodes_in_tree:
            if ",".join(model_list) in self.cache_map:
                continue
            node = self._get_node(model_list)
            if node is not None and node.is_closed():
                continue
            ret.append(model_list)
        return ret
    
    def _early_stop_by_acc(self) -> bool:
        """ Return false if there is an unexplored node which has potential to satisifiy accuracy. """
        unexpanded_model_lists = self.get_unexpanded_model_lists()
        for model_list in unexpanded_model_lists:
            if self.is_promising(model_list):
                return False
        return True
    
    def _early_stop_by_cost(self) -> bool:
        """ Return false if there is an unexplored node which has potential to reduce cost. """
        unexpanded_model_lists = self.get_unexpanded_model_lists()
        for model_list in unexpanded_model_lists:
            if self.ctx.estimate_cost(model_list) >= self._get_minimum_cost():
                continue
            else:
                return False
        return True
    
    
   
    def _get_replace_list(self, node):
        if self.is_satisfied(node):
            return self.cand_list_by_cost
        else:
            return self.cand_list_by_acc

    def _init_candidate_list(self):
        candidate_list = self.ctx.get_good_model_list()
        cost_list = candidate_list[:]
        accuracy_list = candidate_list[:]

        def evaluate_cost(model):
            return self.ctx.estimate_cost([model, model, model])

        def evaluate_accuracy(model):
            return self.ctx.eval_accuracy([model, model, model])

        cost_list.sort(key=evaluate_cost)
        accuracy_list.sort(key=evaluate_accuracy, reverse=True)

        self.cand_list_by_cost = cost_list
        self.cand_list_by_acc = accuracy_list

        # print("cost_list:", self.cand_list_by_cost)
        # print("accuracy_list:", self.cand_list_by_acc)
    

    def _cache_and_update(self, model_list, acc, cost):
        key = ",".join(model_list)
        self.cache_map[key] = {'acc':acc, 'cost':cost}
        self.add_trace(model_list)
        if acc >= self.ctx.get_accuracy_objective():
            self.satisfied_list.append(model_list)


    def _get_neighbos(self, model_list):
        neighbor_list = []
        for step in range(self.ctx.steps()):
            for model in self.model_list:
                if model == model_list[step]:
                    continue
                neighbor = model_list.copy()
                neighbor[step] = model
                neighbor_list.append({'list':neighbor, 'step':step})
        return neighbor_list


    def _check_replace_vote(self, replace) -> bool:
        return not (replace['neg_replace_cnt'] * 2 > replace['cnt'])

    def _eval_with_neighbor(self, model_list) -> float:
        """
            Evalute the accuracy of the model_list with its neighbors.

            Return:
                2 if any neighbor is satisfied
                1.1 if not any neighbor is cached
                ratio of neighbors that agree with this replacement, otherwise
            
        """

        vote = 0.0
        vote_total = 0.0
        neighbor_list = self._get_neighbos(model_list)
        if self._check_impossible_steps(model_list):
            return 0
        for neighbor in neighbor_list:
            if neighbor['list'] in self.satisfied_list:
                return 2
            step = neighbor['step']

            replace_key = f"{step}_{neighbor['list'][step]}_{model_list[step]}"
            if replace_key not in self.replace_map:
                continue
            vote_total += 1
            replace = self.replace_map[replace_key]
            if self._check_replace_vote(replace):
                vote += 1
        if vote_total == 0:
            return 1.1
        else:
            # we use Laplace Smoothing to mitigate estimation biases caused by zero samples
            return (vote + 1) / (vote_total + 2)

    def _check_impossible_steps(self, model_list):
        for step in range(self.ctx.num_steps):
            if model_list[step] in self.impossible_list[step]:
                return True

    def _update_action_trend(self, model_list):
        """ 
            Compute and update 'action' trend.
            Note: an action Act(v, i, m) replace the LLM in i-th stage of node v to LLM m
        """
        neighbor_list = self._get_neighbos(model_list)
        for neighbor in neighbor_list:
            step = neighbor['step']
            key_prev = ",".join(neighbor['list'])
            if key_prev in self.cache_map:
                if self.model_list.index(neighbor['list'][step]) < self.model_list.index(model_list[step]):
                    model_list_prev = neighbor['list']
                    model_list_new = model_list
                else:
                    model_list_prev = model_list
                    model_list_new = neighbor['list']
                key = f"{step}_{model_list_prev[step]}_{model_list_new[step]}"
                # key = ",".join(key_list)
                if key not in self.replace_map:
                    self.replace_map[key] = {
                        'cnt': 0,
                        'neg_replace_cnt': 0,
                        'changes': {},
                    }
                self.replace_map[key]['cnt'] += 1
                change = self.ctx.eval_accuracy(model_list_new) - self.ctx.eval_accuracy(model_list_prev)
                change_key = ",".join(model_list_prev) + "_" + ",".join(model_list_new)
                self.replace_map[key]['changes'][change_key] = change
                if change < 0:
                    self.replace_map[key]['neg_replace_cnt'] += 1
                
                IMPOSSIBLE_STEP_THRESHOLD = 10
                if change > IMPOSSIBLE_STEP_THRESHOLD:
                    self.impossible_list[step].append(model_list_prev[step])
                if change < -IMPOSSIBLE_STEP_THRESHOLD:
                    self.impossible_list[step].append(model_list_new[step])

    def dump_repalace_map(self):
        print("--------------------dump replace map--------------------")
        for key, value in self.replace_map.items():
            print(f"{key:<20}: ", end=' ')
            for change_key, change in value['changes'].items():
                print(f"{change:>6.2}", end=' ')
                # print(f"{change_key}:{change:>6.2}", end=' ')
            print()

    def dump_acc_change(self):
        pair_list = []
        for i in range(len(self.cand_list_by_acc)):
            for j in range(i+1, len(self.cand_list_by_acc)):
                pair_list.append([self.cand_list_by_acc[i], self.cand_list_by_acc[j]])
        
        def _get_rest_pos(pos):
            if (pos == 0): 
                static_pos = [1, 2]
            elif (pos == 1):
                static_pos = [0, 2]
            else:
                static_pos = [0, 1]
            return static_pos

        result = []
        for pair in pair_list:
            for pos in range(3):
                rest_pos = _get_rest_pos(pos)
                res = []
                res.append(f"{pos:<2}:")
                res.append(f"{pair[0]:<8} to {pair[1]:<8}")
                for model1 in self.cand_list_by_acc:
                    for model2 in self.cand_list_by_acc:
                        model_sel_from = ["", "", ""]
                        model_sel_from[pos] = pair[0]
                        model_sel_from[rest_pos[0]] = model1
                        model_sel_from[rest_pos[1]] = model2
                        model_sel_to = model_sel_from.copy()
                        model_sel_to[pos] = pair[1]
                        acc_delta = self.ctx.eval_accuracy(model_sel_to) - self.ctx.eval_accuracy(model_sel_from)
                        # res.append(f"{acc_delta:>6.2f}")
                        res.append(acc_delta)
                result.append(res)
                
                for item in res:
                    if isinstance(item, float):
                        print(f"{item:>6.2f}", end=' ')
                    else:
                        print(f"{item}", end=' ')
                print(f"")
