from datetime import datetime
from algo.context import *
from algo.util import *
from algo.tgs import *
from algo.algo import *
from algo.hpo_algo import *

import random
import sys,re
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

input_file_list = [
       "./input/dinsql_bird_all.json",              # DIN_SQL 
       "./input/dinsql_spider_all.json",            # DIN_SQL
       "./input/macsql_spider_all.json",            # MAC_SQL
       "./input/macsql_bird_all.json",              # MAC_SQL
       "./input/unidm_impute_restaurant_all.json",  # UNIDM
       "./input/lotus_factool_all.json",            # LOTUS
       "./input/esql_bird_all.json",                # E_SQL
       "./input/esql_spider_all.json",              # E_SQL
   ]

exp_output_dir = "./exp_output/vldb26/"


def print_result_and_benefit(algo_name: str,
                             res_model_list: List[str],
                             res_acc: float,
                             res_cost: float,
                             num_iters: int,
                             acc_improved: float,
                             cost_saving: float,
                             history_cost_usage: float
                            ):
    print(f"[{algo_name}]# {res_model_list}, accuracy: {res_acc}, cost: {res_cost}, #iterations: {num_iters}")
    print(f"\t - benefit: {acc_improved}% (accuracy_improved), {cost_saving}% (cost_saving)")
    print(f"\t - history cost usage: {history_cost_usage}x (up)")


def init_context(num_sptes: int, data_file: str, theta = None, good_modle_enable = False) -> Context:
    model_list: List[str]      = [
                                  "gpt-4o",
                                  "qwen-max",
                                  "qwen-2.5",
                                  "gpt-4-turbo",
                                  "gpt-3.5",
                                 ]
    lamda = 100  # we don't filter models by bounds, because model space is small
    ctx: Context = None
    if theta is None:
        ctx = Context(lamda=lamda, good_modle_enable=good_modle_enable)
    else:
        ctx = Context(lamda=lamda, theta=theta, good_modle_enable=good_modle_enable)
    ctx.set_app(3, model_list, data_file)
    ctx.print()
    ctx.single_model_map.print()
    
    return ctx

def __set_result_df(ctx: Context,
                    algo: Algo,
                    algo_name: str,
                    result_df,
                    res_model_list: list[str],
                    res_acc,
                    res_cost):
    if res_model_list is None:
        cost_saving    = "NaN" #np.nan
        acc_improved   = "NaN" #np.nan
        res_acc        = "NaN"
        res_cost       = "NaN"
        cost_per_query = "NaN" #np.nan
        history_usage  = "NaN" #np.nan
        histot_cost    = "NaN" #np.nan
        num_iter       = "NaN" #np.nan
        model_str      = "NaN" #np.nan
    else:
        acc_improved, cost_saving = ctx.benefit_to_best_single(res_acc, res_cost)
        cost_per_query = round(res_cost/ctx.get_num_inputs(), 6)
        history_usage, histot_cost = algo.get_history_cost_usage()
        num_iter = algo.get_num_iterations()
        model_str = ",".join(res_model_list)

    theta = ctx._best_theta

    result_df.loc[len(result_df)] = [algo_name, theta, num_iter, cost_saving, acc_improved, res_acc, res_cost, cost_per_query, histot_cost, history_usage, model_str]
    return result_df

                     


def full_enumeration(ctx: Context, result_df, algo_name):
    bfs_algo = FullEnumerationAlgo(ctx)
    bfs_model_list, bfs_acc, bfs_cost =  bfs_algo.get_the_best()
    result_df = __set_result_df(ctx, bfs_algo, algo_name, result_df, bfs_model_list, bfs_acc, bfs_cost)
    return result_df

def tgs(ctx: Context, result_df, algo_name):
    algo = TGSAlgo(ctx)
    tgs_model_list, tgs_acc, tgs_cost = algo.run()
    result_df = __set_result_df(ctx, algo, algo_name, result_df, tgs_model_list, tgs_acc, tgs_cost)
    return result_df


def bo(ctx: Context, result_df,
       num_trial: int,
       algo_name: str
      ):

    bo_moo = HPO_MOO(ctx, num_trial)
    res_model_list, res_acc, res_cost = bo_moo.optimize()
    result_df = __set_result_df(ctx, bo_moo, algo_name, result_df, res_model_list, res_acc, res_cost)

    return result_df



def run_specific_algo(ctx: Context, 
                      algo_name: str,
                      result_df: pd.DataFrame,
                      num_trial: int = None):
    if algo_name == "TGS":
        print(f"\n==== TGS: Tree-based Greedy Search ====")
        result_df = tgs(ctx, result_df, algo_name)
    elif algo_name == "BFS":
        print(f"\n==== BFS ====")
        result_df = full_enumeration(ctx, result_df, algo_name)
    elif algo_name == "BO":
        assert num_trial is not None
        print(f"\n==== BO-{num_trial}====")
        result_df = bo(ctx, result_df, num_trial, f"{algo_name}-{num_trial}")
    else:
        SystemExit(f"Wrong algo name: {algo_name}")

    return result_df
        
    

def run_all_algos(ctx: Context):
    result_df = pd.DataFrame(columns=["algo_name", 
                                      "theta", 
                                      "num_iters", 
                                      "cost_saving",
                                      "acc_improved",
                                      "accuracy",
                                      "cost",
                                      "cost_per_query",
                                      "search_cost",
                                      "search_cost_ratio",
                                      "model"])

    result_df = run_specific_algo(ctx, "BFS", result_df)

    result_df = run_specific_algo(ctx, "TGS", result_df)

    # BOS
    #num_trial_list = [10, 20, 30, 40, 50]
    num_trial_list = [10, 30, 50]
    for num_trial in num_trial_list:
        result_df = run_specific_algo(ctx, "BO", result_df, num_trial)

    result_df.index = range(1, len(result_df) + 1)
    result_df.index.name = 'idx'
   
    return result_df


def test_all_default():
    for each_file in input_file_list:
        file_sgn = __extract_file_sgn(each_file)
        output_file = f"{exp_output_dir}/default_{file_sgn}_result.csv"
        ctx: Context = init_context(3, each_file)
        result_df = run_all_algos(ctx)
        result_df.to_csv(output_file, sep="\t")


def test_all_algo_by_theta():
    # theta_list = [-2, -1, -0.5, 0, 0.5, 1, 2]
    theta_list = [-1, 0]

    for each_file in input_file_list:
        file_sgn = __extract_file_sgn(each_file)
        for theta in theta_list:
            output_file = f"{exp_output_dir}/theta_{theta}_{file_sgn}_result.csv"
            ctx: Context = init_context(3, each_file, theta=theta)
            result_df = run_all_algos(ctx)
            result_df.to_csv(output_file, sep="\t")
            result_df = result_df.drop(result_df.index)


def __extract_file_sgn(input_str: str):
    match = re.search(r'/([^/]+)\.json', input_str)
    if match:
        extracted_string = match.group(1)
        return extracted_string
    else:
        SystemExit(f"Failed to extract from {input_str}")

if __name__ == '__main__':
    test_all_default()
    test_all_algo_by_theta()
