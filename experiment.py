from query import QuerySelector, BaseQuerySelector, GreedyQuerySelector,RandomQuerySelector, HeuristicQuerySelector
import numpy as np
from fact import FactSet
from llm_api import gpt_check
import numpy as np
import json
import time
import tiktoken

def answers(answers_path):
    """ 返回chatgpt web 中的答案list
    example: ["yes",...,"no",..,"yes"]
    """
    with open(answers_path, "r") as f:
        ans_list = json.load(f)
    return ans_list

def process(matchings_path):
    """
    matchings_path: possible matchings 地址 (json) 参照例子的json形式
    return ([数据类 每个算法的  FactSet 和 selector] , correspondence num, c_set)
    """
    with open(matchings_path, "r") as f:
        content = json.load(f)
   
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    c_set = content["correspondence_set"]
    # chatgpt 的tokens 的num list
    len_list = [len(encoding.encode(i[0][1]+i[1][1])) for i in c_set]
    len_list.sort()
    least_len = sum(len_list[:3])
    matchings = content["matchings"]
    Views = []
    for match in matchings:
        view = []
        for c in c_set:
            if c in match:
                view.append(1)
            else:
                view.append(0)
        Views.append(view)
    
    print("max_cost: ", least_len, f"note that budget of each round must > {least_len} and < {sum(len_list)}:")
    ex_fact = FactSet(facts=np.array(Views), prior_p=np.array(content["prob_all"]), ground_true=2, len_list=np.array(len_list))
    random_fact = FactSet(facts=np.array(Views), prior_p=np.array(content["prob_all"]), ground_true=2, len_list=np.array(len_list))
    brute_fact = FactSet(facts=np.array(Views), prior_p=np.array(content["prob_all"]), ground_true=2, len_list=np.array(len_list))
    heuristic_fact = FactSet(facts=np.array(Views), prior_p=np.array(content["prob_all"]), ground_true=2, len_list=np.array(len_list))
    
    # 对应fact1, 3是0.8, 0.
    query_selector = GreedyQuerySelector()
    # selection_idxes, sub_facts, h = query_selector.select(ex_fact, 2, accuracy, cost_func=2)
    random_selector = RandomQuerySelector()
    base_selector = BaseQuerySelector()
    h_selector = HeuristicQuerySelector()
    return [(ex_fact, query_selector), (random_fact, random_selector), (brute_fact, base_selector), (heuristic_fact, h_selector)], ex_fact.num_fact(), c_set


def approx(ex_fact, query_selector, budget, acc, total_turns, api_use, c_set, ans_list):
    approx_h_list=[ex_fact.compute_entropy()]
    time_list = []
    
    cost_sum = 0 
    turns = total_turns
    
    
    
    while turns>0:
        start = time.time()
        selection_idxes, sub_facts, h = query_selector.select(ex_fact, budget, acc, cost_func=2)
        if api_use:
            ans = [1 if gpt_check(ix_r, c_set)=="yes" else 0 for ix_r in selection_idxes]
        else:
            ans = [1 if ans_list[ix_r]=="yes" else 0 for ix_r in selection_idxes]
        p_a,p_a_v = ex_fact.compute_ans_p(ans, selection_idxes, acc)
        p_post = ex_fact.get_prior_p()*p_a_v/p_a
        ex_fact.set_prior_p(p_post)
        approx_h_list.append(ex_fact.compute_entropy())
        turns -=1
        end = time.time()
        time_list.append(end - start)
         
    approx_timecost = end - start
print("approx:{} s".format(approx_timecost))
