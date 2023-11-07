from query import QuerySelector, BaseQuerySelector, GreedyQuerySelector, RandomQuerySelector, HeuristicQuerySelector
import numpy as np
from fact import FactSet
from llm_api import gpt_check
import numpy as np
import json
import time
import tiktoken
import copy

def answers(answers_path):
    """ 返回chatgpt web 中的答案 list
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
    print("start approx~")
    approx_h_list = [ex_fact.compute_entropy()]
    time_list = []
    cost_list = []
    cost_sum = 0 
    turns = total_turns
    while turns > 0:
        start = time.time()
        selection_idxes, sub_facts, h = query_selector.select(ex_fact, budget, acc, cost_func=2)
        if api_use:
            ans = [1 if gpt_check(ix_r, c_set) == "yes" else 0 for ix_r in selection_idxes]
        else:
            ans = [1 if ans_list[ix_r] == "yes" else 0 for ix_r in selection_idxes]
        p_a, p_a_v = ex_fact.compute_ans_p(ans, selection_idxes, acc)
        p_post = ex_fact.get_prior_p()*p_a_v/p_a
        ex_fact.set_prior_p(p_post)
        approx_h_list.append(ex_fact.compute_entropy())
        turns -= 1
        end = time.time()
        time_list.append(end - start)
        # cost this turn to caculate
        tmp_cost = 0
        for ix in selection_idxes:
            tmp_cost += ex_fact.len_list()[ix]
        cost_list.append(int(tmp_cost))
        
    return approx_h_list, time_list, cost_list, list(ex_fact.get_prior_p())


def brute(ex_fact, query_selector, budget, acc, total_turns, api_use, c_set, ans_list):
    print("start brute~")
    brute_h_list = [ex_fact.compute_entropy()]
    time_list = []
    cost_list = []
    cost_sum = 0 
    turns = total_turns
    while turns > 0:
        start = time.time()
        selection_idxes, sub_facts, h = query_selector.select(ex_fact, budget, acc)
        if api_use:
            ans = [1 if gpt_check(ix_r, c_set) == "yes" else 0 for ix_r in selection_idxes]
        else:
            ans = [1 if ans_list[ix_r] == "yes" else 0 for ix_r in selection_idxes]
        p_a, p_a_v = ex_fact.compute_ans_p(ans, selection_idxes, acc)
        p_post = ex_fact.get_prior_p()*p_a_v/p_a
        ex_fact.set_prior_p(p_post)
        brute_h_list.append(ex_fact.compute_entropy())
        turns -= 1
        end = time.time()
        time_list.append(end - start)
        # cost this turn to caculate
        tmp_cost = 0
        for ix in selection_idxes:
            tmp_cost += ex_fact.len_list()[ix]
            
        cost_list.append(int(tmp_cost))
    return brute_h_list, time_list, cost_list, list(ex_fact.get_prior_p())

def heuristic(ex_fact, query_selector, budget, acc, total_turns, api_use, c_set, ans_list):
    print("start heuristic~")
    heuristic_h_list = [ex_fact.compute_entropy()]
    time_list = []
    cost_list = []
    cost_sum = 0 
    turns = total_turns
    while turns > 0:
        start = time.time()
        selection_idxes, sub_facts, h = query_selector.select(ex_fact, budget, acc, max_iters=10, cost_func=2)
        if api_use:
            ans = [1 if gpt_check(ix_r, c_set) == "yes" else 0 for ix_r in selection_idxes]
        else:
            ans = [1 if ans_list[ix_r] == "yes" else 0 for ix_r in selection_idxes]
        p_a, p_a_v = ex_fact.compute_ans_p(ans, selection_idxes, acc)
        p_post = ex_fact.get_prior_p()*p_a_v/p_a
        ex_fact.set_prior_p(p_post)
        heuristic_h_list.append(ex_fact.compute_entropy())
        turns -= 1
        end = time.time()
        time_list.append(end - start)
        # cost this turn to caculate
        tmp_cost = 0
        for ix in selection_idxes:
            tmp_cost += ex_fact.len_list()[ix]
            
        cost_list.append(int(tmp_cost))
    return heuristic_h_list, time_list, cost_list, list(ex_fact.get_prior_p())

def random_algorithm(random_fact, query_selector, budget, acc, total_turns, api_use, c_set, ans_list):
    print("start random~")
    all_h_list = []
    prior_p_l = []
    for _ in range(100):
        random_i_fact = copy.deepcopy(random_fact)
        cost_sum_r = 0
        turns_r=total_turns
        random_h_list=[random_i_fact.compute_entropy()]
        random_selection_idxes, _, _  =  random_selector.select(random_i_fact, budget, acc)
        while turns_r>0:
            if api_use:
                ans_r = [1 if gpt_check(ix_r, c_set)=="yes" else 0 for ix_r in random_selection_idxes]
            else:
                ans_r = [1 if ans_list[ix_r]=="yes" else 0 for ix_r in random_selection_idxes]
            p_a_r,p_a_v_r = random_i_fact.compute_ans_p(ans_r, random_selection_idxes, acc)
            p_post = random_i_fact.get_prior_p()*p_a_v_r / p_a_r
            random_i_fact.set_prior_p(p_post)
            random_h_list.append(random_i_fact.compute_entropy())
            turns_r -=1
        prior_p_l.append(random_i_fact)
        all_h_list.append(random_h_list)
    
    n = np.array(all_h_list)
    random_h_l = n.mean(axis=0, keepdims=True)
    random_h_l = random_h_l.tolist()[0]
    
    m = np.array(prior_p_l)
    post_p = m.mean(axis=0, keepdims=True)
    post_p = post_p.tolist()[0]
    return random_h_l, post_p


def save_h_file(approx_name_f, random_f, brute_name_f, heuristic_name_f, approx_ob, random_ob, brute_ob, heuristic_ob,
                dataset = "output"):
    """ 
    保存文件，
    
    """
    with open(f"./{dataset}/"+ approx_name_f, "w") as f:
        json.dump(approx_ob, f, indent=2, ensure_ascii=False)
    
    with open(f"./{dataset}/"+ random_f, "w") as w:
        json.dump(random_ob, w, indent=2, ensure_ascii=False)
        
    with open(f"./{dataset}/"+ brute_name_f, "w") as f2:
        json.dump(brute_ob, f2, indent=2, ensure_ascii=False)
        
    with open(f"./{dataset}/"+ heuristic_name_f, "w") as f3:
        json.dump(heuristic_ob, f3, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    budgets = [15, 20, 25, 30]
    
    total_turns = 10
    names = ["author", "employee", "purchase"]
    p_w = [0.89, 0.92, 0.87] 
    paths = [f"./dataset/{i}.json" for i in ["author", "employee", "purchase"]]
    ans_paths = [f"./dataset/{i}_ans.json" for i in ["author", "employee", "purchase"]]
    
    for dataset_id, path_i in enumerate(paths):
        for budget in budgets:
            fact_selectors, c_num, c_set = process(path_i)
        
            acc = np.array([[p_w[dataset_id] for i in range(c_num)]])
            api_use = False
            ans_list = answers(ans_paths[dataset_id])
        
            approx_fact, approx_selector = fact_selectors[0]
            approx_h, approx_time, approx_cost, approx_post_p = approx(approx_fact, approx_selector, budget, acc, total_turns, api_use, c_set, ans_list)
            
            random_fact, random_selector = fact_selectors[1]
            random_h, random_post_p = random_algorithm(random_fact, random_selector, budget, acc, total_turns, api_use, c_set, ans_list)
            
            brute_fact, brute_selector = fact_selectors[2]
            brute_h, brute_time, brute_cost, brute_post_p = brute(brute_fact, brute_selector, budget, acc, total_turns, api_use, c_set, ans_list)
            
            h_fact, h_selector = fact_selectors[3]
            heuristic_h, heuristic_time, heuristic_cost, heuristic_post_p = heuristic(h_fact, h_selector, budget, acc, total_turns, api_use, c_set, ans_list)
            
            approx_result = {"entropy":approx_h, "timecost":approx_time,"cost": approx_cost, "prob":approx_post_p}
            print(type(approx_h)), print(type(approx_time)), print(type(approx_cost)), print(type(approx_post_p))
            random_result = {"entropy":random_h, "prob":random_post_p}
            brute_result = {"entropy":brute_h, "timecost":brute_time,"cost":brute_cost ,"prob":brute_post_p}
            heuristic_result = {"entropy":heuristic_h, "timecost":heuristic_time, "cost":heuristic_cost,"prob":heuristic_post_p}
            
            
            save_h_file(f"{names[dataset_id]}_approx_{budget}.json", f"{names[dataset_id]}_random_{budget}.json", f"{names[dataset_id]}_brute_{budget}.json", f"{names[dataset_id]}_heuristic_{budget}.json", approx_result, random_result, brute_result, heuristic_result)
            