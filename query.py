import numpy as np
from itertools import combinations
from sko.GA import GA
# from regex import subf
from fact import FactSet
from typing import Tuple, List
import abc
from sko.tools import set_run_mode
from numba import jit

def binary_combinations(n):
    # 二进制解空间
    for i in range(2**n):
        binary_string = bin(i)[2:].zfill(n)
        combination = [int(bit) for bit in binary_string]
        yield combination

@jit(nopython=True)
def cac_entropy(prior_p, ans_p_post_o, ans_p):
    cur_h = 0
    o_p_post_ans = prior_p * ans_p_post_o / ans_p
        # 质量增益 论文中的 ΔQ(F|T)
    for i2 in o_p_post_ans:
        cur_h -= (i2 * np.log(i2)).item()  # H(o|AS T CE)
    return cur_h


def expectation_cond(facts: FactSet, selection: List[int], worker_accuracy) -> float:
    """
    可变变量 selection, worker_accuracy
    计算query of correspondences set 的 
    return h(V|T) 
    """
    length = len(selection)
    # space 生成过程
    combinations = list(binary_combinations(length))
    prior_p = facts.get_prior_p()  # 初始化prior_p
    # 排序 后去除 顺序
    selection.sort()

    set_expected_h = 0
    for ans in combinations:
        # 传进去sub_facts[i]，相当于一个 o --- (当CE答案为sub_facts[i]时，获得P(ATCE) 和 P(ATCE|o))
        ans_p, ans_p_post_o = facts.compute_ans_p(ans,
                                                list(selection),
                                                worker_accuracy)
        # h -= p_ans * np.log(p_ans)
        # 获得P(o|ATCE) = P(o) * P(ATCE|o) / P(ATCE)
        assert ans_p!=0, "ans_p = 0 cause exception"
        cur_h = cac_entropy(prior_p, ans_p_post_o, ans_p)
        set_expected_h += ans_p*cur_h
    return set_expected_h


# for test 
def expectation_cond2(facts: FactSet, selection: List[int], worker_accuracy) -> float:
    """
    可变变量 selection, worker_accuracy
    计算query of correspondences set 的 
    return h(V|T) 
    """
    length = len(selection)
    # space 生成过程
    combinations = list(binary_combinations(length))
    prior_p = facts.get_prior_p()  # 初始化prior_p
    # 排序 后去除 顺序
    selection.sort()

    sum_p = 0
    set_expected_h = 0
    for ans in combinations:
        # 传进去sub_facts[i]，相当于一个 o --- (当CE答案为sub_facts[i]时，获得P(ATCE) 和 P(ATCE|o))
        ans_p, ans_p_post_o = facts.compute_ans_p(ans,
                                                list(selection),
                                                worker_accuracy)
        # h -= p_ans * np.log(p_ans)
        # 获得P(o|ATCE) = P(o) * P(ATCE|o) / P(ATCE)
        cur_h = cac_entropy(prior_p, ans_p_post_o, ans_p)
        set_expected_h += ans_p*cur_h
        sum_p+= ans_p 
        print("ans: ", ans, "ans_p: ", ans_p, "cur_h: ", cur_h)
    assert sum_p > 0.9999, "wrong ans_p"
    return set_expected_h


class QuerySelector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select(self, facts: FactSet,
               budget: int,
               worker_accuracy: np.array) -> Tuple[np.ndarray, "FactSet", float]:
        """
        根据该类的策略，固定budget 下 选择 correspondence
        :param facts: 需要被选择作为问题的事实集
        :param budget: 每轮的成本
        :param worker_accuracy: 工人的回答准确率
        :return: 返回facts的一个子集, 包括其相对于原来的索引
        """
        raise NotImplemented("implement me")


class BaseQuerySelector(QuerySelector):
    """
    暴力法的问题选择器
    """
    def select(self, facts: FactSet, budget: int, worker_accuracy: np.ndarray) -> Tuple[np.ndarray, "FactSet", float]:
        """budget each round 成本 cost """
        cost_list = facts.len_list()
        cost_list.sort()
        sum_cost = 0
        max_num = 0
        least_num = 0
        low_num = 0
      
        for i in cost_list[::-1]:
            if least_num + i <= budget:
                least_num += i
                low_num += 1
        # limit of num for brute
        for i in cost_list:
            if sum_cost+i<=budget:
                sum_cost+=i
                max_num+=1
            else:
                break
        assert low_num <= max_num, f"{low_num}, low, {max_num} max"
        num_fact: int = facts.num_fact()
        
        max_selection = []
        max_h = float('-inf')
        
        for num in range(low_num, max_num+1):
            selections = combinations(range(num_fact), num)
            
            for selection in selections:
                selection_cost = 0.
                # 求和
                for ix in selection:
                    selection_cost+=cost_list[ix]
                    
                # 若超过预算 跳过
                if selection_cost > budget:
                    continue
                selection = list(selection)
                set_expected_h = expectation_cond(facts, selection, worker_accuracy)
                # print("selection:", selection,"set_expeted_h:", set_expected_h)
                
                if -set_expected_h > max_h:
                    max_h = -set_expected_h
                    max_selection = selection
        return np.array(max_selection), "No", -max_h


# 贪心法的问题选择器
class GreedyQuerySelector(QuerySelector):  # 改6
    """
    贪心法的问题选择器
    """
    def select(self, facts: FactSet,
               budget: int,
               worker_accuracy: np.ndarray,
               cost_func:int=1
               ) -> Tuple[np.ndarray, "FactSet", float]:
        
        num_fact:int = facts.num_fact()

        max_selection: list = []
        max_hsum = -100.
        # 穷举 计算所有2个 correspondence 最大化 H
        ######################
        c_index_list = list(range(num_fact))
        for two_index in combinations(c_index_list, 2):
            # minimize the h(V|T)
            # maximize the -h(V|T)
            two_cur_h = expectation_cond(facts=facts, selection=list(two_index), worker_accuracy=worker_accuracy)
            if -two_cur_h > max_hsum:
                max_selection = list(two_index)
                max_hsum = -two_cur_h
        # 2个correspondences 的穷举过程，减掉成本
        candidates_list = list(range(num_fact))
        for ix in max_selection:
            budget -= facts.len_list()[ix]
            candidates_list.remove(ix)
        # print("budget:", budget, "two_selection:", max_selection)
        # print("candidates_list:", candidates_list)
        # print("##########################################")
        ########################
        while budget>0:  # 近似找到fact最优组合
            max_h_gain = 0.  # 质量增益最低也得为0
            max_idx = -1
            # 原始
            h = expectation_cond(facts=facts, selection=max_selection, worker_accuracy=worker_accuracy)
            # print("candidate: ", candidates_list)
            for idx in candidates_list:
                ### cost 对比
                if cost_func == 1:
                    w = 1.0
                else:
                    w = facts.len_list()[idx]
                # 检验 cost
                if budget - w <0:
                    continue
                
                max_selection.append(idx)
                cur_h =  expectation_cond(facts=facts, selection=max_selection, worker_accuracy=worker_accuracy)
                h_gain = (h-cur_h)/w # 质量增益 每次的gain(f)\
                
                # 选择最大的gain(f)
                if h_gain >= (max_h_gain-0.000001):
                    max_h_gain = h_gain
                    if idx>num_fact:
                       print(f"idx,{idx},num_fact,{num_fact}") 
                    max_idx = idx   # 每次找出最大的idx在循环外部append
            
                assert max_h_gain >=-0.00000001, f"wrong selection, idx {max_idx}, {h} and {cur_h}"
               
                # 删除 增加的 元素
                max_selection.remove(idx)  # 每次删除的位置一定改用pop()会更快
                
            # 不知道为什么有时候max_idx = -1, 而且这好像是导致熵值不降反增的原因 猜测因为这一轮里面所有的 h_gain都小于0
            # 采取策略是随便塞一个进去
            if max_idx == -1:
                # 无法再加入任何correspondence
                break
            
            
            max_hsum += max_h_gain
            assert max_idx < num_fact, f"exceed num_fact {num_fact},{max_idx}"
                
            budget -= facts.len_list()[max_idx]
            max_selection.append(max_idx)
            candidates_list.remove(max_idx)
            if len(max_selection)==num_fact:
                budget =-1
        
        # 实际选择的set
        return_h = expectation_cond(facts=facts, selection=max_selection, worker_accuracy=worker_accuracy)
        return np.array(max_selection), "No", -return_h


class RandomQuerySelector(QuerySelector):  #2.6
    """
    随机法的问题选择器
    """
    def select(self, facts: FactSet,
               budget: int,
               worker_accuracy: np.ndarray) -> Tuple[np.ndarray, "FactSet", float]:
        """
        每轮固定budget 的选择
        """
        import random
        import numpy as np
        
        num_fact = facts.num_fact()
        cost_list = facts.len_list()
        selection = []
        candidate_list = list(range(0, num_fact))
        while budget>0 and candidate_list!=[]:
            tmp_ix = random.sample(candidate_list,1)
            candidate_list.remove(tmp_ix[0])
            if budget - cost_list[tmp_ix[0]]:
                selection.extend(tmp_ix)
            budget -= cost_list[tmp_ix[0]]
        
            
        # selection = np.random.choice(num_fact,num,replace=False)
        # sub_facts = facts.get_subset(list(selection))
        h = expectation_cond(facts, selection, worker_accuracy)
        return np.array(selection), 0, -h
    
    
class HeuristicQuerySelector(QuerySelector):
    """
    启发式算法问题选择器
    """
    def select(self, facts:FactSet, budget: int, worker_accuracy: np.ndarray, max_iters:int, cost_func:int) -> Tuple[np.ndarray, "FactSet", float]:
        num = facts.num_fact()
        if cost_func==1:
            cost = np.array([1 for _ in range(num)])
        else:
            cost = np.array(facts.len_list())
        
        def func(x):
            k = np.array(x)
            selection = list(np.where(k==1)[0])
            return expectation_cond(facts, selection, worker_accuracy)

        constraint_ueq = [lambda x: np.sum(cost*x)-budget]
        lb = [0 for i in range(num)]
        ub = [1 for i in range(num)]
        precision = [1 for _ in range(num)]
        set_run_mode(func, 'cached')
        ga = GA(func=func, n_dim=num, constraint_ueq= constraint_ueq ,size_pop=50, max_iter=max_iters, prob_mut=0.001, lb=lb, ub=ub, precision=precision)
        best_x, best_y = ga.run()
        k = np.array(best_x)
        selection = np.where(k==1)[0]
        return  list(selection), 0, -best_y