import numpy as np
from itertools import combinations

# from regex import subf
from fact import FactSet
from typing import Tuple
import abc
import random


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

    def select(self, facts: FactSet, num: int, worker_accuracy: np.ndarray) -> Tuple[np.ndarray, "FactSet", float]:
        num_fact: int = facts.num_fact()
        if num > num_fact:  # k等于5，但fact只有4
            num = num_fact
        assert num <= num_fact
        # print('num: ', num)
        selections = combinations(range(num_fact), num)
        # print('worker_accuracy: ', worker_accuracy)
        # max_h: float = -float('inf')
        max_h = 0.
        max_selection: Tuple[int] = (0,) * num_fact
        for selection in selections:
            sub_facts = facts.get_subset(list(selection))
            # h = 0.
            prior_p = sub_facts.get_prior_p()  # 初始化prior_p
            h = 0.
            for i in prior_p:
                if i == 0:  # 有些为0的 不好算log
                    continue
                h -= (i * np.log(i)).item()  # H(O)
            # 暴力枚举并找到fact最优组合
            h_gain = 0.
            for i in range(len(sub_facts)):
                cur_h = 0.
                # 传进去sub_facts[i]，相当于一个 o --- (当CE答案为sub_facts[i]时，获得P(ATCE) 和 P(ATCE|o))
                ans_p, ans_p_post_o = sub_facts.compute_ans_p(sub_facts[i],
                                                list(range(sub_facts.num_fact())),
                                                worker_accuracy[:,list(selection)])

                # h -= p_ans * np.log(p_ans)
                # 获得P(o|ATCE) = P(o) * P(ATCE|o) / P(ATCE)
                o_p_post_ans = prior_p * ans_p_post_o / ans_p

                # 质量增益 论文中的 ΔQ(F|T)
                for i in o_p_post_ans:
                    if i == 0:  # 有些为0的 不好算log
                        continue
                    cur_h -= (i * np.log(i)).item()  # H(o|AS T CE)
                h_gain = (h - cur_h)
            if h_gain > max_h:
                max_h = h_gain
                max_selection = selection
        if max_selection == (0,) * num_fact:
            max_selection = random.sample(range(0, num_fact), num)

        return np.array(max_selection), facts.get_subset(list(max_selection)), max_h


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
        c_max_cost: int = 3
        assert budget >= 3*c_max_cost, "不大于3个最大correspondence 的cost"
        
        max_selection: list = []
        max_hsum = 0.
        # 穷举 计算所有2个 correspondence 最大化 H
        c_index_list = list(range(num_fact))
        
        for two_index in combinations(c_index_list, 2):
            two_sub_facts = facts.get_subset(list(two_index))
            prior_p = two_sub_facts.get_prior_p()
            two_cur_h = 0
            for two_i in range(len(two_sub_facts)):
                    two_cur_h_a_i = 0.  # 获得H(o|AS T∪{idx} CE) 即 cur_h
                    # 传进去sub_facts[i]，相当于一个 o --- (当CE答案为sub_facts[i]时，获得P(ATCE) 和 P(ATCE|o))
                    ans_p, ans_p_post_o = two_sub_facts.compute_ans_p(two_sub_facts[two_i],
                                                       list(range(two_sub_facts.num_fact())),
                                                       worker_accuracy[:, list(two_index)])
                    # h -= np.sum(ans_p * np.log(ans_p)).item()
                    # 获得P(o|ATCE) = P(o) * P(ATCE|o) / P(ATCE)
                    o_p_post_ans = prior_p * ans_p_post_o / ans_p
                    # HC bugs 
                    for  i in o_p_post_ans:
                        if i ==0: # 有些为0的 不好算log
                             i=0.00000001
                        two_cur_h_a_i -= (i * np.log(i)).item() # H(o|AS T∪{idx} CE)
                    
                    # 期望的expectation
                    two_cur_h+=ans_p*two_cur_h_a_i
                    # cur_h -= np.sum(-o_p_post_ans * np.log(o_p_post_ans)).item()
                    if two_cur_h > max_hsum:
                        max_selection = list(two_index)
                        max_hsum = two_cur_h
        # 减掉成本
       
        for ix in max_selection:
            budget -= facts.len_list()[ix]
        
        # >3 的 greedy algorithmn
        now_num = 0 
        while budget>0:  # 近似找到fact最优组合
            max_h_gain = 0.  # 质量增益最低也得为0
            # max_h_gain: float = -float('inf')
            max_idx = -1
            for idx in range(num_fact):
                ### cost 对比
                if cost_func == 1:
                    w = 1.0
                else:
                    w = facts.len_list()[idx]
                    
                if idx in max_selection:
                    continue
                max_selection.append(idx)
                sub_facts = facts.get_subset(list(max_selection))
                
                prior_p = sub_facts.get_prior_p()  # 初始化prior_p               
                h_gain = 0.
                if now_num == 0:  # h最初等于H(O)
                    h = 0
                    # h = np.sum(-prior_p * np.log(prior_p)).item()
                else:  # 后面找的时候 h 需要更新为上次的 H(o|AS T∪{idx} CE)
                    h = cur_h
                
                cur_h = 0
                for i_sub in range(len(sub_facts)):
                    cur_h_a_i = 0.  # 获得H(o|AS T∪{idx} CE) 即 cur_h
                    # 传进去sub_facts[i]，相当于一个 o --- (当CE答案为sub_facts[i]时，获得P(ATCE) 和 P(ATCE|o))
                    ans_p, ans_p_post_o = sub_facts.compute_ans_p(sub_facts[i_sub],
                                                       list(range(sub_facts.num_fact())),
                                                       worker_accuracy[:, list(max_selection)])
                    # h -= np.sum(ans_p * np.log(ans_p)).item()
                    # 获得P(o|ATCE) = P(o) * P(ATCE|o) / P(ATCE)
                    o_p_post_ans = prior_p * ans_p_post_o / ans_p

                    # HC bugs 
                    for  i in o_p_post_ans:
                        if i ==0: # 有些为0的 不好算log
                             i=0.00000001
                        cur_h_a_i -= (i * np.log(i)).item() # H(o|AS T∪{idx} CE)
                    
                    # 期望的expectation
                    cur_h+=ans_p*cur_h_a_i
                    # cur_h -= np.sum(-o_p_post_ans * np.log(o_p_post_ans)).item()
                h_gain = (cur_h-h)/w # 质量增益 每次的gain(f)

                # 选择最大的gain(f)
                if h_gain > max_h_gain:
                    max_h_gain = h_gain
                    if idx>num_fact:
                       print(f"idx,{idx},num_fact,{num_fact}") 
                    max_idx = idx   # 每次找出最大的idx在循环外部append
                    # max_selection = copy.copy(max_selection)
                max_selection.pop(now_num)  # 每次删除的位置一定改用pop()会更快
            # 不知道为什么有时候max_idx = -1, 而且这好像是导致熵值不降反增的原因 猜测因为这一轮里面所有的 h_gain都小于0
            # 采取策略是随便塞一个进去
            if max_idx == -1:
                # print('max_h:', max_h)
                for i in range(num_fact):
                    if i not in max_selection:
                        max_selection.append(i)
                        budget -= facts.len_list[max_idx]
                        break
            else:
                max_hsum += max_h_gain
                assert max_idx < num_fact, f"exceed num_fact {num_fact},{max_idx}"
                max_selection.append(max_idx)
                budget -= facts.len_list()[max_idx]
        # print('len(max_selection): ', len(max_selection))
        # print('max_selection: ', max_selection)
        return np.array(max_selection), facts.get_subset(list(max_selection)), max_hsum


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
        while budget>0:
            tmp_ix = random.sample(range(0, num_fact),1)
            num_fact.pop(tmp_ix)
            if budget - cost_list[tmp_ix[0]]:
                selection.extend(tmp_ix)
            budget -= cost_list[tmp_ix[0]]
        
            
        # selection = np.random.choice(num_fact,num,replace=False)
        # sub_facts = facts.get_subset(list(selection))
        
        sub_facts = facts.get_subset(selection)
        h = 0
        for i in range(len(sub_facts)):
            p_ans, _ = sub_facts.compute_ans_p(sub_facts[i],
                                               list(range(sub_facts.num_fact())),
                                               worker_accuracy[:,list(selection)])
            h -= p_ans * np.log2(p_ans)
        # print("base: ", selection, "max_h: ", h)
        return np.array(selection), facts.get_subset(list(selection)), h