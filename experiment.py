from fact import FactSet
import numpy as np
import json
import tiktoken
from query import QuerySelector, BaseQuerySelector, GreedyQuerySelector,RandomQuerySelector


from typing import Tuple, List
import abc

class Base_Experiment(abc.ABCMeta):
    @abc.abstractclassmethod
    def __init__(self, selector, k, turns):
        self.k = k
        self.turns = turns
        self.selector = selector
    
    @abc.abstractclassmethod
    def llm_service(self):
        raise Exception("llm_service not imple")
    
    @abc.abstractclassmethod
    def run(self):
        raise Exception("run func is not imple")

class Approx(Base_Experiment):
    """our paper core selection approximate algorithm

    Args:
        Base_Experiment (_type_): _description_
    """
    def __init__(self, selector:QuerySelector, k:int, turns:int, view:FactSet, p_w:float, ans_path:str, cost_op:int):
        """ 初始化

        Args:
            selector (QuerySelector): k-selection 的近似算法
            k (int): 参数k
            turns (int): 轮数
            view (FactSet): 不确定性降低的主体:views 
            p_w (float): llm 的准确性
            ans_path (str): 模拟cahtgpt 的回答列表
            cost_op (int): 成本选项
        """
        self.selector = selector
        self.turns = turns
        self.k = k 
        self.view=view
        self.p_w = p_w 
        self.acc = np.array([p_w for _ in range(self.view.num_fact())])
        self.ans_path = ans_path
        self.cost_op = cost_op
        
    def llm_service(self)->List[str]:
        """模拟llm answer

        Args:
            path (str): answer list 地址

        Returns:
            List[str]: correspondence 的回答列表
        """
        if self.ans_path:
            with open(self.ans_path, "r") as f:
                ans_list = json.load(f)
            assert len(self.view.len_list)==len(ans_list),"答案和corespondence 不对应"
        else:
            raise Exception("no answer list, you need establish other LLM service")
        
        return ans_list
    
    def run(self):
        approx_h_list=[self.view.compute_entropy()]
        cost_sum = 0
        ans_list = self.llm_service(self.ans_path)
        while self.turns>0:
            
            selection_idxes, sub_facts, h = self.selector.select(self.view, self.k, self.acc, cost_func=1)
            for ix in selection_idxes:
                cost_sum += self.view.len_list()[ix]
                ans=[1 if ans_list[ix]=="yes" else 0]
                p_a, p_a_v = self.view.compute_ans_p(ans, [ix], self.acc)
            
            # update
            sum_p = []
            for idx,i in enumerate(self.view.get_prior_p()):
                p_post = i*p_a_v[idx]/p_a
                sum_p.append(p_post)
            self.view.set_prior_p(np.array(sum_p))
        approx_h_list.append(self.view.compute_entropy())
        self.turns -=1
        return approx_h_list, cost_sum 
        
        
class Random(Base_Experiment):
    def __init__(self, selector:QuerySelector, k:int, turns:int, view:FactSet, p_w:float, ans_path:str, cost_op:int):
        self.selector = selector
        self.turns = turns
        self.k = k 
        self.view=view
        self.p_w = p_w 
        self.acc = np.array([p_w for _ in range(self.view.num_fact())])
        self.ans_path = ans_path
        self.cost_op = cost_op
        
    def run(self, turns_random):
        