import unittest
from fact import FactSet
import numpy as np

class TestFactSet(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.fact_case1 = FactSet(facts=np.array([[0, 0, 0, 0], [0, 0, 0, 1],
                                                  [0, 0, 1, 0], [0, 0, 1, 1],
                                                  [0, 1, 0, 0], [0, 1, 0, 1],
                                                  [0, 1, 1, 0], [0, 1, 1, 1],
                                                  [1, 0, 0, 0], [1, 0, 0, 1],
                                                  [1, 0, 1, 0], [1, 0, 1, 1],
                                                  [1, 1, 0, 0], [1, 1, 0, 1],
                                                  [1, 1, 1, 0], [1, 1, 1, 1]]),
                                  prior_p=np.array([
                                      0.03, 0.06, 0.07, 0.04, 0.09, 0.01, 0.11,
                                      0.09, 0.04, 0.04, 0.04, 0.05, 0.06, 0.09,
                                      0.07, 0.11
                                  ]),
                                  ground_true=2)

    def test_compute_ans_p(self):
        facts = self.fact_case1
        accuracy = np.array([[0.8, 0.7, 0.9, 0.85]])  # 对应fact1, 3是0.8, 0.9
        ans = np.array([1, 0])
        p, p_o = facts.compute_ans_p(ans, [0, 2], accuracy)
        p2, p2_o = facts.compute_ans_p(ans,[2, 0], accuracy)
        print(p, p2)
        

    def test_compute_entropy(self):
        facts = self.fact_case1
        subfact1 = facts.get_subset([0, 1,3,2])  # entropy 1.993
        subfact2 = facts.get_subset([ 0,1,2,3])
        assert subfact2.get_prior_p().any()==subfact1.get_prior_p().any()
        

       

if __name__ == "__main__":
    unittest.main()