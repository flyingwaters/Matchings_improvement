import unittest

from process import OurProcessor

class TestBaseProcessor(unittest.TestCase):
    def test_i(self):
        a,b,c = OurProcessor("author1.json")
        assert a==b and b==c, "OurProcessor right"

if __name__ == '__main__':
    unittest.main()