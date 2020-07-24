import unittest

from hactap import utils

class TestUtils(unittest.TestCase):

    def test_get_experiment_id(self):
        settings = {
            "aaa": 111
        }
        self.assertIsInstance(utils.get_experiment_id(settings), str)

    def test_get_timestamp(self):
        self.assertIsInstance(utils.get_timestamp(), str)

if __name__ == '__main__':
    unittest.main()
