import unittest
from sklearn.linear_model import LogisticRegression

from hactap.ai_worker import AIWorker

class TestAIWorker(unittest.TestCase):

    def test_ai_worker_create(self):
        aiw = AIWorker(
            LogisticRegression()
        )

        self.assertIsInstance(aiw, AIWorker)

if __name__ == '__main__':
    unittest.main()
