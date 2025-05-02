import unittest
import numpy as np
from packages.generateDataSets import SyntheticMatcherDataset
from packages.calculateStatistics import DatasetEvaluator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDatasetEvaluator(unittest.TestCase):


    def test_evalut_with_1000_size(self):
        
        np.random.seed(42)  # Ensure reproducibility
        dataset = SyntheticMatcherDataset(size=1000, true_positive_ratio=0.7)
        df1, df2 = dataset.df1, dataset.df2
        expected = dataset.expected
        
        evaluator = DatasetEvaluator(self.df1, self.df2, self.expected)
        evaluator.evaluate()
        
        self.assertEqual(evaluator.tp, expected['tp'], "True positives do not match expected value")
        self.assertEqual(evaluator.fp, expected['fp'], "False positives do not match expected value")
        self.assertEqual(evaluator.fn, expected['fn'], "False negatives do not match expected value")


if __name__ == '__main__':
    unittest.main()