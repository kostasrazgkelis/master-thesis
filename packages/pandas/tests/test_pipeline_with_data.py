import unittest
import pandas as pd
from packages.utils.generate_data_set import SyntheticMatcherDataset as smd
from packages.pandas.pandas_pipeline import DatasetEvaluator as de
import jellyfish


class TestDatasetEvaluator(unittest.TestCase):

    def test_normal_data(self):
        expected = {"gt": 5, "tp": 4, "fp": 2, "fn": 1}
        
        # Data for df1
        data1 = [
            ["ID00005", "N039", "E298", "Q412", "V409", "R232"],  # TP1
            ["ID00009", "R822", "W179", "H017", "P323", "F298"],  # TP2
            ["ID00007", "R449", "X716", "M948", "G667", "S702"],  # TP3
            ["ID00004", "N002", "E396", "N843", "I458", "S719"],  # TP4
            ["ID10004", "N002", "E396", "N853", "I623", "S569"],  # FN1
            ["NEW72378", "J547", "B222", "G492", "R551", "S490"],  # FP1
            ["ID00008", "N322", "K685", "T442", "C825", "W967"],  # FP2
            ["ID00000", "W815", "L281", "R155", "F768", "B914"],
            ["ID00001", "C172", "B326", "X400", "M508", "O776"],
            ["ID00002", "V683", "C265", "J127", "D589", "F482"],
            ["ID00003", "E851", "P721", "F745", "D863", "K229"],
            ["ID00016", "T873", "D670", "U046", "Z181", "X621"],
            ["ID00017", "F327", "G856", "E567", "O929", "Q721"],
            ["ID00010", "O283", "T723", "Z034", "V319", "X338"],
        ]

        # Data for df2
        data2 = [
            ["ID00005", "R746", "E298", "Q412", "L291", "R232"],  # TP1
            ["ID00009", "R822", "W179", "H017", "P323", "F298"],  # TP2
            ["ID00007", "Z011", "X716", "M948", "W967", "S702"],  # TP3
            ["ID00004", "N002", "E396", "N843", "V935", "S719"],  # TP4
            ["ID10004", "N002", "E396", "N553", "I453", "S459"],  # FN1
            ["NEW80187", "J547", "B222", "G492", "W673", "S490"],  # FP1
            ["NEW30110", "N322", "K685", "T432", "C225", "W967"],  # FP2
            ["NEW72832", "F875", "Q768", "H822", "Z154", "X678"],
            ["NEW30110", "R560", "C434", "M687", "Q689", "Q863"],
            ["NEW81243", "R762", "N687", "A109", "K476", "R637"],
            ["NEW52689", "A089", "V733", "W158", "A640", "H331"],
            ["NEW67368", "Z079", "J617", "G878", "W111", "Q500"],
            ["NEW72348", "J547", "B222", "G492", "R551", "S490"],
            ["NEW34469", "Y990", "H898", "W673", "L967", "M829"],
        ]

        # Create DataFrames
        columns = [0, 1, 2, 3, 4, 5]
        df1 = pd.DataFrame(data1, columns=columns)
        df2 = pd.DataFrame(data2, columns=columns)

        pipeline = de(
            df1=df1, df2=df2, expected=expected, threshold=3, match_column=0, trim=0
        )

        pipeline.preprocess()
        pipeline.evaluate()
        pipeline.calculate_statistics()

        self.assertEqual(
            pipeline.tp, expected["tp"], "True positives do not match expected value"
        )
        self.assertEqual(
            pipeline.fp, expected["fp"], "False positives do not match expected value"
        )
        self.assertEqual(
            pipeline.fn, expected["fn"], "False negatives do not match expected value"
        )

    def test_with_synthetic_data(self):
        expected = {"gt": 125, "tp": 93, "fp": 69, "fn": 32}
        
# The code snippet you provided is generating synthetic data for testing purposes. Here's a breakdown
# of what it does:
        # Generate synthetic data
        generator = smd(
            size=500,
            ground_truth_ratio=0.25,
            datasets_ratio=(1, 2),
            true_positive_ratio=0.75,
            threshold=3,
        )
        df1, df2 = generator.df1, generator.df2

        # Create expected values based on the generator's output

        pipeline = de(
            df1=df1, df2=df2, expected=expected, threshold=3, match_column="id", trim=0
        )

        pipeline.preprocess()
        pipeline.evaluate()
        pipeline.calculate_statistics()

        self.assertEqual(
            pipeline.tp, expected["tp"], "True positives do not match expected value"
        )
        self.assertEqual(
            pipeline.fp, expected["fp"], "False positives do not match expected value"
        )
        self.assertEqual(
            pipeline.fn, expected["fn"], "False negatives do not match expected value"
        )

    def test_real_data(self):

        expected = {"gt": 5, "tp": 4, "fp": 1, "fn": 1}

        df1 = pd.read_csv("data/df1.csv", header=None, dtype=str)[[0, 1, 2, 3, 4, 5]]
        df2 = pd.read_csv("data/df2.csv", header=None, dtype=str)[[0, 1, 2, 3, 4, 5]]

        for df in [df1, df2]:
            for col_name in df.columns:
                if col_name != 0:
                    df[col_name] = df[col_name].apply(lambda x: jellyfish.soundex(str(x)))

        pipeline = de(df1.sample(frac=0.01, random_state=42), df2.sample(frac=0.01, random_state=55), threshold=3)
        pipeline.preprocess()
        pipeline.evaluate()
        pipeline.calculate_statistics()
        pipeline.printResults()

        self.assertEqual(
            pipeline.tp, expected["tp"], "True positives do not match expected value"
        )
        self.assertEqual(
            pipeline.fp, expected["fp"], "False positives do not match expected value"
        )
        self.assertEqual(
            pipeline.fn, expected["fn"], "False negatives do not match expected value"
        )

if __name__ == "__main__":
    unittest.main()
