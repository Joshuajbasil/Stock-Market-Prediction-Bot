import unittest
import numpy as np
from stockbot.data import synthetic
from stockbot.features import build_features, dataset, validate
from stockbot.backtest import walk_forward, metrics


class PipelineTests(unittest.TestCase):
    def test_features_do_not_see_future(self):
        raw = synthetic('TEST', 400)
        original = build_features(raw)
        changed = raw.copy()
        changed.iloc[300:, :4] *= 3
        np.testing.assert_allclose(original.iloc[:300], build_features(changed).iloc[:300], equal_nan=True)
        self.assertGreaterEqual(len(original.columns), 20)

    def test_targets_and_purged_backtest(self):
        raw = synthetic('TEST', 400)
        data = dataset(raw)
        self.assertNotIn(raw.index[-1], data.index)
        day = data.index[0]
        loc = raw.index.get_loc(day)
        self.assertAlmostEqual(data.loc[day, 'target_return'], raw.Close.iloc[loc+1]/raw.Close.iloc[loc]-1)
        p = walk_forward(data, list(build_features(raw)), 'ridge', 120, 60, 1, 42, 5)
        self.assertTrue((p.train_end < p.index).all())
        self.assertFalse(p.index.duplicated().any())
        np.testing.assert_allclose(p.strategy_return, p.position*(p.trade_return-0.001))
        self.assertEqual(metrics(p)['observations'], len(p))

    def test_bad_data_rejected(self):
        raw = synthetic('TEST', 100)
        raw.iloc[0, 0] = -1
        with self.assertRaises(ValueError):
            validate(raw)

    def test_test_labels_cannot_change_predictions(self):
        raw = synthetic('TEST', 260)
        data = dataset(raw)
        features = list(build_features(raw))
        original = walk_forward(data, features, 'ridge', 120, 500, 1, 42, 5)
        changed = data.copy()
        changed.loc[original.index, 'target_return'] = 0.99
        other = walk_forward(changed, features, 'ridge', 120, 500, 1, 42, 5)
        np.testing.assert_allclose(original.predicted_return, other.predicted_return)


if __name__ == '__main__':
    unittest.main()
