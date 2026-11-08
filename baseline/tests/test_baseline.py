import importlib.util
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

spec = importlib.util.spec_from_file_location('baseline_train', Path(__file__).parents[1] / 'train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BaselineTests(unittest.TestCase):
    def test_validation_never_changes_vocabulary_and_saved_model_roundtrip(self):
        model = module.build_model(min_df=1)
        model.fit(['comfortable shoes', 'excellent comfortable fit',
                   'poor shoes', "don't buy terrible fit"], [1, 1, 0, 0])
        vocab = dict(model.named_steps['tfidf'].vocabulary_)
        reviews = ['validationonlyword comfortable', "don't buy"]
        pred = model.predict_proba(reviews)
        self.assertNotIn('validationonlyword', vocab)
        self.assertEqual(vocab, model.named_steps['tfidf'].vocabulary_)
        self.assertIn("don't", vocab)
        np.testing.assert_allclose(pred.sum(axis=1), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(reviews), pred)

    def test_majority_baseline_does_not_hide_negative_failure(self):
        result = module.evaluate([0, 0, 1, 1, 1], [1, 1, 1, 1, 1])
        self.assertEqual(result['confusion_matrix'], [[0, 2], [0, 3]])
        self.assertEqual(result['classification_report']['negative']['recall'], 0)
        self.assertAlmostEqual(result['accuracy'], .6)
        self.assertAlmostEqual(result['macro_f1'], .375)


if __name__ == '__main__':
    unittest.main()
