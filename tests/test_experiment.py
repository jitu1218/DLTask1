"""Checks for leakage, dropout behavior, reproducibility and checkpoint restoration."""
import unittest
import numpy as np
import torch
from sklearn.datasets import load_digits
from experiment import load_data, network, fit, evaluate


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data, cls.ids, cls.scaler = load_data()

    def test_disjoint_complete_stratified_split(self):
        groups = [set(x) for x in self.ids.values()]
        self.assertEqual(len(set.union(*groups)), 1797)
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        for _, y in self.data.values():
            self.assertEqual(len(torch.unique(y)), 10)

    def test_scaler_uses_training_only(self):
        x, _ = load_digits(return_X_y=True)
        np.testing.assert_allclose(self.scaler.mean_, x[self.ids['train']].mean(0))
        np.testing.assert_allclose(self.data['train'][0].numpy().mean(0), 0, atol=1e-6)

    def test_dropout_disabled_in_evaluation(self):
        model = network(.5)
        model.eval()
        x = self.data['val'][0]
        self.assertTrue(torch.equal(model(x), model(x)))
        model.train()
        self.assertFalse(torch.equal(model(x), model(x)))

    def test_training_repeatable(self):
        config = dict(dropout=.3, l2=.001, early=False)
        a, history_a, _ = fit(self.data, config, 11, epochs=3)
        b, history_b, _ = fit(self.data, config, 11, epochs=3)
        self.assertEqual(history_a, history_b)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_restored_checkpoint_and_probability_contract(self):
        model, history, epoch = fit(self.data, dict(dropout=0., l2=0., early=True),
                                    11, epochs=20, patience=3)
        metrics, prob = evaluate(model, self.data['val'])
        self.assertAlmostEqual(metrics['loss'], history[epoch-1]['val_loss'], places=6)
        self.assertEqual(prob.shape, (len(self.ids['val']), 10))
        np.testing.assert_allclose(prob.sum(1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(prob).all())
        self.assertLess(history[-1]['train_loss'], history[0]['train_loss'])


if __name__ == '__main__':
    unittest.main()
