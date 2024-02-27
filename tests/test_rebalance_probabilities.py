from unittest import TestCase
from src.utils.rebalanceProbabilities import rebalance_probabilities
import torch


class TestRebalanceProbabilities(TestCase):

    def test_rebalance_probabilities(self):
        a = torch.tensor([0.2, 0.8])
        b = a.clone() * 2
        c = torch.tensor([-0.5, 1.5])
        d = torch.tensor([[0.2, 0.8], [0.4, 0.6], [-0.5, 1.5]])
        print(rebalance_probabilities(d))
        self.assertEqual(rebalance_probabilities(a).numpy().tolist(), a.numpy().tolist())
        self.assertEqual(rebalance_probabilities(b).numpy().tolist(), a.numpy().tolist())
        self.assertEqual(rebalance_probabilities(c).numpy().tolist(), torch.tensor([0, 1]).numpy().tolist())
        self.assertEqual(rebalance_probabilities(d).numpy().tolist(),
                         torch.tensor([[0.2, 0.8], [0.4, 0.6], [0, 1]]).numpy().tolist())
