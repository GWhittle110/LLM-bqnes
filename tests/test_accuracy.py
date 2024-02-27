from unittest import TestCase
from src.utils.accuracy import accuracy_from_predictions
import numpy as np


class TestAccuracy(TestCase):

    def test_accuracy_from_predictions(self):
        predictions = np.array([[0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
        targets = np.array([1, 0, 1, 0])
        self.assertEqual(accuracy_from_predictions(predictions, targets), 0.75)
