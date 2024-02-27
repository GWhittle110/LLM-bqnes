from unittest import TestCase
from src.utils.expectedCalibrationError import expected_calibration_error_from_predictions
import numpy as np


class TestAccuracy(TestCase):

    def test_expected_calibration_error_from_predictions(self):
        predictions = np.array([[0.15, 0.85], [0.75, 0.25], [0.65, 0.35], [0.85, 0.15]])
        targets = np.array([1, 0, 1, 0])
        self.assertEqual(np.round(expected_calibration_error_from_predictions(predictions, targets), 4), 0.3)
        predictions = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        targets = np.array([1, 0, 1, 0])
        self.assertEqual(np.round(expected_calibration_error_from_predictions(predictions, targets), 4), 0)

