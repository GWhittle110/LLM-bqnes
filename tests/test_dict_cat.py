from unittest import TestCase
from src.utils.dictCat import dict_cat


class TestAccuracy(TestCase):

    def test_accuracy_from_predictions(self):
        dict0 = {"a": 1, "b": 2}
        dict1 = {"a": [1], "b": [2]}
        dict2 = {"a": [1], "b": 2}
        dict3 = {"a": 1, "b": [2]}
        dict4 = {"a": [1], "c": 3}
        dict5 = dict()
        self.assertEqual(dict_cat(dict0, dict1), {"a": [1, 1], "b": [2, 2]})
        self.assertEqual(dict_cat(dict0, dict2), {"a": [1, 1], "b": [2, 2]})
        self.assertEqual(dict_cat(dict0, dict3), {"a": [1, 1], "b": [2, 2]})
        self.assertEqual(dict_cat(dict0, dict4), {"a": [1, 1], "b": [2], "c": [3]})
        self.assertEqual(dict_cat(dict0, dict3), {"a": [1], "b": [2]})
