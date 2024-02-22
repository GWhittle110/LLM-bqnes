from unittest import TestCase
from src.utils.zipper import zipper


class TestZipper(TestCase):
    def test_zipper(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = 1
        d = [1]
        self.assertEqual(list(zipper(a, b)), [(1, 4), (2, 5), (3, 6)])
        self.assertEqual(list(zipper(a, c)), [(1, 1), (2, 1), (3, 1)])
        self.assertEqual(list(zipper(a, d)), [(1, 1), (2, 1), (3, 1)])
        self.assertEqual(list(zipper(c, c)), [(1, 1)])

    def test_strings(self):
        a = [1, 2, 3]
        b = "hello"
        self.assertEqual(list(zipper(a, b)), [(1, "hello"), (2, "hello"), (3, "hello")])
