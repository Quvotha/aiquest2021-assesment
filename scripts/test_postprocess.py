import unittest

import numpy as np
import pandas as pd

from postprocess import SimilarHomeSearcher


class TestSimilarHomeSearcher(unittest.TestCase):

    @staticmethod
    def get_searcher():
        return SimilarHomeSearcher(
            pd.DataFrame(
                data=[
                    ['desc1', 'city1', 'code1', 'name1', 2.],
                    ['desc1', 'city1', 'code1', 'name2', 4.],
                    ['desc1', 'city1', 'code1', 'name2', 8.],
                    ['desc1', 'city1', 'code2', 'name2', 12.],
                    ['desc1', 'city2', 'code3', 'name2', 16.],
                    ['desc1', 'city2', 'code3', 'name2', 32.],
                ],
                columns=['description', 'city', 'zipcode5', 'name', 'y']
            )
        )

    def test_match_condition1(self):
        searcher = TestSimilarHomeSearcher.get_searcher()
        testdata = (
            # (description, city, zipcode5, name, default), expected
            (('desc1', 'city1', 'code1', 'name1', 1.), 2.),
            (('desc1', 'city1', 'code1', 'name2', 1.), np.mean([4., 8.])),
            (('desc1', 'city1', 'code2', 'name2', 1.), 12.),
            (('desc1', 'city2', 'code3', 'name2', 1.), np.mean([16., 32.])),
        )
        for input, expected in testdata:
            description, city, zipcode5, name, default = (
                input[0], input[1], input[2], input[3], input[4]
            )
            with self.subTest(description=description, city=city, zipcode5=zipcode5, name=name, default=default):
                output = searcher.get_y_of_similar(description, city, zipcode5, name, default)
                self.assertEqual(output, expected, (output, expected))

    def test_match_condition2(self):
        searcher = TestSimilarHomeSearcher.get_searcher()
        testdata = (
            # (description, city, zipcode5, name, default), expected
            (('desc1', 'city1', 'code1', 'name0', 1.), np.mean([2., 4., 8.])),
            (('desc1', 'city1', 'code2', 'name0', 1.), 12.),
            (('desc1', 'city2', 'code3', 'name0', 1.), np.mean([16., 32.])),
        )
        for input, expected in testdata:
            description, city, zipcode5, name, default = (
                input[0], input[1], input[2], input[3], input[4]
            )
            with self.subTest(description=description, city=city, zipcode5=zipcode5, name=name, default=default):
                output = searcher.get_y_of_similar(description, city, zipcode5, name, default)
                self.assertEqual(output, expected, (output, expected))

    def test_match_condition3(self):
        searcher = TestSimilarHomeSearcher.get_searcher()
        testdata = (
            # (description, city, zipcode5, name, default), expected
            (('desc1', 'city1', 'code0', 'name0', 1.), np.mean([2., 4., 8., 12.])),
            (('desc1', 'city2', 'code0', 'name0', 1.), np.mean([16., 32.])),
        )
        for input, expected in testdata:
            description, city, zipcode5, name, default = (
                input[0], input[1], input[2], input[3], input[4]
            )
            with self.subTest(description=description, city=city, zipcode5=zipcode5, name=name, default=default):
                output = searcher.get_y_of_similar(description, city, zipcode5, name, default)
                self.assertEqual(output, expected, (output, expected))

    def test_match_condition4(self):
        searcher = TestSimilarHomeSearcher.get_searcher()
        testdata = (
            # (description, city, zipcode5, name, default), expected
            (('desc1', 'city3', 'code1', 'name1', 1.), 1.),
            (('desc2', 'city1', 'code1', 'name2', 300.), 300.),
        )
        for input, expected in testdata:
            description, city, zipcode5, name, default = (
                input[0], input[1], input[2], input[3], input[4]
            )
            with self.subTest(description=description, city=city, zipcode5=zipcode5, name=name, default=default):
                output = searcher.get_y_of_similar(description, city, zipcode5, name, default)
                self.assertEqual(output, expected, (output, expected))


if __name__ == '__main__':
    unittest.main()
