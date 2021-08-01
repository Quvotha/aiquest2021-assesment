import unittest

import numpy as np
import pandas as pd

from review import Review


class TestReview(unittest.TestCase):

    def test_cleanse_host_response_rate(self):
        testdata = (
            ('100%', 100.),
            ('0%', 0.),
            ('25%', 25.),
            ('31.1%', 31.1)
        )
        for input, expected in testdata:
            output = Review.cleanse_host_response_rate(input)
            self.assertEqual(expected, output, (input, expected, output))

        expected = None
        output = Review.cleanse_host_response_rate(np.nan)
        self.assertIs(expected, output, output)

    def test_extract_review_feature(self):
        input = pd.DataFrame(
            {
                'first_review': ['2000-01-01', '2000-01-02', '2000-01-03', '2000-01-04'],
                'last_review': ['2000-01-10', '2000-01-22', '2000-01-13', '2000-01-29'],
                'number_of_reviews': [100, 50, 200, 200],
                'host_response_rate': [0.25, 0.5, 1.0, 0.0]
            }
        )
        expected = pd.DataFrame(
            {
                'days_between_first_last': [9, 20, 10, 25],
                'average_number_of_reviews': [100. / 9, 2.5, 20., 8.],
                'number_of_host_response': [25., 25., 200., 0.],
                'average_number_of_host_response': [25./9, 1.25, 20., 0.]

            }
        )
        output = Review.extract_review_feature(input)
        self.assertIsNone(pd.testing.assert_frame_equal(expected, output))

        input = input.rename(columns={'first_review': 'host_since'})
        output = Review.extract_review_feature(input, first_date_column='host_since')
        self.assertIsNone(pd.testing.assert_frame_equal(expected, output))


if __name__ == '__main__':
    unittest.main()
