import unittest

import numpy as np
import pandas as pd

from zipcode import ZipCode, ZipCodePreprocessor


class TestZipCode(unittest.TestCase):

    def test_normalize(self):
        # (input, expected)
        testdata = (
            # 典型的なパターン
            ('00002', '00002'),
            ('10002', '10002'),
            ('94014', '94014'),
            ('91606-1412', '91606-1412'),
            ('90034-2203', '90034-2203'),

            # 小数部を持つ整数でも郵便番号として扱える
            ('10002.0', '10002'),
            ('94014.0', '94014'),

            # 接頭語 'Near ' が付いていても郵便番号を読み取れる
            ('Near 91304', '91304'),
            ('Near 91606-1412', '91606-1412'),

            # コンペティションデータに含まれる汚いデータに固有の処理
            ('11249\r\r\r\r\r\r\n11249', '11249'),
            ('95202\r\r\r\r\r\r\n\r\r\r\r\r\r\n\r\r\r\r\r\r\n94158', '94158'),

            # zipcode のように見えても str でなければ郵便番号とはみなさない
            (10002, ''),
            (94014, ''),

            # 無理なものは諦める
            ('1m', ''),
            (' ', ''),
            ('9160611412', ''),
            ('91606-14121', '')
        )
        for (input, expected) in testdata:
            output = ZipCode.normalize(input)
            self.assertEqual(output, expected, msg=f'{input}, {expected}, {output}')

    def test_first_digit_of(self):
        # (input, expected), input = (zipcode, city)
        testdata = (
            # 最初の1桁目を得る
            (('12325', 'Boston'), '1'),
            (('23231', 'Chicago'), '2'),
            (('90014', 'DC'), '9'),
            (('50123', 'LA'), '5'),
            (('21011', 'NYC'), '2'),
            (('20212-1312', 'SF'), '2'),

            # ブランクなら都市名から予測を試みる
            (('', 'Boston'), '0'),
            (('', 'Chicago'), '6'),
            (('', 'DC'), '2'),
            (('', 'LA'), '9'),
            (('', 'NYC'), '1'),
            (('', 'SF'), '9'),

            # 予測できなければブランク
            (('', 'JPN'), ''),
            (('', 'USA'), ''),
        )


class TestZipCodePreprocessor(unittest.TestCase):

    def test_preprocess(self):
        input = pd.DataFrame(
            columns=['city', 'zipcode'],
            data=[
                ('Boston', '00001'),
                ('Boston', np.nan),
                ('NYC', '10002'),
                ('NYC', '10002'),
                ('NYC', '10002'),
                ('NYC', '10003'),
                ('NYC', np.nan),
                ('DC', '20001'),
                ('DC', '20001'),
                ('DC', '20002'),
                ('DC', '20003'),
                ('DC', np.nan),
                ('Chicago', '60001'),
                ('Chicago', '60001'),
                ('Chicago', '60002'),
                ('Chicago', '60002'),
                ('Chicago', np.nan),
                ('SF', '90001'),
                ('SF', '90002'),
                ('SF', '90002'),
                ('SF', '90002-1142'),
                ('SF', np.nan),
            ]
        )
        expected = pd.DataFrame(
            columns=['zipcode_1st_digit', 'zipcode5', 'zipcode_imputed'],
            data=[
                ('0', '00001', 0),
                ('0', '00001', 1),
                ('1', '10002', 0),
                ('1', '10002', 0),
                ('1', '10002', 0),
                ('1', '10003', 0),
                ('1', '10002', 1),
                ('2', '20001', 0),
                ('2', '20001', 0),
                ('2', '20002', 0),
                ('2', '20003', 0),
                ('2', '20001', 1),
                ('6', '60001', 0),
                ('6', '60001', 0),
                ('6', '60002', 0),
                ('6', '60002', 0),
                ('6', '60001', 1),
                ('9', '90001', 0),
                ('9', '90002', 0),
                ('9', '90002', 0),
                ('9', '90002', 0),
                ('9', '90002', 1),
            ],
        )
        expected['zipcode_imputed'] = expected['zipcode_imputed'].astype('uint8')
        preprocessor = ZipCodePreprocessor().fit(input)
        self.assertEqual(
            preprocessor.mode_,
            {
                '0': '00001',
                '1': '10002',
                '2': '20001',
                '6': '60001',
                '9': '90002'
            },
            preprocessor.mode_
        )
        output = preprocessor.transform(input)
        self.assertIsNone(pd.testing.assert_frame_equal(expected, output))


if __name__ == '__main__':
    unittest.main()
