import unittest

from cleansing import ZipCode


class TestZipCode(unittest.TestCase):

    def test_normalize(self):
        # (input, expected)
        testdata = (
            # 典型的なパターン
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


if __name__ == '__main__':
    unittest.main()
