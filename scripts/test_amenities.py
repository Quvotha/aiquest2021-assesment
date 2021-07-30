import unittest

from amenities import Amenities


class TestAmenities(unittest.TestCase):

    def test_tolist(self):
        testdata = (
            ('{TV,Wireless Internet,Kitchen}', ['TV', 'Wireless-Internet', 'Kitchen']),
            ('{TV,"Cable TV",Internet,"Air conditioning",Kitchen,"Free parking on premises"}',
             ['TV', 'Cable-TV', 'Internet', 'Air-conditioning', 'Kitchen',
              'Free-parking-on-premises']),
            ('{"Air conditioning","translation missing: en.hosting_amenity_49"}',
             ["Air-conditioning", "translation-missing:-en.hosting_amenity_49"])
        )
        for input, expected in testdata:
            output = Amenities.tolist(input)
            self.assertEqual(output, expected, msg=f'{input}, {expected}, {output}')


if __name__ == '__main__':
    unittest.main()
