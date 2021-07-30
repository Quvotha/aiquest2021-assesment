import math
import re
from typing import Any


class ZipCode(object):

    NOT_ZIP_CODE = ''

    # General patterns
    PATTERN_GENERAL = re.compile(r'\d{5}-\d{4}$|\d{5}$')

    # Competition specific dirty pattern
    PATTERN_STARTS_WITH_Near = re.compile(r'^Near \d{5}-\d{4}$|^Near \d{5}$')
    PATTERN_DIRTY = re.compile(r'\d{5}-\d{4}[\r+\n]+\d{5}-\d{4}$|\d{5}[\r+\n]+\d{5}$')

    CITY_FIRST_DIGIT_MAP = {
        'Boston': 0,
        'Chicago': 6,
        'DC': 2,
        'LA': 9,
        'NYC': 1,
        'SF': 9
    }

    @classmethod
    def normalize(cls, zipcode: Any) -> str:
        """Format zipcode in 'NNNNN' or 'NNNNN-NNNN' format string if possible.

        Verify if `zipcode` can be interpreted as zipcode represented as 5 digit number or 5 digit number + "-" + 4 digit number format.
        Return zipcode formatted in "NNNNN" or "NNNNN-NNNNN" style if interpretable, otherwise return blank string.

        `zipcode` is tried to be interpreted only when it is str instance.

        - '10002', '10002.0: Both will be interpreted as '10002'.
        - '91606-1412': Interpreted as '91606-1412'.
        - '10002.1', '91606-14121', '9160611412', '1m': They will not be interpreted as zipcode.

        There are some specific patterns in competition data.

        - 'Near 91304': zipcode starts with 'Near '. It will be interpreted as '91304'.
        - '11249\r\r\r\r\r\r\n11249', '95202\r\r\r\r\r\r\n\r\r\r\r\r\r\n\r\r\r\r\r\r\n94158': 2 zipcodes which are separated by following pattern.
            <several '\r' + single 'n' pattern> + <several '\r' + single 'n' pattern> + ...
        First one will be interpreted as '11249', second one as '94158'. 

        Parameters
        ----------
        zipcode : Any
            Zipcode to be normalized.

        Returns
        -------
        str
             'NNNNN' or 'NNNNN-NNNN' style zipcode, or blank string
        """
        if not isinstance(zipcode, str):
            return cls.NOT_ZIP_CODE

        # if `zipcode` is float-like integer like '10002.0', convert it into integer.
        # i.e. '10002.0' --> '10002', '10004.0' --> '10004'
        try:
            float_zipcode = float(zipcode)
            decimal_part, integer_part = math.modf(float_zipcode)
            if decimal_part != 0:
                return cls.NOT_ZIP_CODE
            zipcode = str(int(integer_part))
        except ValueError:
            # `zipcode` is not float-like
            pass

        # NNNNN or NNNNN-NNNN pattern
        m = cls.PATTERN_GENERAL.match(zipcode)
        if m is not None:
            return zipcode

        # General pattern starts with 'Near '
        m = cls.PATTERN_STARTS_WITH_Near.match(zipcode)
        if m is not None:
            return zipcode.replace('Near ', '')

        # 2 zipcodes separated by several <several '\r's + single '\n'> patterns
        m = cls.PATTERN_DIRTY.match(zipcode)
        if m is not None:
            zipcodes = re.findall(cls.PATTERN_GENERAL, zipcode)
            return zipcodes[-1]

        # Any pattern is not matched
        return cls.NOT_ZIP_CODE

    @classmethod
    def first_digit_of(cls, zipcode: str, city: str) -> int:
        if not isinstance(zipcode, str):
            return cls.NOT_ZIP_CODE
        else:
            return zipcode[0] if zipcode else cls.CITY_FIRST_DIGIT_MAP.get(city, cls.NOT_ZIP_CODE)
