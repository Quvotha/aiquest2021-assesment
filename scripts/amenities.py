from typing import List, Union


class Amenities(object):

    @staticmethod
    def tolist(amenities: str) -> List[str]:
        return [a.replace('"', '').replace(' ', '-') for a in amenities[1:-1].split(',')]
