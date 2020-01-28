from os import path
import unittest
import real_path


class TestCase:

    def __init__(self, input_data, expected):
        super().__init__()
        self.input_data = input_data
        self.expected = expected


class TestRealPath(unittest.TestCase):

    def test_case(self):
        test_case = [
            TestCase('.../.s/././/sd//../.d./../..',
                     path.realpath('.../.s/././/sd//../.d./../..')),
            TestCase('/.s/././/sd//../.d./../..',
                     path.realpath('/.s/././/sd//../.d./../..')),
            TestCase('.../.s/././/sd///.d./../..',
                     path.realpath('.../.s/././/sd///.d./../..')),
            TestCase('.../.s/././/sd/gj//.d.//',
                     path.realpath('.../.s/././/sd/gj//.d.//'))
        ]
        for t in test_case:
            RP = real_path.RealPath(t.input_data)
            self.assertEqual(t.expected, RP.real_path())
            del RP
