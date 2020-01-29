from os import path
import unittest
import real_path
import random
import string

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
        for t in enumerate(test_case):
            print('Test number {num} is run'.format(num=t[0]), end=' ... ')
            RP = real_path.RealPath(t[1].input_data)
            self.assertEqual(t[1].expected, RP.real_path())
            del RP
            print('Done')


def get_random_path(size=256):
    slash_dot_empty = ['.', '/', '/./', '/../']
    slash_dot_empty.extend(['' for _ in range(100)])
    random_path = ''.join(random.choice(string.ascii_letters + string.digits) + ''.join(random.choices(slash_dot_empty, k=10)) for _ in range(size))
    return random.choice([random_path, './' + random_path])

