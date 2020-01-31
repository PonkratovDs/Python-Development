from os import path
import unittest
import real_path
import random
import string


class TestCase:

    def __init__(self, input_data, expected, num_test):
        super().__init__()
        self.input_data = input_data
        self.expected = expected
        self.num_test = num_test


class TestRealPath(unittest.TestCase):
    def _get_test_case(self):
        test_cases = []
        for num in range(0, 1000):
            random_path = get_random_path()
            test_cases.append(
                TestCase(
                    input_data=random_path,
                    expected=path.realpath(random_path),
                    num_test=num)
            )
        return test_cases

    def test_case(self):
        test_cases = self._get_test_case()
        for t in test_cases:
            print('Test number {num}  is run'.format(num=t.num_test), end=' ... ')
            RP = real_path.RealPath(t.input_data)
            #print(RP.user_path)                    #for debug
            self.assertEqual(hash(t.expected), hash(RP.real_path()))
            del RP
            print('Done')


def get_random_path(size=256):
    slash_dot_empty = ['.', '/', '/./', '/../']
    slash_dot_empty.extend(['' for _ in range(100)])
    random_path = ''.join(
        random.choice(
            string.ascii_letters +
            string.digits) +
        ''.join(
            random.choices(
                slash_dot_empty,
                k=10)) for _ in range(size))
    return random.choice([random_path, './' + random_path])
