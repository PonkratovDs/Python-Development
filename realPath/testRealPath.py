import unittest
import os
import realPath


class testRealPathCase:
    def __init__(self, nameTest, inputData, expected):
        self.nameTest = nameTest
        self.inputData = inputData
        self.expected = expected


class TestRealPathCase(unittest.TestCase):
    def test_case(self):
        testCase = [
            testRealPathCase('global test', {'test': './../.././//cd/etc/../pas.sw//.././sdf/.'},
                             os.path.realpath(
                                 './../.././//cd/etc/../pas.sw//.././sdf/.')
                             ),
            testRealPathCase('Checking the operation of two points', {'test': '/dfg/..'},
                             os.path.realpath(
                '/dfg/..')
            ),
            testRealPathCase('Checking the operation of one points', {'test': '/dfg/.'},
                             os.path.realpath(
                '/dfg/.')
            ),
            testRealPathCase('Checking the operation of lot points', {'test': './cd/etc/../pas.sw//...././sdf/..'},
                             '/home/danilp/py_file/Python-Development/realPath/cd/pas.sw'),
        ]

        for t in testCase:
            print("Running test :", t.nameTest)
            self.assertEqual(t.expected, realPath.main(t.inputData['test']))
            print('done')
