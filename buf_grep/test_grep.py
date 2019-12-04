import unittest
import grep

class testGrepCase:
    def __init__(self, nameTest, inputData, expected):
        self.nameTest  = nameTest
        self.inputData = inputData
        self.expected  = expected

class TestGrepCase(unittest.TestCase):
    def test_case(self):
        testCase = [
            testGrepCase('Check that a string is in [] and not a substring', ['grep', "301 0", "GET https://mail.ru/fitness"],
             '21/Mar/2018 21:53:10 [301 0] "GET https://mail.ru/fitness/pay_list HTTP/1.1"'
            ),
            testGrepCase('Check that if a match is in [], but without a match in the message, a None is displayed',
            ['grep', "500 120426", "GET https://mail.ru/fitness"], None
            ),
            testGrepCase('Check that if a message matches, but does not match in [], a None is displayed',
            ['grep', "500 120", "GET https:"], None
            ),
            testGrepCase('Check that if the message length is less than the line length in [], then everything matches correctly',
            ['grep', "200 1845 500 120426", "GE"], '21/Mar/2018 21:32:11 [200 1845 500 120426] "GET https://mail.ru/static/js/jquery-go-top/go-top.png HTTP/1.1"'
            )
        ]
        for t in testCase:
            print("Running test :", t.nameTest)
            self.assertEqual(t.expected, grep.main(t.inputData))
            print('done')

if __name__ == '__main__':
    unittest.main()
