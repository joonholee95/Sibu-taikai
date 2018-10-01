import unittest

def plus(one, two):
    return one + two

class Time(unittest.TestCase):
    def testalzio(self):
        self.assertEquals(plus(15, 5), 20)
        
if __name__ == '__main__':        
    unittest.main()