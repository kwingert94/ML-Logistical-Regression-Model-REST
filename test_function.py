import unittest

from pandas import Index

from main import *



class Test(unittest.TestCase):

    def test_find_variables(self):

        variables = ['x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October', 'x5_sunday', 'x31_asia',
                     'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', 'x53', 'x81_November',
                     'x44',
                     'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany', 'x58', 'x56']
        toTest = [Index(['x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October',
                         'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May', 'x5_monday',
                         'x81_September', 'x81_March', 'x53', 'x81_November', 'x44', 'x81_June',
                         'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany',
                         'x58', 'x56'],
                        dtype='object')]
        result = find_variables(toTest)
        result.sort(), variables.sort()
        self.assertEqual(result, variables)

