import json
import unittest
import requests
from main import app

with open("testData/goodTestData.json") as jsonfile:
    goodData = json.load(jsonfile)

class TestRestAPI(unittest.TestCase):


    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def testsingleJSON(self):
        response = requests.post('http://127.0.0.1:8080/predict', json = goodData[0], headers = {'Content-Type': 'application/json'})
        print(response.status_code)
        self.assertEqual(response.status_code, 200)
    def testmultipleJSON(self):
        response = requests.post('http://127.0.0.1:8080/predict', json = goodData[0:3], headers = {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 200)
    # Todo fix error code to return 400 not 500
    def testNull(self):
        response = requests.post('http://127.0.0.1:8080/predict', json ={}, headers = {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 500)

if __name__ == '__main__':
    unittest.main()
