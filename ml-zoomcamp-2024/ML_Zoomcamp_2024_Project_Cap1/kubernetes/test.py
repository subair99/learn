import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://github.com/subair99/ML_Zoomcamp_2024_Project_Cap1/blob/main/test_data/0_857_1223599388.jpg'}

result = requests.post(url, json=data).json()
print(result)