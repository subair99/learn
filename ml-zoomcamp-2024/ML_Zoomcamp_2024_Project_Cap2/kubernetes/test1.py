import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://github.com/subair99/ML_Zoomcamp_2024_Project_Cap2/blob/main/test_data/glioma-Te-glTr_0007.jpg'}

result = requests.post(url, json=data).json()
print(result)