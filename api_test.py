import json
import requests

url = "http://127.0.0.1:8000/titanic_prediction"

input_data_for_model = {
    "PassangerId": 710,
    "Pclass": 3,
    "Name": "Moubarek, Master. Halim Gonios ('William George')",
    "Sex": "male",
    "Age": -9999.0,
    "SibSp": 1,
    "Parch": 1,
    "Ticket": "2661",
    "Fare": 15.2458,
    "Cabin": -9999,
    "Embarked": "C",
}

# transformando o dict em json para mandar os dados para nossa api

input_json = json.dumps(input_data_for_model)
response = requests.post(url, data=input_json)
print(response.text)
