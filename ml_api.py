from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

input_sample_keys = [
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]

input_sample_value = [
    710,
    3,
    'Moubarek, Master. Halim Gonios ("William George")',
    "male",
    -9999.0,
    1,
    1,
    "2661",
    15.2458,
    -9999,
    "C",
]


class model_input(BaseModel):
    PassangerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: int
    Embarked: str


titanic_model = pickle.load(open("modelo.pkl", "rb"))


@app.post("/titanic_prediction")
def titanic_pred(input_params: model_input):
    input_data = input_params.json()

    # converts json input to python dict
    input_dictionary = json.loads(input_data)

    # now we convert our dict to a list
    passanger_id = input_dictionary["PassangerId"]
    pclass = input_dictionary["Pclass"]
    name = input_dictionary["Name"]
    sex = input_dictionary["Sex"]
    age = input_dictionary["Age"]
    sibsb = input_dictionary["SibSp"]
    parch = input_dictionary["Parch"]
    ticket = input_dictionary["Ticket"]
    fare = input_dictionary["Fare"]
    cabin = input_dictionary["Cabin"]
    embarked = input_dictionary["Embarked"]

    input_list = [
        passanger_id,
        pclass,
        name,
        sex,
        age,
        sibsb,
        parch,
        ticket,
        fare,
        cabin,
        embarked,
    ]

    prediction = titanic_model.predict([input_list])

    if prediction[0] == 0:
        return "Essa pessoa sobreviveu ao Titanic"
    else:
        return "Essa pessoa não sobreviveu ao titanic"
