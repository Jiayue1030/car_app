import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.neighbors import NearestNeighbors

features = [
    "data_installment",
    "seat_capacity",
    "data_transmission",
    "engine_cc",
    "fuel_type",
]
FILE_PATH = "https://drive.google.com/uc?id=1GEm1cil7gRrvzINAaaHJWOpVyX9LmnYX"
FULL_DETAILS_FILE_PATH = ""
carlist = pd.read_csv(FILE_PATH)

def get_recommendation(carlist, user_input):
    result = dict()
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    user_input = {
        "data_installment": user_input["data_installment"],
        "seat_capacity": user_input["seat_capacity"],
        "engine_cc": user_input["engine_cc"],
        "data_transmission": user_input["data_transmission"],
        "fuel_type": user_input["fuel_type"],
    }

    # carlist = carlist.loc[carlist['data_installment']<=user_input['data_installment']]
    carlist = carlist.loc[carlist["seat_capacity"] == user_input["seat_capacity"]]
    # carlist = carlist.loc[carlist['data_transmission']==user_input['data_transmission']]
    # carlist = carlist.loc[carlist['fuel_type']==user_input['fuel_type']]

    user_input = pd.DataFrame(user_input, index=features)
    user_input = user_input.transpose()

    if len(carlist) > 1:
        model.fit(carlist[features])
        distances, indices = model.kneighbors(user_input)
        similar_vehicles = carlist.iloc[indices[0], :]
        result = {"code": 200, "similar_vehicles": similar_vehicles}
    else:
        result["code"] = 400
        result["message"] = "Not Found. Please reset your criteria."

    return result
