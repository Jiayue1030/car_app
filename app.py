from shiny import *
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import requests

# https://docs.google.com/document/d/1HxcNTPyc8kSnJPwXxZL_vK80l60Mjam72omXK9yfxOk/edit

app_ui = ui.page_fluid(
    ui.h2("Vehicle Recommendation System"),
    ui.row(ui.column(6,ui.input_slider("data_installment",
                "What is your preferred monthly installment?",
                300,6000,500,),),
        ui.column(6,ui.input_slider("seat_capacity", "How many seats do you prefer?", 2, 9, 5),),
        ui.column(6, ui.input_slider("engine_cc", "Preferred engine cc", 300, 6000, 1000)),
        ui.column(6,ui.input_select("fuel_type",
                "Preferred Fuel Type",
                {"0": "Diesel","1": "Electric","2": "Hybrid","3": "Petrol - Unleaded (ULP)",},),),
        ui.column(6,ui.input_select("data_transmission","Preferred Transmission",
                {"0": "Automatic","1": "Manual",},),),
        ui.column(6,ui.input_select("car_model","Preferred car model",
                {"0": "Sedans","1": "Hatchbacks","2": "Sports-Utility Vehicle (SUV)",
                 "3": "Station Wagon","4": "Multi-Purpose Van (MPV)","5": "Coupe",
                 "6": "Convertible",},),),
    ),
    # ui.row(ui.output_text_verbatim("get_user_inputs")),
    ui.row(ui.input_action_button("run", "Surprise Me!", class_="btn-primary m-10")),
    # ui.output_text_verbatim("get_user_inputs")
    ui.output_ui("get_user_inputs")
)

def server(input, output, session):
    @output
    # @render.text()
    @render.ui()
    def get_user_inputs():
        CLEANED_FILE_PATH = Path(__file__).parent / "df_carlist_cleaned.csv" #for live editor
        #CLEANED_FILE_PATH = 'https://gist.githubusercontent.com/Jiayue1030/335451e504b842c0821842eb9ab45fe6/raw/5c95f1432fb122340d03e31935ce3a69c8a8e352/df_carlist_cleaned.csv'
        full_carlist = pd.read_csv(CLEANED_FILE_PATH)
        
        user_inputs_dict = {
            "data_installment": input.data_installment(),
            "seat_capacity": input.seat_capacity(),
            "engine_cc": input.engine_cc(),
            "data_transmission": input.data_transmission(),
            "fuel_type": input.fuel_type(),
        }
        
        result = get_recommendation(user_inputs_dict)

        if result["code"] == 200:
            vehicles = result["similar_vehicles"]
            vehicle_ids = vehicles.loc[:, 'vehicle_id']
            vehicles_data = full_carlist[full_carlist['vehicle_id'].isin(vehicle_ids)]
            vehicle_cards = build_recommendation_ui(vehicles_data)
            
        else:
            vehicle_cards = result
        
        return vehicle_cards
        
    def build_recommendation_ui(vehicles_data):
        cards = [ui.column(4,class_="card", children=[
                    ui.img(src="https://img1.icarcdn.com/4031499/gallery_new-car-carlist-proton-saga-standard-sedan-malaysia_000004031499_67a33b5c_65c9_4b22_a02d_dfd947d034ab.png"),
                    ui.column(12,class_="card-body", children=[
                        ui.h5(row["data_display_title"]),
                        ui.p("Monthly Installment : RM ",row["data_installment"]),
                        ui.p("Engine CC:",row["engine_cc"]),
                        ui.p("Seat Capacity:",row["seat_capacity"]),
                        ui.p("Price: RM ",row["listing_price"]),
                    ])
                ]) for i,row in vehicles_data.iterrows()]
        
        cards_ui = ui.div(class_="row mt-5", children=cards)
        
        return cards_ui


def get_recommendation(user_input):
    features = [
        "data_installment",
        "seat_capacity",
        "data_transmission",
        "engine_cc",
        "fuel_type",
    ]
    path = Path(__file__).parent / "df_modelling.csv" #for live editor
    #path = 'https://gist.githubusercontent.com/Jiayue1030/335451e504b842c0821842eb9ab45fe6/raw/a07b9d4a69fe0a638551083ee1d5937482331896/df_modelling.csv'
    carlist = pd.read_csv(path)

    result = dict()
    model_path = Path(__file__).parent / "model.pkl"
    model = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
    model.fit(carlist[features])

    user_input = {
        "data_installment": user_input["data_installment"],
        "seat_capacity": user_input["seat_capacity"],
        "engine_cc": user_input["engine_cc"],
        "data_transmission": user_input["data_transmission"],
        "fuel_type": user_input["fuel_type"],
    }

    carlist = carlist.loc[carlist["seat_capacity"] == user_input["seat_capacity"]]

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

##--------------------------------------------------------------------------------------

app = App(app_ui, server)

# display成card，link去url
# 按了button后才显示