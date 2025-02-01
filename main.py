from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware  

# Load the trained model
with open("crop_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)
    print("Model loaded successfully!")  # Console print to check if the model loaded correctly

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins, adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Input data schema
class CropRecommendationInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Endpoint to get crop recommendations
@app.post("/predict")
async def predict(input_data: CropRecommendationInput):
    # Prepare input data as DataFrame
    input_df = pd.DataFrame([{
        "N": input_data.N,
        "P": input_data.P,
        "K": input_data.K,
        "temperature": input_data.temperature,
        "humidity": input_data.humidity,
        "ph": input_data.ph,
        "rainfall": input_data.rainfall
    }])

    # Add engineered features
    input_df["NPK_ratio"] = input_df["N"] / (input_df["P"] + input_df["K"] + 1)
    input_df["Temp_Rainfall"] = input_df["temperature"] * input_df["rainfall"]

    # Normalize the input features using StandardScaler
    scaler = StandardScaler()
    input_df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]] = scaler.fit_transform(
        input_df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    )

    # After normalization, create norm columns explicitly
    for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        input_df[f"{col}_norm"] = input_df[col]

    # Add the engineered features to the final feature set
    feature_cols = ["NPK_ratio", "Temp_Rainfall"] + [f"{col}_norm" for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    
    # Select the final feature columns used in training
    final_input = input_df[feature_cols]

    # Make prediction
    try:
        prediction = model.predict(final_input)
        return {"recommended_crop": prediction[0]}
    except Exception as e:
        return {"error": str(e)}
