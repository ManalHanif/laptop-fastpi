from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
#import sklearn
model = joblib.load('Kmeans_model.joblib')
scaler = joblib.load('scaler.joblib')
app = FastAPI()
# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}
# get request
@app.get("/items/")
def create_item(item: dict):
    return {"item": item}

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Processor_Speed: float
    RAM_Size: int
    Storage_Type_encoded: int
    Storage_Value: int

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Processor Speed': input_features.Processor_Speed,
        'RAM Size': input_features.RAM_Size,
        'Storage_Type_encoded': input_features.Storage_Type_encoded,
        'Storage Value': input_features.Storage_Value,
    }
    
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    # Scale the input features
    scaled_features = scaler.transform([list(dict_f.values())])
    return scaled_features
@app.get("/predict")
def predict(input_features: InputFeatures):
    return preprocessing(input_features)


@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}
