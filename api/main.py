from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="ML Prediction API")

# Load the model (and vectorizer)
# Best practice: these are bundled as a single Pipeline object
try:
    model = joblib.load("models/model.pkl")
except FileNotFoundError:
    model = None
    print("Warning: Model file not found.")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str  # or int, depending on your model

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # If 'model' is a Pipeline, it handles the vectorization internally
        # We access the text via the Pydantic request object
        prediction = model.predict([request.text])[0]
        return {"prediction": str(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#from fastapi import FastAPI 
#import joblib
#app = FastAPI() 
#model, vectorizer = joblib.load("models/model.pkl")
#
#@app.post("/predict")
#def predict(text: str):
#    vec = vectorizer.transform([text])
#    return {"prediction": model.predict(vec)[0]}
#

#from fastapi import FastAPI
#import joblib
#
#app = FastAPI()
#model, vectorizer = joblib.load("models/model.pkl")
#
#@app.post("/predict")
#def predict(text: str):
#    vec = vectorizer.transform([text])
#    return {"prediction": model.predict(vec)[0]}
#


#@app.get("/hello") 
#def read_root(): 
#    return {"Hello": "World"}
