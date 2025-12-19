from fastapi import FastAPI 
import joblib
app = FastAPI() 
model, vectorizer = joblib.load("models/model.pkl")

@app.get("/") 
def read_root(): 
    return {"Hello": "World"}

@app.post("/predict")
def predict(text: str):
    vec = vectorizer.transform([text])
    return {"prediction": model.predict(vec)[0]}


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