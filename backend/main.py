from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from typing import Optional, List
from pydantic import BaseModel
from inference import predict
import uvicorn
import json
import pickle
import time


app = FastAPI()
#templates = Jinja2Templates(directory="templates/")

class Test(BaseModel):
    id: int
    text: List[str] = []
    result: Optional[List[str]] = None
    softmax: Optional[List[str]] = None 
    ground_truth: Optional[List[bool]] = None 

@app.get("/")
def read_root():
    return {"message": "is this a joke"}

@app.post('/test/')
def recognize_enteties(test: Test):
    if test.ground_truth is not None:
        with open(f"{test.id}{time.datetime}_gt.json") as f:
            json.dumps(test, f)
    else:
        pred = predict(test.text, json_file_out=f'{test.id}{time.datetime}.json')
        test.result = pred[1]
        test.softmax = pred[2]
        return {"message": test}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)