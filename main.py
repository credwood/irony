from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from typing import Optional, List
from pydantic import BaseModel
from inference import predict
import json
import pickle


app = FastAPI()
#templates = Jinja2Templates(directory="templates/")

class Test(BaseModel):
    id: int
    text: List[str] = []
    result: Optional[List[str]] = None
    softmax: Optional[List[str]] = None 

@app.post('/test/')
def recognize_enteties(test: Test):
    pred = predict(test.text, json_file_out=f'{test.id}.json')
    test.result = pred[1]
    test.softmax = pred[2]
    return {"message": f"{test.id}, {test.text}, {test.result}, {test.softmax}"}
