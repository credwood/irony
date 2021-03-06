from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from typing import Optional, List
from pydantic import BaseModel
from inference import predict
import uvicorn
import json
import datetime


app = FastAPI()
#templates = Jinja2Templates(directory="templates/")

class Test(BaseModel):
    id: int
    text: List[str] = []
    result: Optional[List[str]] = None
    softmax: Optional[List[str]] = None 
    ground_truth: Optional[List[int]] = [5] 

@app.get("/")
def read_root():
    return RedirectResponse("http://backend:8084/test")

@app.post('/test/')
def handle_tests(test: Test):
    if test.ground_truth[0] != 5:
        with open(f"/user_tests/{test.id}{datetime.datetime.now()}_gt.json", 'w') as f:
            json.dump(test.json(), f)
        return {"message": test.ground_truth[0]}
    else:
        pred = predict(test.text)
        test.result = pred[1]
        test.softmax = pred[2]
        return test

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8084)
