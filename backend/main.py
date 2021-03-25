from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from typing import Optional, List
from pydantic import BaseModel
from inference import predict
import uvicorn
import json
import pickle
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
    return RedirectResponse("http://backend:8080/test")

@app.post('/test/')
def recognize_enteties(test: Test):
    if test.ground_truth[0] != 5:
        with open(f"/user_tests/{test.id}{datetime.datetime.now()}_gt.json") as f:
            json.dumps(test, f)
    else:
        pred = predict(test.text, json_file_out=f'/user_tests/{test.id}{datetime.datetime.now()}.json')
        test.result = pred[1]
        test.softmax = pred[2]
        return {"message": test}

@app.post('/about/')
def about():
    html_content = """
    <html>
        <head>
            <title>about</title>
        </head>
        <body>
            <p>Some jokes are defined by their context, some jokes are defined by their form. Many are a mix of the two.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
