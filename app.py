from fastapi import FastAPI
import uvicorn
import os
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textsummarization.pipeline.prediction import PredictionPipeline

text: str = "what is text summarization"

app = FastAPI()
@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training Seccessful !!!")
    except Exception as e:
        print(e)
    return Response(f"Error Occurred! {str(e)}")
@app.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e
        print(e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)