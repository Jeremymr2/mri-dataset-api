from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
from PIL import Image
import io
import imghdr
from model import predict as model_predict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def process_image(image_bytes: bytes) -> BytesIO:
    image_stream = BytesIO(image_bytes)
    image_stream.seek(0)
    return image_stream

@app.get("/")
def home(request:Request):
    datos = {"health_check": "OK", "title":"Datamining"}
    return templates.TemplateResponse("index.html",{"request":request,"datos":datos})

@app.post("/predict/")
async def predict(request: Request,file: UploadFile):
    file_read = await file.read()
    file_type = imghdr.what(None, h=file_read)
    # Error - tipo de archivo invalido
    if file_type not in ['jpg', 'png', 'jpeg']:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    image = process_image(file_read)
    prediction = model_predict(image)

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})