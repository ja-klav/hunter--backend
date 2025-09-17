from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import pipeline
from PIL import Image, ImageDraw
import io
from facenet_pytorch import MTCNN
import base64


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Face detector
mtcnn = MTCNN(keep_all=False)

mtcnn_video = MTCNN(keep_all = True)

# Load your model pipeline
pipe = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    use_fast=True
)

@app.get('/')
def read_root():
    return {"message": "Hello, FastAPI"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "query": q}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        box, _ = mtcnn.detect(image)

        if box is None:
            return JSONResponse(content={"error": "No face detected"})

        x1, y1, x2, y2 = map(int, box[0])
        face = image.crop((x1, y1, x2, y2))

        preds = pipe(face)
        top_pred = preds[0]

        # Draw result
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{top_pred['label']} ({top_pred['score']:.2f})", fill="red")

        # Encode image as base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "label": top_pred['label'],
            "score": float(top_pred['score']),
            "all_predictions": preds,
            "image_base64": img_str
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})