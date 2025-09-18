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
mtcnn = MTCNN(keep_all=True)

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

        boxes, _ = mtcnn.detect(image)

        if boxes is None:
            return JSONResponse(content={"error": "No face detected"})

        predictions = []
        draw = ImageDraw.Draw(image)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            face = image.crop((x1, y1, x2, y2))

            # Run classification
            preds = pipe(face)
            top_pred = preds[0]

            # Encode cropped face
            buf_face = io.BytesIO()
            face.save(buf_face, format="PNG")
            face_str = base64.b64encode(buf_face.getvalue()).decode("utf-8")

            predictions.append({
                "face_id": i,
                "label": top_pred['label'],
                "score": float(top_pred['score']),
                "all_predictions": preds,
                "face_image_base64": face_str
            })

            # Draw on the full image
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{top_pred['label']} ({top_pred['score']:.2f})", fill="red")

        # Encode annotated full image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "faces": predictions,
            "image_base64": img_str
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})