from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import pipeline
from PIL import Image, ImageDraw
import io
from facenet_pytorch import MTCNN
import base64
import cv2
import tempfile
from fastapi.responses import FileResponse
import numpy as np
from PIL import ImageFont
import os


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
    

@app.post("/predict/video/")
async def predict_video(file: UploadFile = File(...), frame_skip: int = 5, request: Request = None):
    """
    Process a video frame by frame and detect emotions.
    - frame_skip: run detection every Nth frame (default=5) but keep boxes/labels visible on all frames.
    """

    try:
        # Save uploaded video temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(await file.read())
        temp_input.close()

        # Open video
        cap = cv2.VideoCapture(temp_input.name)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "Could not open video"})

        # Output video writer - Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*"H264")  # Changed from mp4v to H264
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

        frame_idx = 0
        last_predictions = []  # store last known predictions
        # Try loading a bundled font or fall back to default
        try:
            font = ImageFont.truetype("fonts/DejaVuSans.ttf", size=36)  # much bigger
        except:
            font = ImageFont.load_default()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)

            # Run detection every Nth frame
            if frame_idx % frame_skip == 0:
                boxes, _ = mtcnn.detect(image)
                if boxes is not None:
                    last_predictions = []
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        face = image.crop((x1, y1, x2, y2))
                        preds = pipe(face)
                        top_pred = preds[0]
                        last_predictions.append((box, top_pred))

           # Draw last known predictions (so boxes/labels persist)
            for box, top_pred in last_predictions:
                x1, y1, x2, y2 = map(int, box)
                label_text = f"{top_pred['label']} ({top_pred['score']:.2f})"

                # Measure text size (Pillow >= 10)
                bbox = font.getbbox(label_text)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                # Draw filled rectangle for background
                draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill="red")

                # Draw text on top of it
                draw.text((x1 + 3, y1 - text_h - 3), label_text, font=font, fill="white")

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

            # Convert back to OpenCV frame
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        file_path = temp_output.name
        file_size = os.path.getsize(file_path)

        range_header = request.headers.get("range")
        if range_header:
            # Parse "bytes=start-end"
            start, end = range_header.replace("bytes=", "").split("-")
            start = int(start)
            end = int(end) if end else file_size - 1
            chunk_size = (end - start) + 1

            def iterfile(path, start, end):
                with open(path, "rb") as f:
                    f.seek(start)
                    yield f.read(chunk_size)

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": "video/mp4",  # Added explicit content type
                "Cache-Control": "no-cache",  # Prevent caching issues
            }
            return StreamingResponse(iterfile(file_path, start, end),
                                    status_code=206,
                                    media_type="video/mp4",
                                    headers=headers)

        # fallback: full file with proper headers
        headers = {
            "Content-Type": "video/mp4",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        }
        return FileResponse(file_path, 
                          media_type="video/mp4", 
                          headers=headers,
                          filename="predicted_video.mp4")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})