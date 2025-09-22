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
    Uses FFmpeg for web-compatible output.
    """
    import subprocess

    try:
        # Save uploaded video temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(await file.read())
        temp_input.close()

        # Create temporary directory for frame processing
        temp_frames_dir = tempfile.mkdtemp()
        
        # Open video to get properties
        cap = cv2.VideoCapture(temp_input.name)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "Could not open video"})

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        frame_idx = 0
        last_predictions = []
        processed_frames = 0
        
        # Try loading a font or fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", size=36)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=36)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

        # Process frames and save as images
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
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(image.width, x2)
                        y2 = min(image.height, y2)
                        
                        if x2 > x1 and y2 > y1:
                            face = image.crop((x1, y1, x2, y2))
                            preds = pipe(face)
                            top_pred = preds[0]
                            last_predictions.append((box, top_pred))

            # Draw predictions
            for box, top_pred in last_predictions:
                x1, y1, x2, y2 = map(int, box)
                label_text = f"{top_pred['label']} ({top_pred['score']:.2f})"

                if font:
                    try:
                        bbox = font.getbbox(label_text)
                        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except AttributeError:
                        text_w, text_h = font.getsize(label_text)
                else:
                    text_w, text_h = len(label_text) * 10, 20

                # Draw background rectangle
                bg_x1 = max(0, x1)
                bg_y1 = max(0, y1 - text_h - 6)
                bg_x2 = min(image.width, x1 + text_w + 6)
                bg_y2 = max(0, y1)
                
                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="red")
                
                # Draw text
                text_x = max(3, x1 + 3)
                text_y = max(3, y1 - text_h - 3)
                draw.text((text_x, text_y), label_text, font=font, fill="white")
                
                # Draw bounding box
                box_x1 = max(0, x1)
                box_y1 = max(0, y1)
                box_x2 = min(image.width, x2)
                box_y2 = min(image.height, y2)
                draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline="red", width=4)

            # Save processed frame
            frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:06d}.jpg")
            image.save(frame_path, "JPEG", quality=95)
            
            frame_idx += 1
            processed_frames += 1
            
            if processed_frames % 30 == 0:
                print(f"Processed {processed_frames}/{total_frames} frames")

        cap.release()
        
        # Create web-compatible video using FFmpeg
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output_path = temp_output.name
        temp_output.close()

        # FFmpeg command for web-compatible MP4
        ffmpeg_cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", os.path.join(temp_frames_dir, "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # Enable web streaming
            "-y",  # Overwrite output
            temp_output_path
        ]

        print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        result = subprocess.run(
            ffmpeg_cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            return JSONResponse(status_code=500, content={
                "error": f"FFmpeg failed: {result.stderr}"
            })

        # Clean up frame files
        for frame_file in os.listdir(temp_frames_dir):
            os.unlink(os.path.join(temp_frames_dir, frame_file))
        os.rmdir(temp_frames_dir)
        
        # Clean up input file
        try:
            os.unlink(temp_input.name)
        except:
            pass

        # Verify output
        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            return JSONResponse(status_code=500, content={
                "error": "Output video file was not created or is empty"
            })

        file_size = os.path.getsize(temp_output_path)
        print(f"Web-compatible video created: {temp_output_path}, size: {file_size} bytes")

        # Handle range requests for video streaming
        range_header = request.headers.get("range") if request else None
        if range_header:
            try:
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
                    "Content-Type": "video/mp4",
                    "Cache-Control": "no-cache",
                }
                return StreamingResponse(
                    iterfile(temp_output_path, start, end),
                    status_code=206,
                    media_type="video/mp4",
                    headers=headers
                )
            except Exception as e:
                print(f"Range request failed: {e}")

        # Return full file
        headers = {
            "Content-Type": "video/mp4",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Content-Length": str(file_size),
        }
        
        return FileResponse(
            temp_output_path, 
            media_type="video/mp4", 
            headers=headers,
            filename="predicted_video.mp4"
        )

    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=500, content={
            "error": "Video processing timed out"
        })
    except Exception as e:
        print(f"Video processing error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})