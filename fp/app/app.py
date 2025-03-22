from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

import io
import logging
import time
import argparse
import uuid
import tempfile
import shutil
import os
import subprocess
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = None
conf_threshold = None
iou_threshold = None

active_videos = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("person-detector")


parser = argparse.ArgumentParser(description='Person Detection Service')
parser.add_argument('--weights', type=str, default='best_weights.pt',
                    help='Path to model weights file (default: best_weights.pt)')
parser.add_argument('--conf', type=float, default=0.25,
                    help='Confidence threshold (default: 0.25)')
parser.add_argument('--iou', type=float, default=0.45,
                    help='IoU threshold for NMS (default: 0.45)')
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help='Host to run server on (default: 0.0.0.0)')
parser.add_argument('--port', type=int, default=8000,
                    help='Port to run server on (default: 8000)')
parser.add_argument('--segment-duration', type=int, default=4,
                    help='Duration of HLS segments in seconds (default: 4)')


app = FastAPI(
    title="Person Detection Service",
    description="API for detecting people in images and videos using YOLO",
    version="1.0.0"
)

os.makedirs("static", exist_ok=True)
os.makedirs("static/videos", exist_ok=True)
os.makedirs("static/hls", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def load_model(weights_path: str, conf: float, iou: float) -> YOLO:
    """Load YOLO model with specified parameters"""
    global model, conf_threshold, iou_threshold
    try:
        conf_threshold = conf
        iou_threshold = iou

        model = YOLO(weights_path)

        logger.info(f"YOLO model successfully loaded from {weights_path}")
        logger.info(f"Parameters: conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def process_image(image_data: bytes) -> Tuple[Image.Image, List]:
    """Process image data and detect people"""
    image = Image.open(io.BytesIO(image_data))
    results = model(image, classes=[0], conf=conf_threshold, iou=iou_threshold)
    return image, results


def add_count_overlay(image: np.ndarray, count: int) -> np.ndarray:
    """Add people count overlay to the image"""

    overlay = image.copy()
    text_bg_height = 40
    cv2.rectangle(overlay, (10, 10), (300, 10 + text_bg_height), (0, 0, 0), -1)
    alpha = 0.6 
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"People detected: {count}"
    cv2.putText(image, text, (15, 35), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def prepare_output_image(result_image: np.ndarray, people_count: int) -> bytes:
    # WARNING: BGR format expected
    result_image = add_count_overlay(result_image, people_count)

    # Convert BGR to RGB for PIL
    if len(result_image.shape) == 3 and result_image.shape[2] == 3:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    buf = io.BytesIO()
    img_result = Image.fromarray(result_image)
    img_result.save(buf, format="JPEG")
    buf.seek(0)

    return buf


def extract_person_data(boxes) -> List[Dict]:
    people_detected = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()

        people_detected.append({
            "confidence": float(confidence),
            "bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            },
            "center": {
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2)
            }
        })

    return people_detected


def process_video_for_hls(input_path: str, hls_dir: str, video_id: str, segment_duration: int, status_dict: Dict):
    try:
        output_dir = os.path.join(hls_dir, video_id)
        os.makedirs(output_dir, exist_ok=True)

        playlist_path = os.path.join(output_dir, "playlist.m3u8")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            status_dict["status"] = "error"
            status_dict["message"] = f"Could not open video file: {input_path}"
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # downgrade result fps
        TARGET_FPS = 5
        frame_skip = max(1, int(original_fps / TARGET_FPS))
        frames_per_segment = int(TARGET_FPS * segment_duration)
        expected_frames = total_frames // frame_skip

        # needed for YUV420
        width = width - (width % 2)
        height = height - (height % 2)

        # Start tracking progress
        status_dict["total_frames"] = expected_frames
        status_dict["processed_frames"] = 0
        status_dict["total_segments"] = (expected_frames + frames_per_segment - 1) // frames_per_segment
        status_dict["processed_segments"] = 0

        with open(playlist_path, 'w') as playlist:
            playlist.write("#EXTM3U\n")
            playlist.write("#EXT-X-VERSION:3\n")
            playlist.write(f"#EXT-X-TARGETDURATION:{segment_duration}\n")
            playlist.write("#EXT-X-PLAYLIST-TYPE:EVENT\n")
            playlist.write("#EXT-X-MEDIA-SEQUENCE:0\n")

        # Process video in segments
        segment_count = 0
        frame_counter = 0
        processed_frames = 0
        current_segment_frames = []

        status_dict["status"] = "processing"
        status_dict["start_time"] = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % frame_skip != 0:
                frame_counter += 1
                continue

            results = model(frame, classes=[0], conf=conf_threshold, iou=iou_threshold)

            result_frame = results[0].plot()
            result_frame = add_count_overlay(result_frame, len(results[0].boxes))
            result_frame = cv2.resize(result_frame, (width, height))

            current_segment_frames.append(result_frame)
            processed_frames += 1
            status_dict["processed_frames"] = processed_frames

            # Create segment when enough frames
            if len(current_segment_frames) >= frames_per_segment:
                segment_filename = f"segment_{segment_count}.ts"
                segment_path = os.path.join(output_dir, segment_filename)

                command = [
                    'ffmpeg',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{width}x{height}',
                    '-pix_fmt', 'bgr24',
                    '-r', str(TARGET_FPS),
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-f', 'mpegts',
                    segment_path
                ]

                process = subprocess.Popen(command, stdin=subprocess.PIPE)
                for f in current_segment_frames:
                    process.stdin.write(f.tobytes())
                process.stdin.close()
                process.wait()

                # Update playlist
                segment_duration_actual = len(current_segment_frames) / TARGET_FPS
                with open(playlist_path, 'a') as playlist:
                    playlist.write(f"#EXTINF:{segment_duration_actual:.6f},\n")
                    playlist.write(f"{segment_filename}\n")

                segment_count += 1
                status_dict["processed_segments"] = segment_count
                current_segment_frames = []
            frame_counter += 1

        with open(playlist_path, 'a') as playlist:
            playlist.write("#EXT-X-ENDLIST\n")

        status_dict["status"] = "completed"
        status_dict["end_time"] = time.time()
        status_dict["processing_time"] = status_dict["end_time"] - status_dict["start_time"]

        cap.release()
        logger.info(f"HLS video processing completed. Created {segment_count} segments.")

    except Exception as e:
        logger.error(f"Error in HLS processing: {str(e)}")
        status_dict["status"] = "error"
        status_dict["message"] = f"Processing error: {str(e)}"


@app.on_event("startup")
async def startup_event():
    args = app.state.args
    load_model(args.weights, args.conf, args.iou)


@app.get("/")
async def root():
    return {"message": "Person Detection Service is running. Go to /ui for web interface or /docs for API documentation."}


@app.get("/hls/{video_id}/playlist.m3u8", response_class=FileResponse)
async def get_playlist(video_id: str):
    return FileResponse(
        f"static/static/hls/{video_id}/playlist.m3u8",
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/hls/{video_id}/{segment}", response_class=FileResponse)
async def get_segment(video_id: str, segment: str):
    return FileResponse(
        f"static/hls/{video_id}/{segment}",
        media_type="video/MP2T",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/ui")
async def web_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect_people/image")
async def detect_people_image(file: UploadFile = File(...)):
    """
    Detect people and display results on an image
    
    - **file**: Uploaded image
    
    Returns:
        Image with marked detected people
    """
    start_time = time.time()

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()

        _, results = process_image(contents)

        result_image = results[0].plot()
        people_count = len(results[0].boxes)

        output_buffer = prepare_output_image(result_image, people_count)

        processing_time = time.time() - start_time
        logger.info(f"Image processing completed in {processing_time:.2f} seconds. Found {people_count} people.")

        return StreamingResponse(output_buffer, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/detect_people/json")
async def detect_people_json(file: UploadFile = File(...)):
    """
    Detect people and return data in JSON format
    
    - **file**: Uploaded image
    
    Returns:
        JSON with information about detected people (count, positions, bounding boxes)
    """
    start_time = time.time()

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()

        image, results = process_image(contents)
        img_width, img_height = image.size

        people_detected = extract_person_data(results[0].boxes)

        result = {
            "image_info": {
                "width": img_width,
                "height": img_height,
            },
            "people_count": len(people_detected),
            "people": people_detected,
            "processing_time_sec": time.time() - start_time
        }

        processing_time = time.time() - start_time
        logger.info(f"Image processing and JSON creation completed in {processing_time:.2f} seconds")

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/detect_people/video-stream")
async def detect_people_video_streaming(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Process video with people detection and stream it using HLS
    
    - **file**: Uploaded video file
    
    Returns:
        JSON with information about the stream
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        video_id = str(uuid.uuid4())
        input_filename = f"input_{video_id}.mp4"

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        input_path = f"static/videos/{input_filename}"

        os.makedirs("static/videos", exist_ok=True)
        shutil.copy(temp_file_path, input_path)
        os.unlink(temp_file_path)

        status_dict = {
            "video_id": video_id,
            "status": "initializing",
            "message": "Starting video processing",
            "progress": 0
        }

        active_videos[video_id] = status_dict

        args = app.state.args

        background_tasks.add_task(
            process_video_for_hls,
            input_path,
            "static/hls",
            video_id,
            args.segment_duration,
            status_dict
        )

        return JSONResponse({
            "video_id": video_id,
            "status": "initializing",
            "message": "Video processing started",
            "stream_url": f"/static/hls/{video_id}/playlist.m3u8",
            "status_url": f"/detect_people/video-status/{video_id}"
        })

    except Exception as e:
        logger.error(f"Error starting video streaming: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")


@app.get("/detect_people/video-status/{video_id}")
async def video_status(video_id: str):
    """
    Get the status of video processing
    
    - **video_id**: ID of the video being processed
    
    Returns:
        JSON with processing status
    """
    if video_id not in active_videos:
        raise HTTPException(status_code=404, detail="Video not found")

    status = active_videos[video_id]

    if "total_frames" in status and status["total_frames"] > 0:
        processed = status.get("processed_frames", 0)
        total = status["total_frames"]
        progress = (processed / total) * 100
        status["progress"] = round(progress, 2)

    return JSONResponse(status)


if __name__ == "__main__":
    import uvicorn

    args = parser.parse_args()
    app.state.args = args

    logger.info(f"Starting server with parameters:")
    logger.info(f"  - weights: {args.weights}")
    logger.info(f"  - conf: {args.conf}")
    logger.info(f"  - iou: {args.iou}")
    logger.info(f"  - host: {args.host}")
    logger.info(f"  - port: {args.port}")
    logger.info(f"  - segment_duration: {args.segment_duration}")

    uvicorn.run(app, host=args.host, port=args.port)
