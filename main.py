# main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn

app = FastAPI(title="Traffic Sign Detection API", version="1.0.0")

# CORS（Streamlitからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 必要に応じてドメインを絞ってください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルは起動時に一度だけロード
model = YOLO("best.pt")
class_names = model.names  # {0: 'stop_sign', ...}

class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int
    class_name_en: str

class PredictResponse(BaseModel):
    detections: list[Detection]
    image_w: int
    image_h: int
    num_detections: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect", response_model=PredictResponse)
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.4),
    imgsz: int = Form(640),
):
    # 画像を読み込む（np.ndarray BGR）
    file_bytes = await file.read()
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return PredictResponse(detections=[], image_w=0, image_h=0, num_detections=0)

    h, w = img.shape[:2]

    # 推論
    results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
    detections: list[Detection] = []

    if len(results) > 0:
        r = results[0]
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=conf_val,
                    class_id=cls_id,
                    class_name_en=class_names.get(cls_id, "unknown"),
                )
            )

    return PredictResponse(
        detections=detections,
        image_w=w,
        image_h=h,
        num_detections=len(detections),
    )

if __name__ == "__main__":
    # ローカル実行用: http://127.0.0.1:8000/docs
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
