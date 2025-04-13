from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fast_alpr import ALPR
import numpy as np
from PIL import Image, ImageFilter
import re
import os
import io

app = FastAPI()

# Initialize ALPR with specified models and force CPU execution provider
detector_model = "yolo-v9-s-608-license-plate-end2end"
ocr_model = "global-plates-mobile-vit-v2-model"
alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model, detector_providers=[
            "CPUExecutionProvider"], ocr_providers=["CPUExecutionProvider"])


@app.post("/alpr")
async def upload_file(file: UploadFile = File(...)):
    # Convert uploaded file to a format compatible with OpenCV
    image = Image.open(file.file)
    img_array = np.array(image.convert("RGB"))

    # Run ALPR on the uploaded image
    results = alpr.predict(img_array)

    # Prepare response
    response = []
    if results:
        for result in results:
            plate_text = result.ocr.text if result.ocr else "N/A"
            plate_confidence = result.ocr.confidence if result.ocr else 0.0
            # Add space between letters and numbers
            formatted_plate_text = re.sub(
                r'([A-Z]+)([0-9]+)', r'\1 \2', plate_text)
            response.append({
                "detected_plate": formatted_plate_text,
                "confidence": plate_confidence
            })
    else:
        response.append({"message": "No license plate detected."})

    return {"results": response}


@app.post("/blur/images")
async def blur_image(file: UploadFile = File(...)):
    try:
        # Convert uploaded file to a format compatible with OpenCV
        image = Image.open(file.file)
        img_array = np.array(image.convert("RGB"))

        # Run ALPR on the uploaded image to detect license plates
        results = alpr.predict(img_array)

        # Apply Gaussian blur to the detected license plate regions
        for result in results:
            if hasattr(result, 'detection') and result.detection.bounding_box:
                x_min, y_min, x_max, y_max = result.detection.bounding_box.x1, result.detection.bounding_box.y1, result.detection.bounding_box.x2, result.detection.bounding_box.y2
                plate_region = image.crop((x_min, y_min, x_max, y_max))
                plate_region = plate_region.filter(ImageFilter.GaussianBlur(radius=40)) 
                image.paste(plate_region, (x_min, y_min))
                

        # Convert the blurred image to a byte stream
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='JPEG')
        byte_arr.seek(0)

        return StreamingResponse(byte_arr, media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.100.161", port=8000)