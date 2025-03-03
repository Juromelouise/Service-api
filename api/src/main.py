from fastapi import FastAPI, File, UploadFile
from fast_alpr import ALPR
import numpy as np
from PIL import Image
import re

app = FastAPI()

# Initialize ALPR with specified models and force CPU execution provider
detector_model = "yolo-v9-s-608-license-plate-end2end"
ocr_model = "global-plates-mobile-vit-v2-model"
alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model, detector_providers=["CPUExecutionProvider"], ocr_providers=["CPUExecutionProvider"])

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
            formatted_plate_text = re.sub(r'([A-Z]+)([0-9]+)', r'\1 \2', plate_text)
            response.append({
                "detected_plate": formatted_plate_text,
                "confidence": plate_confidence
            })
    else:
        response.append({"message": "No license plate detected."})

    return {"results": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)