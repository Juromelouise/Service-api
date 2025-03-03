# FastAPI OCR Project

This project implements an automatic license plate recognition (ALPR) system using FastAPI. It allows users to upload images of vehicles with license plates, which are then processed using specified OCR and Detector models.

## Project Structure

```
fastapi-ocr-project
├── src
│   ├── main.py          # Entry point of the FastAPI application
│   └── models
│       └── __init__.py  # Placeholder for future model definitions
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Requirements

To run this project, you need to install the following dependencies:

- FastAPI
- Uvicorn
- OpenCV
- NumPy
- Pillow
- fast-alpr

You can install the required packages using pip:

```
pip install -r requirements.txt
```

## Running the Application

1. Navigate to the project directory:

   ```
   cd fastapi-ocr-project
   ```

2. Start the FastAPI application using Uvicorn:

   ```
   uvicorn src.main:app --reload
   ```

3. The application will be running at `http://127.0.0.1:8000`.

## API Endpoint

- **POST /upload**: Upload an image of a vehicle with a license plate. The image should be sent as form data.

### Example Request

You can use Insomnia or any other API client to test the endpoint. Set up a POST request to `http://127.0.0.1:8000/upload` with the image file included in the form data.

## License

This project is licensed under the MIT License.