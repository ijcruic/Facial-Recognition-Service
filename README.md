# Facial Recognition API

This FastAPI service provides facial recognition capabilities using deep learning models. It allows users to upload images, detect faces, generate embeddings, and search for similar faces using OpenSearch.

## Features
- Detect faces in uploaded images
- Generate facial embeddings using deep learning models
- Query a database for similar faces
- Upload new face images for storage and indexing
- Containerized with Docker for easy deployment

## Running the Service

To run the service using Docker, use the following commands:

```sh
docker build -t facial_recognition_api .
docker run -p 8000:8000 facial_recognition_api
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### 1. Root Endpoint
**GET /**

Provides the API documentation as an HTML response.

### 2. Facial Recognition
**POST /facial_recognition**

Detects faces in an uploaded image and returns possible matches.

#### Request
- **file**: Image file (e.g., `.jpg`, `.png`).

#### Example (Python):
```python
import requests

url = "http://localhost:8000/facial_recognition"
image_path = "path/to/image.jpg"

with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)
    print(response.json())
```

#### Response
```json
{
  "results": [
    {
      "person_bbox": [x1, y1, x2, y2],
      "person_image_embedding": [0.12, -0.07, ..., 0.33],
      "neighbors": [
        {
          "similarity": 0.23,
          "s3_key": "/images/johndoe.jpg",
          "plexus_id": "askjvvn4378qtq8ahcv",
          "miscellaneous": "tagged as VIP"
        }
      ]
    }
  ]
}
```

If "results" is empty, look at the "message" in the return object.

### 3. Upload Face Image
**POST /upload_face_image**

Uploads a face image with metadata for indexing.

#### Request
- **file**: Image file (e.g., `.jpg`, `.png`).
- **s3_key**: Path or key where the image is stored (e.g., /images/johndoe.jpg)
- **plexus_id**: Optional identifier for the person
- **miscellaneous**: Optional metadata or tags

#### Example (Python):
```python
import requests

url = "http://localhost:8000/upload_face_image"
image_path = "path/to/image.jpg"

data = {
    "s3_key": "/images/johndoe.jpg",
    "plexus_id": "askjvvn4378qtq8ahcv",
    "miscellaneous": "tagged as VIP"
}
files = {"file": open(image_path, "rb")}
response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Response
```json
{
  "message": "Face image successfully uploaded and indexed."
}
```

## Configuration

The service reads configurations from `app/config.yaml`, including model paths and database credentials.

## Deployment

To deploy the service, ensure OpenSearch is running and properly configured. Update `config.yaml` with the correct credentials and URLs.

## License
TBD

## Developer Notes -- Data Links
Places UC Datasets pulled from:
1. CASIA: https://www.kaggle.com/datasets/kenny3s/casia-webface
2. Celeb A: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
3. CFP-dataset: https://www.kaggle.com/datasets/chinafax/cfpw-dataset
4. I think I used this one for VGGFace2: https://huggingface.co/datasets/logasja/VGGFace2, or possibly this one: https://www.kaggle.com/datasets/hearfool/vggface2
5. LFW dataset: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

Keeping in mind UC data is skewed towards Western Facial Patterns 
This makes applicability difficult to non-caucasian ethnicities
