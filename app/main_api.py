from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pathlib import Path
import yaml
from io import BytesIO
import hashlib
from PIL import Image, UnidentifiedImageError
import datetime

from app.recognition import FaceDetector
from app.representation import InceptionResnetV1

from opensearchpy import OpenSearch

from app.log_helper import matomo_track_event


config_file_path = Path("app/config.yaml")
if not config_file_path.exists():
    raise FileNotFoundError(f"Config file not found at: {config_file_path}")

with open(config_file_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


@asynccontextmanager
async def lifespan(app: FastAPI):
    matomo_track_event("unknown user", "service startup", app.title, "success", 1)
    
    app.state.embedding_model = InceptionResnetV1(
        weights_filename=config["embedding_model_weights_path"], device=config["embedding_model_device"]
    ).eval()

    app.state.recognition_model = FaceDetector(
        keep_all=True, model_path=config["recognition_model_weights_path"], device=config["recognition_model_device"]
    ).eval()

    app.state.index_recognition_model = FaceDetector(
        keep_all=False, model_path=config["recognition_model_weights_path"], device=config["recognition_model_device"], margin=32
    ).eval()


    app.state.os_client = OpenSearch(
        hosts=[config["cluster_url"]],
        http_auth=(config["username"], config["password"]),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False
    )

    # Instantiate the face vector index (for embeddings)
    index_name = config["face_vector_index_name"]
    app.state.db_index = index_name

    # Create the index with a mapping for dense vectors if it does not already exist.
    if not app.state.os_client.indices.exists(index=index_name):
        mapping = {
            "settings": {
                "index": {
                    "knn": True 
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 512,
                        "store": True
                    },
                    "s3_key": {"type": "keyword"},
                    "plexus_id": {"type": "keyword"},
                    "miscellaneous": {"type": "keyword"}
                }
            }

        }
        app.state.os_client.indices.create(index=index_name, body=mapping)

    # Instantiate the telemrty database for model and data monitoring
    telemetry_index = config["telemetry_index_name"]
    app.state.telemetry_index = telemetry_index
    if not app.state.os_client.indices.exists(index=telemetry_index):
        telemetry_mapping = {
            "mappings": {
                "properties": {
                    "query_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "distances": {"type": "float"}
                }
            }
        }
        app.state.os_client.indices.create(index=telemetry_index, body=telemetry_mapping)
    
    yield
    
    matomo_track_event("unknown user", "service shutdown", app.title, "success", 1)


app = FastAPI(title="Facial Recognition Service", lifespan=lifespan, version="0.1.0")


@app.get(
    "/", response_class=HTMLResponse, summary="Provides the documentation, right here!"
)
async def index():
    matomo_track_event("unknown user", "page visit", "README html", "success", 1)
    return FileResponse("README.html")


@app.get("/favicon.ico")
async def favicon():
    matomo_track_event("unknown user", "page visit", "favicon ico", "success", 1)
    return FileResponse(
        path="favicon.ico",
        headers={"Content-Disposition": "attachment; filename=favicon.ico"},
    )


@app.post("/facial_recognition", summary="Get Candidates for Any Faces in an Image")
async def facial_recognition(file: UploadFile = File(...)):
    """
    Detects faces in an uploaded image, generates embeddings, and queries OpenSearch
    for the nearest neighbors. Additionally, telemetry data (average similarities and timestamp)
    is stored to help monitor data drift.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        List[Dict]: A list of embeddings and possible matches for each detected face.
    """
    matomo_track_event(
        user="unknown user",
        category="api interaction",
        action="upload image for recognition",
        name="facial_recognition",
        value=1
    )

    try:
        # Validate and read media
        image_rgb = await read_and_validate_media(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading and validating image: {str(e)}")

    try:
        recognition_model = app.state.recognition_model
        embedding_model = app.state.embedding_model

        # Detect faces and return tensors of cropped faces
        face_tensors = recognition_model(image_rgb)  # Returns [num_faces, c1, c2, c3]
        face_boxes, _ = recognition_model.detect(image_rgb)
        if face_tensors is None or face_tensors.size(0) == 0:
            return {"message": "No faces detected in the uploaded image.",
                    "results": []
                    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recognizing faces: {str(e)}")

    try:
        # Generate embeddings for all detected faces at once
        embeddings = embedding_model(face_tensors).detach().cpu().numpy()  # [num_faces, 512]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding faces: {str(e)}")

    try:
        os_client = app.state.os_client
        index_name = app.state.db_index

        # Parse results for each face box
        results = []
        for idx, face_box in enumerate(face_boxes):
            query_vector = embeddings[idx].tolist()
            query_body = {
                "size": 3, 
                "query": {
                    "knn": {"embedding": {"vector": query_vector, "k": 3}}
                },
                "_source": ["s3_key", "plexus_id", "miscellaneous", "embedding"],
            }
            
            response = os_client.search(index=index_name, body=query_body)
            neighbors = []
            for hit in response["hits"]["hits"]:
                src = hit["_source"]
                neighbors.append({
                    "s3_key":      src.get("s3_key"),
                    "plexus_id":   src.get("plexus_id"),
                    "miscellaneous": src.get("miscellaneous"),
                    "similarity":  hit["_score"],
                    # only available if you added `"store": true` to your embedding mapping
                    "embedding":   src.get("embedding"),
                })
            results.append({"person_bbox": face_box.tolist(), 
                            "person_image_embedding": query_vector,
                            "neighbors": neighbors})
        
        #Store telemetry data for model monitoring
        telemetry_index = app.state.telemetry_index
        # For each face, compute the max similarity from the returned neighbors.
        all_face_max_similarities = []
        for result in results:
            neighbor_similarities = [n["similarity"] for n in result["neighbors"] if n["similarity"] is not None]
            if neighbor_similarities:
                max_similarities = max(neighbor_similarities)
                all_face_max_similarities.append(max_similarities)

        # Create a unique query ID.
        query_id =  hashlib.sha256(datetime.datetime.now(datetime.timezone.utc).isoformat().encode("utf-8")).hexdigest()
        telemetry_doc = {
            "query_id": query_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "distances": all_face_max_similarities
        }
        os_client.index(index=telemetry_index, id=query_id, body=telemetry_doc, refresh=True)
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying database or processing query results: {str(e)}" +f" query results: {results}")

    return {"results": results}
    

@app.post("/upload_face_image", summary="Upload a face image with metadata for storage")
async def upload_face_image(
    file: UploadFile = File(...), s3_key: str = Form(...), plexus_id: Optional[str] = Form(None), miscellaneous: Optional[str] = Form(None)
):
    """
    Upload a face image, detect and embed the face, and store the embedding with metadata.

    Args:
        file (UploadFile): The uploaded image file.
        s3_key (str): The S3 key associated with the face image.
        plexus_id (Optional[str]): The optional Plexus ID.
        miscellaneous (Optional[str]): Any miscellaneous information on the image, like its optic.

    Returns:
        dict: A success message with the stored metadata.
    """
    matomo_track_event(
        user="unknown user",
        category="api interaction",
        action="upload face image",
        name="upload_face_image",
        value=1
    )

    warnings = []

    try:
        # Validate and read the uploaded image
        image_rgb = await read_and_validate_media(file)
        recognition_model = app.state.index_recognition_model
        embedding_model = app.state.embedding_model
        os_client = app.state.os_client
        index_name = app.state.db_index

        # Detect faces and extract tensors
        try:
            face_tensors = recognition_model(image_rgb)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error with face recognition model: {str(e)}")

        # Handle issues where there is no face or multiple faces in the image
        if face_tensors is None or len(face_tensors) == 0:
            raise HTTPException(status_code=400, detail="No faces detected in the uploaded image.")

        # Generate embeddings for detected face. Note - only works with single face coming back
        embedding = embedding_model.forward(face_tensors).detach().cpu().numpy().flatten()
        
        try:
            id = hashlib.sha256(s3_key.encode()).hexdigest() # Generate a unique ID using a hash of the file path
            doc = {
                "embedding": embedding.tolist(),
                "s3_key": s3_key,
                "plexus_id": plexus_id,
                "miscellaneous": miscellaneous
            }
            # Index the document into OpenSearch.
            os_client.index(index=index_name, id=id, body=doc, refresh=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing data in OpenSearch: {str(e)}")

        # Return success response with optional warnings
        response = {
            "message": "Face and metadata successfully stored.",
            "metadata": {"s3_key": s3_key, "plexus_id": plexus_id},
        }
        if warnings:
            response["warnings"] = warnings

        return response

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def read_and_validate_media(file: UploadFile) -> Image.Image:
    """
    Reads and validates the uploaded file as an image.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        Image.Image: A PIL Image object in RGB format.

    Raises:
        HTTPException: If the file is invalid or cannot be processed as an image.
    """
    try:
        file_data = await file.read()
        return Image.open(BytesIO(file_data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
