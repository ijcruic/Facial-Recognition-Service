import os
from pathlib import Path
import requests
import logging

# Configuration
API_URL = "http://localhost:80/facial_recognition"
QUERY_IMAGES_DIR = "./query-images"
LOG_FILE = "recognition_test.log"

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def recognize_face(file_path: Path):
    """
    Send a face image to the facial_recognition API endpoint and get the result.

    Args:
        file_path (Path): Path to the image file.

    Returns:
        dict: The API response if successful, None otherwise.
    """
    try:
        with file_path.open("rb") as img_file:
            files = {"file": img_file}
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
            logging.info(f"Processed {file_path.name}: {response.json()}")
            return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error processing {file_path.name}: {response.json()}")
        return None

def main():
    # Process all image files in the query-images directory
    query_images_dir = Path(QUERY_IMAGES_DIR)
    
    if not query_images_dir.exists():
        logging.error(f"Directory {QUERY_IMAGES_DIR} does not exist!")
        return

    image_files = [file for file in query_images_dir.iterdir() if file.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    
    if not image_files:
        logging.warning(f"No images found in {QUERY_IMAGES_DIR}")
        return

    for file_path in image_files:
        logging.info(f"Processing image: {file_path}")
        
        # Send image for recognition
        result = recognize_face(file_path)

    logging.info("Recognition process complete.")

if __name__ == "__main__":
    main()
