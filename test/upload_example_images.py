import os
from pathlib import Path
import requests
import logging

# Configuration
API_URL = "http://localhost:80/upload_face_image"
DATA_DIR = "../data/lfw-deepfunneled"
LOG_FILE = "upload_test.log"

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def upload_image(file_path: Path):
    """
    Upload a face image and associated metadata to the API.

    Args:
        file_path (Path): Path to the image file.
        name (str): Name of the person in the image.

    Returns:
        dict: The API response if successful, None otherwise.
    """
    try:
        with file_path.open("rb") as img_file:
            files = {"file": img_file}
            data = {"s3_key": str(file_path), "plexus_id":None, "miscellaneous":None}
            response = requests.post(API_URL, files=files, data=data)
            response.raise_for_status()
            logging.info(f"Uploaded {file_path.name}: {response.json()}")
            return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error uploading {file_path.name}: {response.json()}")
        return None


def main():
    success_count = 0
    failure_count = 0

    # Process subdirectories
    data_dir = Path(DATA_DIR)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()][0:10]
    
    for subdir in subdirs:
        name = subdir.name
        logging.info(f"Processing directory: {subdir}")
        
        # Iterate through image files in the subdirectory
        for file_path in subdir.iterdir():
            if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            
            logging.info(f"Uploading {file_path} ...")
            result = upload_image(file_path)
            if result:
                success_count += 1
            else:
                failure_count += 1

    # Summary of results
    logging.info(f"Upload complete. Success: {success_count}, Failures: {failure_count}")
    print(f"Upload complete. Success: {success_count}, Failures: {failure_count}")


if __name__ == "__main__":
    main()