import requests
from pathlib import Path
import os

from tqdm import tqdm

from app.helpers.integrity_checker import check_file_integrity

def download_file(model_name: str, file_path: str, correct_hash: str, url: str) -> bool:
    """
    Downloads a file and verifies its integrity.

    Parameters:
    - model_name (str): Name of the model being downloaded.
    - file_path (str): Path where the file will be saved.
    - correct_hash (str): Expected hash value of the file for integrity check.
    - url (str): URL to download the file from.

    Returns:
    - bool: True if the file is downloaded and verified successfully, False otherwise.
    """
    # Remove the file if it already exists and restart download
    if Path(file_path).is_file():
        if check_file_integrity(file_path, correct_hash):
            #print(f"\nSkipping {model_name} as it is already downloaded!")
            return True
        else:  
            print(f"\n{file_path} already exists, but its file integrity couldn't be verified. Re-downloading it!")
            os.remove(file_path)

    print(f"\nDownloading {model_name} from {url}")
    
    # Add a User-Agent header to mimic a browser request, which is often required by services like Hugging Face
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, stream=True, timeout=10, headers=headers) # Added headers and increased timeout
        response.raise_for_status()  # Raise an error for bad HTTP responses (e.g., 404, 500)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {model_name}: {e}")
        return False

    total_size = int(response.headers.get("content-length", 0))  # File size in bytes
    block_size = 1024  # Size of chunks to download
    max_attempts = 3
    attempt = 1

    def download_and_save():
        """Handles the file download and saves it to disk."""
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=model_name) as progress_bar:
            with open(file_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
    
    while attempt <= max_attempts:
        try:
            download_and_save()
            
            # Verify file integrity
            if check_file_integrity(file_path, correct_hash):
                print(f"\nFile integrity verified successfully for {model_name}!")
                print(f"File saved at: {file_path}")
                return True
            else:
                print(f"\nIntegrity check failed for {file_path}. Retrying download (Attempt {attempt}/{max_attempts})...")
                os.remove(file_path)
                # Re-establish the request for retry
                response = requests.get(url, stream=True, timeout=10, headers=headers)
                response.raise_for_status()
                attempt += 1
        except requests.exceptions.Timeout:
            print("\nConnection timed out! Retrying download...")
            attempt += 1
            if attempt <= max_attempts:
                response = requests.get(url, stream=True, timeout=10, headers=headers)
                response.raise_for_status()
        except Exception as e:
            print(f"\nAn error occurred during download: {e}")
            attempt += 1
            if attempt <= max_attempts:
                response = requests.get(url, stream=True, timeout=10, headers=headers)
                response.raise_for_status()


    print(f"Failed to download {model_name} after {max_attempts} attempts.")
    if os.path.exists(file_path):
        os.remove(file_path) # Clean up partial file
    return False