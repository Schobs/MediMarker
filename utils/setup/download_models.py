import os
import gdown
import logging


def download_model_from_google_drive(google_drive_model_path, google_drive_save_local_path):
    logger = logging.getLogger()

    # Extract Google Drive file id from the path

    file_id = google_drive_model_path.split('/')[-2]
    # model_file_name = google_drive_save_local_path.split('/')[-1]
    dl_directory = "/".join(google_drive_save_local_path.split('/')[:-1])

    # Prepare the local file path
    # local_file_path = os.path.join(google_drive_save_local_path, dataset_name, model_type)
    # local_file_path = google_drive_save_local_path

    # Make sure the directory exists
    os.makedirs(dl_directory, exist_ok=True)

    # Prepare the full local file path (including the file name)
    # local_file_path = os.path.join(local_file_path, google_drive_model_path.split('/')[-1])

    # If the file doesn't exist, download it
    if not os.path.exists(google_drive_save_local_path):
        logger.info(f"{google_drive_save_local_path} does not exist, downloading...")
        gdown.download(f'https://drive.google.com/uc?id={file_id}', google_drive_save_local_path, quiet=False)
    else:
        logger.info(f"{google_drive_model_path} exists, skipping download")
