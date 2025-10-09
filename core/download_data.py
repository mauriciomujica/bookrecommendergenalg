import os
import gdown
import zipfile

def download_and_extract_from_gdrive(file_id, output_dir="books_data"):
    """
    Download a zip file from Google Drive using gdown and extract it.
    """
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "dataset.zip")

    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading from Google Drive file ID: {file_id}")
    gdown.download(url, zip_path, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)

    print(f"Dataset is ready in '{output_dir}'")


if __name__ == "__main__":
    FILE_ID = "1o7DF--65hU9NuQX8OXPoUC579bxL3bxW"
    download_and_extract_from_gdrive(FILE_ID)
