import os
import shutil
import zipfile
import pystow
from pathlib import Path

# def download_trained_weights(model_url: str, model_path: str, verbose=1):
#     """This function downloads the trained models and tokenizers to a default
#     location. After downloading the zipped file the function unzips the file
#     automatically. If the model exists on the default location this function
#     will not work.

#     Args:
#         model_url (str): trained model url for downloading.
#         model_path (str): model default path to download.

#     Returns:
#         path (str): downloaded model.
#     """
#     # Download trained models
#     if verbose > 0:
#         print("Downloading trained model to " + str(model_path))
#     model_path = pystow.ensure("DECIMER-V2", url=model_url)
#     if verbose > 0:
#         print(model_path)
#         print("... done downloading trained model!")

#     with zipfile.ZipFile(model_path.as_posix(), "r") as zip_ref:
#         zip_ref.extractall(model_path.parent.as_posix())

#     # Delete zipfile after downloading
#     if Path(model_path).exists():
#         Path(model_path).unlink()

def download_trained_weights(model_url: str, model_path: str, verbose=1):
    """This function downloads the trained models and tokenizers to a default
    location. After downloading the zipped file the function unzips the file
    automatically. If the model exists on the default location this function
    will not work.

    Args:
        model_url (str): trained model url for downloading.
        model_path (str): model default path to download.

    Returns:
        path (str): downloaded model.
    """
    # Download trained models
    if verbose > 0:
        print(f"Downloading trained model from {model_url} to {model_path}")
    model_zip_path = pystow.ensure("DECIMER-V2", url=model_url)
    if verbose > 0:
        print(f"Downloaded to: {model_zip_path}")
        print("Extracting files...")

    # Extract to parent directory
    with zipfile.ZipFile(model_zip_path.as_posix(), "r") as zip_ref:
        zip_ref.extractall(Path(model_path).parent.as_posix())
    
    # Delete zipfile after downloading
    if Path(model_zip_path).exists():
        Path(model_zip_path).unlink()
        print("Zip file deleted.")
    
    # Ensure correct structure (files need to be in DECIMER_model/assets)
    expected_tokenizer_path = os.path.join(model_path, "assets", "tokenizer_SMILES.pkl")
    extracted_path = Path(model_path).parent / "models"
    
    if not os.path.exists(expected_tokenizer_path) and extracted_path.exists():
        print(f"Fixing directory structure...")
        # Check various possible locations for the tokenizer
        possible_locations = [
            extracted_path / "tokenizer_SMILES.pkl",
            extracted_path / "assets" / "tokenizer_SMILES.pkl",
            Path(model_path).parent / "tokenizer_SMILES.pkl",
        ]
        
        for loc in possible_locations:
            if loc.exists():
                print(f"Found tokenizer at {loc}")
                # Create destination directory if needed
                os.makedirs(os.path.join(model_path, "assets"), exist_ok=True)
                # Copy the tokenizer to the expected location
                shutil.copy(loc, expected_tokenizer_path)
                print(f"Copied tokenizer to {expected_tokenizer_path}")
                break
        
        # Copy other necessary files
        if os.path.exists(extracted_path / "saved_model.pb"):
            print("Copying saved_model.pb")
            shutil.copy(extracted_path / "saved_model.pb", model_path / "saved_model.pb")
            
            # Copy any subdirectories like variables or assets
            for subdir in ["variables", "assets"]:
                if os.path.exists(extracted_path / subdir):
                    if os.path.exists(model_path / subdir):
                        shutil.rmtree(model_path / subdir)
                    shutil.copytree(extracted_path / subdir, model_path / subdir)
                    print(f"Copied {subdir} directory")

def ensure_model(default_path: str, model_urls: dict) -> dict:
    """Function to ensure models are present locally.

    Convenient function to ensure model downloads before usage

    Args:
        default_path (str): Default path for model data
        model_urls (dict): Dictionary containing model names as keys and their corresponding URLs as values

    Returns:
        dict: A dictionary containing model names as keys and their local paths as values
    """
    model_paths = {}
    # Store st_size of each model
    model_sizes = {
        "DECIMER": 28080309
    }
    for model_name, model_url in model_urls.items():
        model_path = os.path.join(default_path, f"{model_name}_model")
        if os.path.exists(model_path) and os.stat(
            os.path.join(model_path, "saved_model.pb")
        ).st_size != model_sizes.get(model_name):
            print(f"Working with model {model_name}")
            shutil.rmtree(model_path)
            download_trained_weights(model_url, default_path)
        elif not os.path.exists(model_path):
            download_trained_weights(model_url, default_path)

        # Store the model path
        model_paths[model_name] = model_path
    return model_paths


def download_model1():
    default_path = "./.data/DECIMER-V2"
    os.makedirs(default_path, exist_ok=True)

    model_urls = {
        "DECIMER": "https://zenodo.org/record/8300489/files/models.zip",
        # "DECIMER_HandDrawn": "https://zenodo.org/records/10781330/files/DECIMER_HandDrawn_model.zip",
    }

    model_paths = ensure_model(default_path=default_path, model_urls=model_urls)
    return model_paths


if __name__ == "__main__":
    print(download_model1())