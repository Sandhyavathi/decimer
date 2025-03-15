from typing import Optional, List, Dict, Tuple
from flask import Flask, request, jsonify
import os
import asyncio
import aiohttp
import requests
import time
import logging
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pickle
import tensorflow as tf
from utils import decode_image, decoder
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MAX_SEARCH_RETURN = 15

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{SMILE_STRING}/cids/JSON"
CID_DESCRIBE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{CID}/JSON"

class APIError(Exception):
    def __init__(self, error_code, message, status_code=400, details=None):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    model_paths["DECIMER"] = "/root/decimer/models/DECIMER_model"
    tokenizer_path = os.path.join(
        model_paths["DECIMER"], "assets", "tokenizer_SMILES.pkl"
    )


    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    model = tf.saved_model.load(model_paths["DECIMER"])

    return model, tokenizer

model, tokenizer = load_model()

def detokenize_output(predicted_array: int) -> str:
    """This function takes the predicted tokens from the DECIMER model and
    returns the decoded SMILES string.

    Args:
        predicted_array (int): Predicted tokens from DECIMER

    Returns:
        (str): SMILES representation of the molecule
    """
    outputs = [tokenizer.index_word[i] for i in predicted_array[0].numpy()]
    prediction = (
        "".join([str(elem) for elem in outputs])
        .replace("<start>", "")
        .replace("<end>", "")
    )
    return prediction


def get_compound_patents(cid: int, max_patents: int = 10) -> List[Dict]:
    """
    Retrieve patent information for a compound using its PubChem CID.
    
    Args:
        cid: The PubChem Compound ID (CID)
        max_patents: Maximum number of patents to return (default: 10)
        
    Returns:
        List of dictionaries with patent information
    """
    logger.info(f"Retrieving patent information for CID: {cid}")
    
    # PubChem API endpoint for patent information
    patent_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PatentID/JSON"
    
    try:
        # Get patent IDs associated with the compound
        response = requests.get(patent_url, timeout=30)
        response.raise_for_status()
        
        patent_data = response.json()
        
        # Check if patent information exists
        if "InformationList" not in patent_data or "Information" not in patent_data["InformationList"]:
            logger.info(f"No patent information found for CID: {cid}")
            return []
        
        # Extract patent IDs
        patent_info = patent_data["InformationList"]["Information"][0]
        if "PatentID" not in patent_info:
            logger.info(f"No patent IDs found for CID: {cid}")
            return []
        
        patent_ids = patent_info["PatentID"][:max_patents]
        logger.info(f"Found {len(patent_ids)} patents for CID: {cid}")
        
        # Get detailed information for each patent
        patents_with_details = []
        
        for patent_id in patent_ids:
            patent_details = get_patent_details(patent_id)
            if patent_details:
                patents_with_details.append(patent_details)
                
            # Be nice to the API with a small delay
            time.sleep(0.2)
        
        return patents_with_details
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error retrieving patent information for CID {cid}: {str(e)}")
        return []

def get_patent_details(patent_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific patent.
    
    Args:
        patent_id: The patent ID to retrieve details for
        
    Returns:
        Dictionary with patent details or None if details couldn't be retrieved
    """
    # For USPTO patents, use the PatFT API format
    # Example format for patent_id: "US20090123409A1"
    
    try:
        # Parse patent_id to extract information
        country_code = patent_id[:2] if patent_id[:2].isalpha() else ""
        number = ""
        kind_code = ""
        
        # Extract the patent number and kind code
        for i, char in enumerate(patent_id[2:]):
            if char.isalpha():
                number = patent_id[2:i+2]
                kind_code = patent_id[i+2:]
                break
        else:
            number = patent_id[2:]
        
        # Try to get additional information from PubChem
        # Since PubChem doesn't provide a direct API for patent details,
        # we'll return structured information based on the patent ID
        
        # For a production system, you might want to integrate with a patent database API
        # such as the USPTO's PatentsView API or the European Patent Office's OPS API
        
        return {
            "patent_id": patent_id,
            "country_code": country_code,
            "patent_number": number,
            "kind_code": kind_code,
            "url": get_patent_url(patent_id),
            "source": "PubChem"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving details for patent {patent_id}: {str(e)}")
        return None

def get_patent_url(patent_id: str) -> str:
    """
    Generate a URL to view the patent based on its ID.
    
    Args:
        patent_id: The patent ID
        
    Returns:
        URL to view the patent
    """
    # Handle different patent formats
    if patent_id.startswith("US"):
        # USPTO patents
        if "A" in patent_id:  # Application
            return f"https://patentimages.storage.googleapis.com/pdfs/{patent_id}.pdf"
        else:  # Granted patent
            return f"https://patents.google.com/patent/{patent_id}"
    elif patent_id.startswith("EP"):
        # European patents
        return f"https://patents.google.com/patent/{patent_id}"
    elif patent_id.startswith("WO"):
        # WIPO/PCT patents
        return f"https://patents.google.com/patent/{patent_id}"
    elif patent_id.startswith("JP"):
        # Japanese patents
        return f"https://patents.google.com/patent/{patent_id}"
    else:
        # Default to Google Patents search
        return f"https://patents.google.com/?q={patent_id}"

def call_pubchem_api(url, params=None) -> Optional[dict]:
    """
    Wrapper function to call PubChem API with proper error handling
    """
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json(), None
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error when calling PubChem API: {str(e)}")
        if "NameResolutionError" in str(e):
            return None, {
                "error": "network_error",
                "message": "Unable to connect to PubChem API. Please check your internet connection.",
                "details": "DNS resolution failed for PubChem API server."
            }
        return None, {
            "error": "connection_error",
            "message": "Failed to connect to PubChem API.",
            "details": str(e)
        }
    
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout when calling PubChem API: {str(e)}")
        return None, {
            "error": "timeout_error",
            "message": "Request to PubChem API timed out.",
            "details": "The request took too long to complete. Please try again later."
        }
    
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        logger.error(f"HTTP error {status_code} when calling PubChem API: {str(e)}")
        return None, {
            "error": "api_error",
            "message": f"PubChem API returned an error (HTTP {status_code}).",
            "details": str(e)
        }
    
    except Exception as e:
        logger.error(f"Unexpected error when calling PubChem API: {str(e)}")
        return None, {
            "error": "unexpected_error",
            "message": "An unexpected error occurred while calling PubChem API.",
            "details": str(e)
        }


def get_value_from_section(section):
    info=section.get("Information", [])
    if info:
        info = info[0]
        val = info["Value"]["StringWithMarkup"][0]["String"]
        return val

    return None


def similar_compound_info(s):
    iupac_name = None
    smile_string = None
    for k in s["Section"]:
        if (k["TOCHeading"] == "Computed Descriptors"):
            for l in k["Section"]:
                if l["TOCHeading"] == 'IUPAC Name':
                    iupac_name = get_value_from_section(l)
                if l["TOCHeading"] == "SMILES":
                    smile_string = get_value_from_section(l)
    
    return iupac_name, smile_string

async def fetch_compound_data(session, cid):
    """Fetch data for a single compound asynchronously"""
    url = CID_DESCRIBE_URL.format(CID=cid)
    async with session.get(url) as response:
        compound_data = await response.json()
        try:
            title = compound_data["Record"]["RecordTitle"]
            iupac_name = None
            smile_string = None
            
            for s in compound_data["Record"]["Section"]:
                if s["TOCHeading"] == "Names and Identifiers":
                    iupac_name, smile_string = similar_compound_info(s)
                    break
            
            return {
                "recordTitle": title,
                "iupacName": iupac_name,
                "smile": smile_string,
                "cid": cid
            }
        except:
            return None

async def get_similar_compounds_async(cid: int, topk=5) -> List[dict]:
    """Asynchronous version of get_similar_compound"""
    similar_compound_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/cid/{cid}/cids/JSON"
    
    async with aiohttp.ClientSession() as session:
        # Get similar CIDs
        async with session.get(similar_compound_url) as response:
            res = await response.json()
            similar_cids = res["IdentifierList"]["CID"][:topk]
        
        # Fetch all compound data concurrently
        tasks = [fetch_compound_data(session, similar_cid) for similar_cid in similar_cids]
        similar_compounds = await asyncio.gather(*tasks)
        
        return [s for s in similar_compounds if s is not None]

def get_similar_compound(cid: int, topk=5) -> List[dict]:
    """Optimized synchronous wrapper for the async implementation"""
    return asyncio.run(get_similar_compounds_async(cid, topk))

@lru_cache(maxsize=50)
def process_smile(smile:str) -> Optional[dict]:
    
    res, err = call_pubchem_api(URL.format(SMILE_STRING=smile))

    if err:
        raise APIError(
            error_code=err["error"],
            status_code=500,
            message=err["message"],
            details=err["details"]            
        )

    cids = res.get("IdentifierList", {}).get("CID", [])

    if len(cids) == 0:
        raise APIError(
            error_code=501,
            status_code=400,
            message="CID is not found or not detected from SMILE",
            details="Model predicted Smile has no CID"
        )

    cid = cids[0]
    c_res = requests.request("GET", CID_DESCRIBE_URL.format(CID = cid))
    compound_detail = c_res.json()
    if "Record" not in compound_detail:
        raise APIError(
            error_code=500,
            status_code=500,
            message="record not found in compound",
            details=f"record not found in compound {compound_detail}"
        )
    
    title = compound_detail["Record"]["RecordTitle"]
    similar_compound = get_similar_compound(cid, topk=MAX_SEARCH_RETURN)
    patents = get_compound_patents(cid, max_patents=MAX_SEARCH_RETURN)

    return {
        "currentCompound" : {
            "cid" : cid,
            "recordTitle" : title,
            "smile": smile
        },
        "similarCompound" : similar_compound,
        "patents" : patents
    }

def create_smile(out, threshold = 0.70):
    smile = ""
    for atom, score in out:
        if score >= threshold:
            smile += atom

    return smile

@app.route('/api/process-image', methods=['POST'])
def upload_image():
    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({"error": "No image provided in the request"}), 400
    
    file = request.files['image']
    
    # Check if the user submitted an empty file
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    try:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image
        # sm, results = predict_SMILES(file_path, confidence=True)
        # results = create_smile(results, threshold=0.7)
        # results = predict_SMILES(file_path)

        chemical_structure = decode_image(file_path)
        predicted_tokens, _ = model(tf.constant(chemical_structure))
        results = decoder(detokenize_output(predicted_tokens))
        res = process_smile(results)
        # Return the results
        return jsonify({
            "success": True,
            "filename": filename,
            "detectedSmile": results,
            "pubchemResults" : res
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search-smile', methods=['POST'])
def search_smile():
    req = request.json
    result = req.get("smileString")

    if result is None:
        return jsonify({"error": "No smile string in request"}), 400
        
    try:
        res = process_smile(result)
        # Return the results
        return jsonify({
            "success": True,
            "pubchemResults" : res
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
