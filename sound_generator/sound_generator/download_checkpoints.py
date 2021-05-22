"""
Script to download checkpoints from Google Drive

Original snippet:
https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

Run:
    python download_checkpoints.py
"""
import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768 * 16

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

    files_to_process = {
        "ddsp_generator": [
            {
                "file": "operative_config-30000.gin", 
                "file_id": "1BHQYyLRlNhnj4kM1-TZbZpNjgLTqNN0D"
            },
            {
                "file": "ckpt-30000.index", 
                "file_id": "1r928iQnrdHK-aCfe_xWt1zDsA_pCrEAR"
            },
            {
                "file": "ckpt-30000.data-00000-of-00001",
                "file_id": "1YTo1H0uVN-HYNnXPP3dc--9iLMZE6SOI"
            },
            {
                "file": "checkpoint",
                "file_id": "1E6wvvKOSGQZRf_vUNi-9BKBm45GQiAT3"
            }
        ],
        "f0_ld_generator": [
            {
                "file": "cp.ckpt.index",
                "file_id": "1mduQ9trZPDt7WF79qjxI0FkdN5cyXLIj"
            },
            {
                "file": "cp.ckpt.data-00000-of-00001",
                "file_id": "1UmOZsL65z5pddg17qrg-nH7OlEsqYoj5"
            },
            {
                "file": "checkpoint",
                "file_id": "1OAqJ7-2FwIJ9HKvDWbRHJqov0L9z07VD"
            }
        ],
        "z_generator": [
            {
                "file": "cp.ckpt.index",
                "file_id": "1HeB7J0Hxw5sFw6I52td6y4Wh-yCK4u5Q"
            },
            {
                "file": "cp.ckpt.data-00000-of-00001",
                "file_id": "1lbNQF5A1AU_5wQ-QmOHCfkaW9HDMWpoT"
            },
            {
                "file": "checkpoint",
                "file_id": "17dHEeWUlHsaF9cqpISAV736AX2ZKY8m5"
            }
        ]
    }

    for checkpoint in files_to_process:
        print(checkpoint)
        checkpoint_folder = os.path.join(CHECKPOINTS_DIR, checkpoint)
        
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        
        for file_to_process in files_to_process[checkpoint]:
            file_path = os.path.join(checkpoint_folder, file_to_process["file"])
            file_id = file_to_process["file_id"]

            download_file_from_google_drive(file_id, file_path)
            print(f"File downloaded: {file_path}")
            

    

if __name__ == "__main__":
    main()