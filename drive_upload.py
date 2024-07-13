import os
import pickle
import json
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive.readonly']

def authenticate():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        with open('token.json', 'r') as token:
            creds = google.oauth2.credentials.Credentials.from_authorized_user_info(json.load(token), SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_folder_id(service, folder_name, parent_id=None):
    """Get the folder ID by name, create it if it doesn't exist."""
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)').execute()
    items = results.get('files', [])
    
    if not items:
        # Folder does not exist, create it
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]
        folder = service.files().create(body=file_metadata,
                                        fields='id').execute()
        print(f"Created folder {folder_name} with ID: {folder.get('id')}")
        return folder.get('id')
    else:
        return items[0]['id']

def delete_existing_file(service, file_name, parent_id):
    """Delete the existing file with the same name in the specified folder."""
    query = f"name = '{file_name}' and '{parent_id}' in parents"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)').execute()
    items = results.get('files', [])
    
    for item in items:
        service.files().delete(fileId=item['id']).execute()
        print(f"Deleted file: {item['name']} (ID: {item['id']})")

def upload_file(file_path, drive_service, parent_id):
    file_name = os.path.basename(file_path)
    # Delete existing file if it exists
    delete_existing_file(drive_service, file_name, parent_id)
    
    # Upload new file using resumable upload
    file_metadata = {
        'name': file_name,
        'parents': [parent_id]
    }
    media = MediaFileUpload(file_path, mimetype='application/octet-stream', resumable=True)
    request = drive_service.files().create(body=file_metadata, media_body=media, fields='id')
    
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")
    print(f'Uploaded file ID: {response.get("id")}')


def upload(file_path_local, parent_directory_remote, file_name_remote):
    creds = authenticate()
    drive_service = build('drive', 'v3', credentials=creds)
    dissertation_folder_id = get_folder_id(drive_service, 'Dissertation')

    # This takes a file from the local machine and uploads it to the Google Drive, the file is saved at file_path_local and is uploaded to file_path_remote
    # First we need the folder id of file_path_remote
    parent_directory_id = get_folder_id(drive_service, parent_directory_remote, parent_id=dissertation_folder_id)
    # Extract name of local file from file_path_local
    file_name = os.path.basename(file_path_local)
    # Upload the file with message to the user
    print(f"Uploading {file_name} to {parent_directory_remote}...")
    upload_file(file_path_local, drive_service, parent_directory_id)

def update_rl_models():
    # This function will update the RL-Model on the Google Drive, by inserting all the latest versions of the files which have been edited here
    # The RL model is stored in the folder 'RL Model' in the 'Dissertation' folder
    rl_model_document_list = ["CartPole.ipynb"]
    for document in rl_model_document_list:
        filename = f"RL-Models/{document}"
        # check it exists
        if not os.path.exists(filename):
            print(f"File {filename} does not exist")
            pass
        upload(filename, "RL-Models", filename)

def download_folder(folder_name):
    # This function will download all the models in the 'Models' folder in the 'Dissertation' folder and save them to the local machine in the Trained-Models folder
    print(f"Downloading {folder_name} folder from Google Drive...")
    creds = authenticate()
    drive_service = build('drive', 'v3', credentials=creds)
    dissertation_folder_id = get_folder_id(drive_service, 'Dissertation')
    models_folder_id = get_folder_id(drive_service, folder_name, parent_id=dissertation_folder_id)

    results = drive_service.files().list(q=f"'{models_folder_id}' in parents", spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])

    for item in items:
        file_id = item['id']
        file_name = item['name']
        request = drive_service.files().get_media(fileId=file_id)
        with open(f'{folder_name}/{file_name}', 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Downloaded {int(status.progress() * 100)}%")


def main():
    # This function will handle syncing the local machine with the Google Drive
    # collect user input if they want to upload or download.
    # If they enter 'upload', run update_rl_models()
    # If they enter 'download', run download_trained_models() and download_rl_models()

    # collect user input
    user_input = input("Do you want to upload or download? ")
    if user_input == "upload" or user_input == "u":
        update_rl_models()
    elif user_input == "download" or user_input == "d":
        download_folder("Trained-Models")
        download_folder("RL-Models")

    else:
        print("Invalid input, please enter 'upload' or 'download'")

if __name__ == "__main__":
    main()