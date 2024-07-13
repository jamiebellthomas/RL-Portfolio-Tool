import os
import pickle
import json
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from create_universe import main_create
from macro_economic_factors import generate_macro_economic_file

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

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

def main():

    # Create the asset universe and macro economic factors collections
    #main_create()
    #generate_macro_economic_file()

    creds = authenticate()
    drive_service = build('drive', 'v3', credentials=creds)

    # Ensure the Dissertation/Collections folder exists and get its ID
    dissertation_folder_id = get_folder_id(drive_service, 'Dissertation')
    collections_folder_id = get_folder_id(drive_service, 'Collections', parent_id=dissertation_folder_id)

    # Path to the .pkl file in your project folder
    project_folder = 'Collections'
    macro_economic_pkl_file = 'macro_economic_factors.pkl'
    macro_economic_pkl_file = os.path.join(project_folder, macro_economic_pkl_file)

    asser_universe_pkl_file = 'asset_universe.pkl'
    asser_universe_pkl_file = os.path.join(project_folder, asser_universe_pkl_file)



    # Ensure the .pkl file exists
    if os.path.exists(macro_economic_pkl_file):
        upload_file(macro_economic_pkl_file, drive_service, collections_folder_id)
        print(f"File uploaded successfully: {macro_economic_pkl_file}")
    else:
        print(f"File does not exist: {macro_economic_pkl_file}")

    if os.path.exists(asser_universe_pkl_file):
        upload_file(asser_universe_pkl_file, drive_service, collections_folder_id)
        print(f"File uploaded successfully: {asser_universe_pkl_file}")
    else:
        print(f"File does not exist: {asser_universe_pkl_file}")

if __name__ == '__main__':
    main()
