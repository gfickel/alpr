import os
import pickle
import sys
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def get_or_create_folder(service, folder_name):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    folders = results.get('files', [])
    
    if folders:
        return folders[0]['id']
    else:
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        return folder.get('id')

def upload_file(service, file_path, parent_folder_id):
    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [parent_folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path1> <file_path2> ...")
        sys.exit(1)

    files_to_upload = sys.argv[1:]
    
    service = get_drive_service()
    
    # Get or create the 'ocr_training' folder
    # ocr_folder_id = get_or_create_folder(service, 'ocr_training')
    ocr_folder_id = get_or_create_folder(service, 'alpr_datasets')
    
    for file_path in files_to_upload:
        if not os.path.isfile(file_path):
            print(f"Error: {file_path} is not a valid file.")
            continue

        uploaded_file_id = upload_file(service, file_path, ocr_folder_id)
        print(f"{uploaded_file_id}")