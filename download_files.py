import os
import pickle
import sys
import io
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

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

def download_file(service, file_id, output_path):
    try:
        # Get the file metadata
        file = service.files().get(fileId=file_id).execute()
        file_name = file.get('name', 'untitled')

        # Create a BytesIO stream to store the file content
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        # Download the file
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

        # Save the file
        full_path = os.path.join(output_path, file_name)
        with open(full_path, 'wb') as f:
            fh.seek(0)
            f.write(fh.read())

        print(f"File downloaded successfully: {full_path}")
        return full_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python download_script.py <file_id> <output_directory>")
        sys.exit(1)

    file_id = sys.argv[1]
    output_directory = sys.argv[2]

    if not os.path.isdir(output_directory):
        print(f"Error: {output_directory} is not a valid directory.")
        sys.exit(1)

    service = get_drive_service()
    downloaded_file_path = download_file(service, file_id, output_directory)

    if downloaded_file_path:
        print(f"File downloaded to: {downloaded_file_path}")
    else:
        print("File download failed.")