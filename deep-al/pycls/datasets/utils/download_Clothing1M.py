import gdown
from google_drive_downloader import GoogleDriveDownloader as gdd

url = 'https://drive.google.com/drive/folders/0B67_d0rLRTQYTmVnT1FuTVBuOTg?resourcekey=0-h8rTzyGd6PAR767zs3Nzvw'
file_id = "0-_XIUumw2vUE7EohI5h_qJw" # '0B67_d0rLRTQYdFpwZ09fNzF3NjA'
destination = "/cs/labs/daphna/nettashaf/TypiClustNoisy/data/Clothing1M/images/0.zip"

gdd.download_file_from_google_drive(file_id='1iytA1n2z4go3uVCwE__vIKouTKyIDjEq',
                                    dest_path=destination,
                                    unzip=True)

