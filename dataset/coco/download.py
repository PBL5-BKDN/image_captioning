import os
import urllib.request
import zipfile

from settings import BASE_DIR

# Thư mục lưu trữ dữ liệu
DATA_DIR = os.path.join(BASE_DIR, "dataset", "coco")
os.makedirs(DATA_DIR, exist_ok=True)

# Danh sách các URL cần tải (ảnh và caption annotations)
urls = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

def download_and_extract(name, url, save_dir):
    zip_path = os.path.join(save_dir, f'{name}.zip')
    if not os.path.exists(zip_path):
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(url, zip_path)
        print(f'{name} downloaded.')

    extract_dir = os.path.join(save_dir)
    print(f'Extracting {name}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f'{name} extracted.')

# Tải và giải nén
for name, url in urls.items():
    download_and_extract(name, url, DATA_DIR)
