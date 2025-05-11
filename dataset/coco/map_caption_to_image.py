import json
import os.path

import pandas as pd

from settings import BASE_DIR

# Đường dẫn tới file JSON
json_path = os.path.join(BASE_DIR, "dataset", "coco", "annotations", "captions_train2017.json")

# Đọc file JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Tạo mapping từ image_id -> file_name
id_to_filename = {img['id']: img['file_name'] for img in data['images']}

# Tạo danh sách các dòng dữ liệu: (file_name, caption)
records = []
for ann in data['annotations']:
    image_id = ann['image_id']
    caption = ann['caption']
    file_name = id_to_filename[image_id]
    records.append({'filename': file_name, 'caption': caption})

# Tạo DataFrame
df = pd.DataFrame(records)

# Hiển thị vài dòng đầu
print(df.info())

df.to_csv(os.path.join(BASE_DIR, "dataset", "coco", "captions_train.csv"), index=False)

