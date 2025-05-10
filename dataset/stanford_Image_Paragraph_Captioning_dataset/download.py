import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("vakadanaveen/stanford-image-paragraph-captioning-dataset")
print("Downloaded to:", path)

# Get current working directory (thư mục hiện tại)
current_dir = os.getcwd()
print("Current working directory:", current_dir)

# Move all files/folders from downloaded path to current directory
for item in os.listdir(path):
    s = os.path.join(path, item)  # source path
    d = os.path.join(current_dir, item)  # destination path

    if os.path.isdir(s):
        # Copy entire directory
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        # Copy individual file
        shutil.copy2(s, d)

print("✅ Dataset files moved to current directory.")