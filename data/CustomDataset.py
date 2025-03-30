from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
import os
from processing import text_processing
def get_project_root() -> str:
    # Get the current working directory
    current_dir = os.getcwd()

    # Traverse up to the root directory (you can adjust the number of levels as needed)
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

    return project_root

# Example usage
images_dir = os.path.join(get_project_root(),'data/Images')





class CustomDataset(Dataset):
    def __init__(self, path):

        df = pd.read_csv(path, delimiter=',')
        self.data = df['image']
        self.labels = df['caption']
        print(f"Loaded {len(self.data)} images and {len(self.labels)} captions.")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        img_path = os.path.join(images_dir, self.data[index])
        image = transform(Image.open(img_path))
        label = text_processing.caption_preprocessing(self.labels[index])
        return image, label


