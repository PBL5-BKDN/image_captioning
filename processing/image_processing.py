from torchvision.transforms import transforms


def process_image(image):
    """
    Preprocess the image for the model.
    :param image: PIL Image
    :return: Tensor of shape (3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)