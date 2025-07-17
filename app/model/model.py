import torch
import segmentation_models_pytorch as smp
import time

# Hàm load model
def load_model(weights_path, num_classes=4):
    model = smp.DeepLabV3Plus(
        encoder_name='mobilenet_v2',
        encoder_weights=None,  
        in_channels=3,
        classes=num_classes,
    )
    model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    model.eval().to('cuda')
    return model

# Hàm đo thời gian suy luận
@torch.no_grad()
def infer_and_measure(model, image_tensor):
    start = time.time()
    output = model(image_tensor)
    end = time.time()
    infer_time_ms = (end - start) * 1000  # milliseconds
    return output, infer_time_ms