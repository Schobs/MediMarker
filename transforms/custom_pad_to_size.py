import torch
import torchio as tio

class CustomPadToSize(tio.Transform):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def apply_transform(self, subject):
        for image_name, image in subject.get_images_dict().items():
            h, w = image.shape[1:3]
            pad_h = max(self.target_size[0] - h, 0)
            pad_w = max(self.target_size[1] - w, 0)

            # Calculate padding for each dimension
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            # Use PyTorch's pad function
            padded_image = torch.nn.functional.pad(
                image.data, (pad_left, pad_right, pad_top, pad_bottom)
            )

            # Update image data in the subject
            image.set_data(padded_image)

        return subject
