import torch.nn as nn
from torchvision import transforms
from PIL import Image

class Preprocessing(nn.Module):
    def __init__(self, w, h):
        super(Preprocessing, self).__init__()
        assert (w == h), "[Preprocessing-init] Width and Height should be the same !"
        self.transforms = transforms.Compose([
            transforms.Resize((w, h)),
            transforms.ToTensor(),
        ])
    
    def process_img(self, img_path):
        print(f'Loading image from path : {img_path}')

        img = Image.open(img_path)
        # print(f'Image shape after loading : {img.size}')
        img = self.transforms(img)
        # print(f'Image shape after transforms : {img.shape}')
        img = img.unsqueeze(0)
        # print(f'Image shape after unsqueeze : {img.shape}')

        return img