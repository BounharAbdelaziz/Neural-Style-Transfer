from PIL import Image
import os

def save_img(generated_img, img_name, path):
    array = generated_img.detach().cpu().numpy()

    print(f'Image shape : {array.size}')

    img = Image.fromarray(array)
    img.save(os.path.join(path, img_name))
