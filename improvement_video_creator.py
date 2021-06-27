import numpy as np
import cv2
import os
import argparse
import pathlib

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default='./images/transformed/abdel/style_3/', help='Path to the images from which create the video.')
    parser.add_argument("--dest", type=str, default='./videos/transformed/abdel/style_3/', help='Path where to store the video.')
    parser.add_argument("--filename", type=str, default='abdel_style.mp4', help='Name of the video file.')

    args = parser.parse_args()

    source = pathlib.Path(args.source)
    dest = pathlib.Path(args.dest)
    filename = args.filename

    dest.mkdir(parents=True, exist_ok=True)

    image_names = sorted(source.iterdir(), key=os.path.getmtime) #os.listdir(source)

    # images_ = np.zeros(len(image_names))
    images_ = []
    print(f'Total number of images : {len(image_names)}')

    for idx, image_name in enumerate(image_names) :
        images_.append(cv2.imread(os.path.join(str(image_name))))

    w, h, c = images_[0].shape

    video = cv2.VideoWriter(os.path.join(str(dest), filename), -1, 10, (w, h))

    for i in range(len(images_)):
        video.write(images_[i])
    
    cv2.destroyAllWindows()
    video.release()
    
    