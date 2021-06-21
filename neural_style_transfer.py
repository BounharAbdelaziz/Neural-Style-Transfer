import torch.nn as nn
import torch
import argparse

from model import model
from model.parameters import Parameters
from model.model import StyleTransfer, Optimization
from data.preprocessing import Preprocessing

def run_optimization(params, model, original_img_path, style_img_path, w=128, h=128):
    # preprocess images
    preprocessing = Preprocessing(w, h)
    original_img = preprocessing.process_img(original_img_path)
    style_img = preprocessing.process_img(style_img_path)
    
    generated_img = original_img.clone().requires_grad_(True).to(params.device)

    optimization = Optimization(params)
    # start optimization
    optimization.train(model, params.n_epochs, params.lr, original_img, style_img, generated_img)

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--with_normalization", type=int, default=0, help="Normalize the inpyt image to the .")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-03, help="Learning rate of the optimizer.")
    parser.add_argument("--content_weight", type=float, default=1, help="Weight of the content in the loss function.")
    parser.add_argument("--style_weight", type=float, default=0.2, help="Weight of the style in the loss function.")
    parser.add_argument("--save_image", type=int, default=100, help="Number of iterations before saving an image.")
    
    # Paths and exp name
    parser.add_argument("--org_img_path", type=str, default='./images/originals/lion.jpg', help="Path to the original image.")
    parser.add_argument("--style_img_path", type=str, default='./images/styles/style_0.jpg', help="Path to the style image.")
    parser.add_argument("--transformed_img_path", type=str, default='./images/transformed/', help="Path where to store the transformed image.")
    parser.add_argument("--experiment_name", type=str, default='nst_0', help="Experiment name, will create a directory in /images and store the generated images there accordingly.")

    args = parser.parse_args()

    params = Parameters(lr=args.lr, n_epochs=args.n_epochs, with_normalization=args.with_normalization, save_image=args.save_image, content_weight=args.content_weight, style_weight=args.style_weight,
                  org_img_path=args.org_img_path, style_img_path=args.style_img_path, transformed_img_path=args.transformed_img_path, experiment_name=args.experiment_name)



    print("## ------------------------------------------------------------------------- ##")
    print("lr : ", args.lr)
    print("n_epochs : ",args.n_epochs)
    print("with_normalization : ",args.with_normalization)
    print("save_image : ",args.save_image)
    print("org_img_path : ",args.org_img_path)
    print("style_img_path : ",args.style_img_path)
    print("experiment_name : ",args.experiment_name)
    
    # Init style transfer model
    StyleTransfer = StyleTransfer()

    # start the optimization process
    run_optimization(params, StyleTransfer, original_img_path=params.org_img_path, style_img_path=params.style_img_path,  w=256, h=256)