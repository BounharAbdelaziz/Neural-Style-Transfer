import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.utils import save_image
import numpy as np
from model.loss import StyleLoss, ContentLoss

import os
from tqdm import tqdm


class StyleTransfer(nn.Module) :

    def __init__(self, device="cpu"):
        super(StyleTransfer, self).__init__()

        # load the pretrained model. should be in eval mode !
        self.model = models.vgg19_bn(pretrained=True).features[:28].to(device).eval() 

        # we take only 5 first layers that comes after a MaxPool2d according to https://arxiv.org/pdf/1508.06576.pdf
        self.features_layers = ['0', '7', '10', '14', '27']
    
    def forward(self, x):

        extracted_features = []

        for layer_idx, layer in enumerate(self.model):
            # forward on each layer
            x = layer(x)

            # add features matrices of the layers we are intersted in
            if str(layer_idx) in self.features_layers :
                extracted_features.append(x)

        return np.asarray(extracted_features)


class Optimization(nn.Module) :

    def __init__(self, params):
        super(Optimization, self).__init__()

        self.style_model = StyleTransfer()
        self.params = params
        self.contentCriterion = ContentLoss(params.content_weight)
    
    def train(self, model, n_epochs, lr, original_img, style_img, generated_img):
        

        optimizer = optim.Adam([generated_img], lr=lr)

        print("## ------------------------------------------------------------------------- ##")
        print("[INFO] Started optimization using device : ",self.params.device)
        print("## ------------------------------------------------------------------------- ##")

        for step in tqdm(range(n_epochs)):
            original_features = self.style_model(original_img)
            style_features = self.style_model(style_img)
            generated_features = self.style_model(generated_img)

            style_loss = 0
            content_loss = 0

            for orig_features, stl_features, gen_features in zip(original_features, style_features, generated_features) :
                
                # get shape
                batch_size, c, w, h = gen_features.shape

                # compute content loss ; between original image and generated image
                content_loss = content_loss + self.contentCriterion(orig_features, gen_features)

                # compute style loss ; between style image and generated image
                self.styleCriterion = StyleLoss(self.params.style_weight, c, w, h)
                style_loss = style_loss + self.styleCriterion(stl_features, gen_features)


            loss = self.params.style_weight*style_loss + self.params.content_weight*content_loss
            
            print(f'Step [{step}/{n_epochs}] : loss = {loss}')

            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            optimizer.step()

            if step % self.params.save_image == 0 :
                path_save_img = os.path.join(self.params.transformed_img_path, "gen_"+str(step)+".png")
                print(f'Step [{step}/{n_epochs}] saving image in : {path_save_img}')

                save_image(generated_img, path_save_img)
