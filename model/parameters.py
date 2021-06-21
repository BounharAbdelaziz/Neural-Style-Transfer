import torch

class Parameters():
  
  def __init__(self, 
                  lr=0.001, n_epochs=1000, with_normalization=False, save_image=100, content_weight=0.15, style_weight=0.02,
                  org_img_path="./images/originals/lion.jpg", style_img_path="./images/styles/style_0.jpg", transformed_img_path="./images/transformed/", experiment_name="nst_0"):

    # Params and Hyperparams
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.lr = lr
    self.n_epochs = n_epochs
    self.with_normalization=with_normalization
    self.save_image=save_image

    # content and style loss
    self.content_weight = content_weight
    self.style_weight = style_weight

    # Paths and exp name
    self.org_img_path=org_img_path
    self.style_img_path=style_img_path
    self.transformed_img_path=transformed_img_path
    self.experiment_name=experiment_name
