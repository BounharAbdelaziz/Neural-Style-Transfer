import torch.nn as nn

class StyleLoss(nn.Module) :

    # Gram matrix loss
    def __init__(self, style_weight, c, w, h):
        super(StyleLoss, self).__init__()
        # weight of the style loss
        self.c = c
        self.w = w
        self.h = h
        self.criterion = nn.MSELoss()

    def gram_matrix(self, A, c, w, h):

        # Gram matrix = A*A^T
        return A.view(c, w*h).mm(A.view(c, w*h).t())

    def forward(self, stl_features, gen_features):

        
        # compute gram matrices
        G_stl = self.gram_matrix(stl_features, self.c, self.w, self.h)
        G_gen = self.gram_matrix(gen_features, self.c, self.w, self.h)

        return self.criterion(G_stl, G_gen)

class ContentLoss(nn.Module) :

    # Content loss (L2 or L1)
    def __init__(self, content_weight):
        super(ContentLoss, self).__init__()
        # weight of the content loss
        self.criterion = nn.MSELoss()

    def forward(self, orig_features, gen_features):
        return self.criterion(orig_features, gen_features)
