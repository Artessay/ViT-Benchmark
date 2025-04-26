from vision_transformer import VisionTransformer

def active_head(model: VisionTransformer):
    for name, param in model.named_parameters():
        if "heads" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def active_full(model: VisionTransformer):
    for param in model.parameters():
        param.requires_grad = True