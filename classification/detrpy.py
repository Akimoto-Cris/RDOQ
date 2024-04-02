import torch
import torch.nn as nn
from detr.models import build_model
from detr.models.detr import MLP

def detr_coco(args):
    model, criterion, postprocessors = build_model(args)
    
    for m in model.modules():
        if isinstance(m, )

    return model

