import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb


def reconstruct(model, imgs, device):
    imgs = imgs.to(device)
    e = model._pre_vq_conv(model._encoder(imgs))
    _, q, _, _ = model._vq_vae(e)
    return model._decoder(q)
