import os
import torch
import shutil

__all__ = ['save_checkpoint']


def save_checkpoint(state, directory, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.path.tar'))

