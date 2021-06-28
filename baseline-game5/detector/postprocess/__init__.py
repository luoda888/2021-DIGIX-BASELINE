import subprocess
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# if subprocess.call(['make', '-C', BASE_DIR]) != 0:
if subprocess.call([r'make', '-C', BASE_DIR]) != 0:
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

from .simple_dilate import simple_dilate as csimple_dilate

def simple_dilate(score, kernel, score_thr=0.8,
                  expand_scale=1.8, min_area=32, min_border=3):
    dilated_polys = []
    ret = csimple_dilate(kernel, score, score_thr, expand_scale, min_area, min_border)
    for poly in ret:
        dilated_polys.append(np.array(poly).reshape(-1, 2))
    return dilated_polys

