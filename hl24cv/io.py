from pathlib import Path
import numpy as np

def load_extrinsics(extrinsics_path):
    assert Path(extrinsics_path).exists()
    mtx = np.loadtxt(str(extrinsics_path), delimiter=',').reshape((4, 4))
    return mtx

def load_rig2world_transforms(path):
    transforms = {}
    data = np.loadtxt(str(path), delimiter=',')
    for value in data:
        timestamp = value[0]
        transform = value[1:].reshape((4, 4))
        transforms[timestamp] = transform
    return transforms