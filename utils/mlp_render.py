import cv2
import os
import imageio
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.ndimage import correlate
import scipy.ndimage as ndimage

from fots_sim.planar_shadow import planar_shadow
from fots_sim.utils.prepost_mlp import preproc_mlp
from fots_sim.mlp_model import MLP

w,h = 240, 320

def padding(img):
    # pad one row & one col on each side
    if len(img.shape) == 2:
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')
    elif len(img.shape) == 3:
        return np.pad(img, ((1, 1), (1, 1), (0, 0)), 'symmetric')


def generate_normals(height_map):
    [h, w] = height_map.shape
    center = height_map[1:h - 1, 1:w - 1]  # z(x,y)
    top = height_map[0:h - 2, 1:w - 1]  # z(x-1,y)
    bot = height_map[2:h, 1:w - 1]  # z(x+1,y)
    left = height_map[1:h - 1, 0:w - 2]  # z(x,y-1)
    right = height_map[1:h - 1, 2:w]  # z(x,y+1)
    dzdx = (bot - top) / 2.0
    dzdy = (right - left) / 2.0
    
    # Numerical Hardening: Capture and neutralize physical divergence
    dzdx = np.nan_to_num(dzdx, nan=0.0, posinf=0.0, neginf=0.0)
    dzdy = np.nan_to_num(dzdy, nan=0.0, posinf=0.0, neginf=0.0)
    dzdx = np.clip(dzdx, -10.0, 10.0)
    dzdy = np.clip(dzdy, -10.0, 10.0)
    direction = np.ones((h - 2, w - 2, 3))
    direction[:, :, 0] = dzdy
    direction[:, :, 1] = -dzdx

    magnitude = np.sqrt(direction[:, :, 0] ** 2 + direction[:, :, 1] ** 2 + direction[:, :, 2] ** 2)
    normal = direction / magnitude[:, :, np.newaxis]  # unit norm

    normal = padding(normal)

    normal = (normal+1.0) * 0.5

    return normal


class CalibData:
    def __init__(self, data):
        data = data

        self.numBins = data['bins']
        self.grad_r = data['grad_r']
        self.grad_g = data['grad_g']
        self.grad_b = data['grad_b']


class MLPRender:

    def __init__(self, **config):
        self.background = config['background_img']
        self.bg_depth = config['bg_depth']
        self.bg_render = config['bg_render']
        self.model = config['model']
        
        # Pre-scale initial depth to avoid recomputing every frame
        # Conversion factor: -1000 / (0.0266 * 2)
        self._scale = -1000.0 / (0.0266 * 2.0)
        self._pre_scaled_bg = self.bg_depth * self._scale

    def smooth_heightMap(self, height_map, bg_depth):
        diff_depth = np.abs(height_map - bg_depth)
        
        contact_mask_0 = diff_depth > 0.0
        # Aggressive threshold to reduce noise
        contact_mask = diff_depth > (np.max(diff_depth) * 0.4)
        
        height_map = diff_depth.copy()
        zq_back = height_map.copy()

        # Gaussian smoothing is expensive; reduced iterations for speed
        kernel_size = [21, 11, 5]
        for ks in kernel_size:
            # Sanitize height_map to prevent NaN-blooming during blur
            height_map = np.nan_to_num(height_map, nan=0.0)
            height_map = cv2.GaussianBlur(height_map.astype(np.float32), (ks, ks), 0)
            
            # Sanitize zq_back before copy-back
            clean_zq = np.nan_to_num(zq_back, nan=0.0)
            height_map[contact_mask] = clean_zq[contact_mask]
        
        return height_map, contact_mask_0, diff_depth

    def generate(self, heightMap, shadow = True):
        # Scale current depth
        hMap = heightMap * self._scale
        
        # Smooth and get mask
        hMap_smoothed, contact_mask, contact_height = self.smooth_heightMap(hMap, self._pre_scaled_bg)
        
        # Generate normals
        normal = generate_normals(hMap_smoothed)
        img_n = preproc_mlp(normal)
        
        with torch.no_grad():
            sim_img_r = self.model(img_n).cpu().numpy()

        sim_img = sim_img_r.reshape(320, 240, 3) - self.bg_render
        sim_img *= 255.0
        sim_img += self.background
        
        if not shadow:
            return np.clip(sim_img, 0, 255).astype(np.uint8)

        # Light positions in pixel coordinate
        light_type = "spot"
        light_r = [-40, -120, 130.0]
        light_g = [-40, 360, 130.0]
        light_b = [500, 120, 100.0]

        # Generate shadow from rgb channel respectively
        shadow_g = 1 - (1 - planar_shadow(light_g, hMap_smoothed, light_type)) * (1 - contact_mask)
        shadow_b = 1 - (1 - planar_shadow(light_b, hMap_smoothed, light_type)) * (1 - contact_mask)
        shadow_r = 1 - (1 - planar_shadow(light_r, hMap_smoothed, light_type)) * (1 - contact_mask)

        sim_img[:,:,0] *= np.clip(shadow_b + 0.65, 0, 1)
        sim_img[:,:,1] *= np.clip(shadow_r + 0.65, 0, 1)
        sim_img[:,:,2] *= np.clip(shadow_g + 0.65, 0, 1)

        return np.clip(sim_img, 0, 255).astype(np.uint8)