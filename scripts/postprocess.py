import argparse
from glob import glob
from multiprocessing import Pool
from pathlib import Path
import os
import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def process(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(path.replace('matted', 'seg_mask'), cv2.IMREAD_UNCHANGED)[:, :, 2:3]
    mask = (mask == 90)
    mask = (mask > 0).astype(np.int32)
    mask = ndimage.median_filter(mask, size=5)
    mask = (ndimage.binary_dilation(mask, iterations=3) > 0).astype(np.uint8)
    alpha = img[:, :, 3:4] / 1.0
    img *= (1 - mask)
    cv2.imwrite(path, img.astype(np.uint8))
    alpha = ndimage.median_filter(alpha, size=5)
    alpha = np.where(mask, np.zeros_like(alpha), alpha)
    alpha[alpha > 127.5] = 255
    alpha = alpha.astype(np.uint8)
    img[:, :, 3:4] = alpha
    cv2.imwrite(path.replace('matted', 'images'), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='Output', required=True)
    args = parser.parse_args()
    actor = args.i

    Path(f'{actor}/images').mkdir(parents=True, exist_ok=True)

    images = sorted(glob(actor + '/matted/*'))

    with Pool(processes=8) as pool:
        for result in tqdm(pool.imap_unordered(process, images), total=len(images)):
            pass
