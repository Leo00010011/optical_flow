import cv2
import numpy as np
from typing import Generator
import os
import torch

from model.vitwarp_v8 import ViTWarpV8
from utils.utils import load_ckpt, coords_grid, bilinear_sampler
from utils.flow_viz import flow_to_image
from inference_tools import InferenceWrapper


WINDOWS = os.name == 'nt'
# for the statistical test get ideas from error metrics in optical flow
# from args need
class DotDict(dict):
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
args = {
'iters':  5,#iterations of the optimization of the optical flow
'var_min':  0,#lamping value
'var_max':  10,#clamping value
'name':  'common',#
"image_size": [432, 960],
"scale": 0,
"dav2_backbone": "vits",
"network_backbone": "vits",
"algorithm": "vitwarp",
}

if WINDOWS:
    args['ckpt'] = 'model\\weights\\tar-c-t.pth', # path to the checkpoint#
else:
    args['ckpt'] = 'model/weights/tar-c-t.pth'

args = DotDict(args)


def batch_iterator(video_path: str, batch_size: int):
    '''
    A frame iterator that reads the data in batches


    Parameters
    ----------
    video_path : path to the video.
    batch_size: number of frames of the batch.

    Returns
    -------
        a list of numpy array with the frames in RGB.

    '''
    cap = cv2.VideoCapture(video_path)
    start = 0
    eof = False
    while True:
        if eof:
            break
        batch = []
        timestamp_ms = []
        for _ in range(batch_size):
            ret, frame = cap.read()

            if not ret:
                eof = True
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(img_rgb)
            timestamp_ms.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        end = start + len(batch)
        print(f'Frame: {start} -> {end}')
        start = end
        yield batch, timestamp_ms
    cap.release()

def demo_data(model, image1, image2):
    path = f"results/"
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    cv2.imwrite(f"{path}image1.jpg", cv2.cvtColor(image1[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{path}image2.jpg", cv2.cvtColor(image2[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    output = model.calc_flow(image1, image2)
    for i in range(len(output['flow'])):
        flow= output['flow'][i]
        flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite(f"{path}flow_{i}.jpg", flow_vis)

# load the model
# device = 'cuda'
# model = ViTWarpV8(args)
# load_ckpt(model, args.ckpt)
# model = model.cuda()
# model.eval()
# wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)
# # load the images
if WINDOWS:
    video_path = "C:\\Users\\ulloa\\OneDrive\\Desktop\\Practicas\\projectes\\k-coefficient\\repsol_BV1000330_20.mp4"
else:
    video_path = "/home/jlpp/SOAT_b/test/turtuid/videos/repsol_BV1000330_20.mp4"

for pair in batch_iterator(video_path, 2):
    break
# transform the images
[img1, img2] = pair[0]
cv2.imshow('test1',img1)
cv2.imshow('test2', img2)
cv2.waitKey(-1)
cv2.destroyAllWindows()
# img1 = np.array(img1).astype(np.uint8)[..., :3]
# img2 = np.array(img2).astype(np.uint8)[..., :3]
# img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
# img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
# img1 = img1[None].to(device)
# img2 = img2[None].to(device)
# # get the optical flow in compressed format
# demo_data(model, img1, img2)
# get the optical flow visualization