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

@torch.no_grad()
def demo_data(model, image1, image2):
    H, W = image1.shape[2:]
    output = model.calc_flow(image1, image2)
    flow = output['flow'][-1]
    flow = flow[0].permute(1, 2, 0).cpu().numpy()
    flow_vis = flow_to_image(flow, convert_to_bgr=True)
    return flow, flow_vis

def prepare_image(img):
    img = np.array(img).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img[None].to(device)
    return img

##############################################
# load the model
device = 'cuda'
model = ViTWarpV8(args)
load_ckpt(model, args.ckpt)
model = model.cuda()
model.eval()
wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)

# ########################################
# load the video
if WINDOWS:
    video_path = "C:\\Users\\ulloa\\OneDrive\\Desktop\\Practicas\\projectes\\k-coefficient\\repsol_BV1000330_20.mp4"
else:
    video_path = "/home/jlpp/SOAT_b/test/turtuid/videos/repsol_BV1000330_20.mp4"

path = f"results/"
batch_size = 100
last_one = []
flow_list = []
viz_list = []

for idx, (batch, times) in enumerate(batch_iterator(video_path, batch_size)):
    batch = last_one + batch # pairing last one from previous batch with first one of the current 
    for i in range(len(batch) - 1):
        img1 = prepare_image(batch[i])
        img2 = prepare_image(batch[i + 1])
        flow, flow_viz = demo_data(wrapped_model, img1, img2)
        flow_list.append(flow)
        viz_list.append(flow_viz)
    last_one = [batch[-1]]

    batch_id = (idx + 1)* batch_size
    flow_data = {
        'times': times,
        'flow': flow_list
    }
    np.savez_compressed(f"{path}flow_{batch_id}.npz", flow_data)
    for j, img in enumerate(viz_list):
        img_id = (idx*batch_size + j) + 1
        cv2.imwrite(f"{path}flow_{img_id}.jpg", img)
