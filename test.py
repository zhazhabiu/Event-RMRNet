import numpy as np
import random
import os
import torch
import argparse
from yacs.config import CfgNode as CN
from model import RMRNet
from evaluation import *
from utils import *
from tqdm import tqdm

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)

image_size = (180, 240)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def unwarp_events(indice, box):
    '''
    To get the original trajectories composed of events.
    box - np.array((xyxy))
    trajectory - t(s), x, y, p
    '''
    take_id = np.logical_and(np.logical_and(indice[:, 1] >= box[0], indice[:, 1] <= box[2]), \
                                np.logical_and(indice[:, 2] >= box[1], indice[:, 2] <= box[3]))
    trajectory = indice[take_id]
    return trajectory
       
def test(cfg, model, test_loaders, datasets):
    logger = setup_logger("RMRNet.test")
    logger.info('Start test...')
    model.eval()
    criterion = torch.nn.MSELoss().to(device)
    for i, test_loader in enumerate(test_loaders):
        print(f'{cfg.DATASETS.TEST[i]} testing...')
        prev_locations = {}
        losses = []
        events = datasets[i].events.numpy()
        tslices = datasets[i].tslices
        for images, prev_gts, next_gts, ids, img_id in tqdm(test_loader):
            images = images.to(device)
            prev_gts = prev_gts.to(device) # [B, 4] xmin ymin xmax ymax
            next_gts = next_gts.to(device) # [B, 4]
            with torch.no_grad():
                dx, dy, theta, sx, sy = model(images) # [1] * 5
            if ids in prev_locations.keys():
                prev_gts[0] = prev_locations[ids]
            X = prev_gts[:, [0, 2]] + dx
            Y = prev_gts[:, [1, 3]] + dy
            cosTheta = torch.cos(theta)
            sinTheta = torch.sin(theta)
            prev_gts[:, [0, 2]] = (cosTheta*X - sinTheta*Y)*sx
            prev_gts[:, [1, 3]] = (sinTheta*X + cosTheta*Y)*sy
            prev_gts[:, 0] = torch.clip(prev_gts[:, 0], min=torch.tensor(0))
            prev_gts[:, 1] = torch.clip(prev_gts[:, 1], min=torch.tensor(0))
            prev_gts[:, 2] = torch.clip(prev_gts[:, 2], max=torch.tensor(image_size[1]))
            prev_gts[:, 3] = torch.clip(prev_gts[:, 3], max=torch.tensor(image_size[0]))
            losses.append(criterion(prev_gts, next_gts))
            prev_locations[ids] = prev_gts.clone() # [1, 4]
            ts = tslices[img_id-1]
            ts1 = tslices[img_id]
            a = np.nonzero(events[:, 0] >= ts)[0][0]
            b = np.nonzero(events[:, 0] >= ts1)[0]
            if len(b) == 0:
                b = len(events)-1
            else:
                b = b[0]     
            prev_gts = prev_gts.detach().cpu().numpy().flatten()
            with open(f'./{cfg.DATASETS.TEST[i]}_tracking_box/{ids.item():d}.txt', 'a+') as f:
                np.savetxt(f, np.c_[np.insert(prev_gts, 0, ts1)[None, :]], fmt='%f', delimiter=',') # t(s), xxyy
            trajectory = unwarp_events(events[a:b], prev_gts)
            with open(f'./{cfg.DATASETS.TEST[i]}_tracking_res/{ids.item():d}.txt', 'a+') as f:
                np.savetxt(f, np.c_[trajectory], fmt='%f', delimiter=',') # t(s), x, y, p
        logger.info(f"Testing  loss: {torch.tensor(losses).cpu().numpy().mean()}")

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="./config.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = CN.load_cfg(f)
    for dataset in cfg.DATASETS.TEST:
        if not os.path.exists(f'{dataset}_tracking_res'):
            os.makedirs(f'{dataset}_tracking_res')
        if not os.path.exists(f'{dataset}_tracking_box'):
            os.makedirs(f'{dataset}_tracking_box')
    
    # data loader
    test_loaders, datasets = make_dataloader(cfg, is_train=False)
    model = RMRNet()
    if cfg.MODEL.TEST_WEIGHT != "":
        print(f'Loading pretrained model {cfg.MODEL.TEST_WEIGHT}.')
        model.load_state_dict(torch.load(cfg.MODEL.TEST_WEIGHT))
    model.to(device)
    test(cfg, model, test_loaders, datasets)
