import numpy as np
import random
import os
import torch
import argparse
from yacs.config import CfgNode as CN
from model import RMRNet
from evaluation import *
from utils import *

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
torch.backends.cudnn.benchmark = False

image_size = (180, 240)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(cfg, model, train_loader, optimizer):
    logger = setup_logger("RMRNet.trainer")
    logger.info('Start training...')
    max_epoch = cfg.SOLVER.MAX_EPOCH
    model.train()
    criterion = torch.nn.MSELoss().to(device)
    best_loss = 1e+6
    for epoch in range(max_epoch):
        losses = []
        for images, prev_gts, next_gts, _, _ in train_loader:
            images = images.to(device)
            prev_gts = prev_gts.to(device) # [B, 4] xmin ymin xmax ymax
            next_gts = next_gts.to(device) # [B, 4]

            dx, dy, theta, sx, sy = model(images) # [B, 1] * 5
            X = prev_gts[:, [0, 2]] + dx
            Y = prev_gts[:, [1, 3]] + dy
            cosTheta = torch.cos(theta)
            sinTheta = torch.sin(theta)
            prev_gts[:, [0, 2]] = (cosTheta*X - sinTheta*Y)*sx
            prev_gts[:, [1, 3]] = (sinTheta*X + cosTheta*Y)*sy
            
            loss = criterion(prev_gts, next_gts)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        losses = torch.tensor(losses).mean()
        logger.info(f"Epoch: {epoch}, loss: {losses}, max mem: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}")
        if best_loss > losses:
            best_loss = losses
            torch.save(model.state_dict(), f'{cfg.OUTPUT_DIR}/model_best.pth')
        torch.save(model.state_dict(), f'{cfg.OUTPUT_DIR}/model_last.pth')
 
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
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    
    # data loader
    train_loader, _ = make_dataloader(cfg, is_train=True)
    model = RMRNet()
    if cfg.MODEL.WEIGHT != "":
        print(f'Loading pretrained model {cfg.MODEL.WEIGHT}.')
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHT))
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total learnable parameters {num_learnable_params}.')
    model.to(device)
    optimizer = optimizer = torch.optim.Adam(model.parameters(),
                                            lr=cfg.SOLVER.BASE_LR)
    train(cfg, model, train_loader, optimizer)
