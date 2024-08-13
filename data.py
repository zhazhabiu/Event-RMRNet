import torch
import os
from torchvision.transforms import Resize
import numpy as np
from tqdm import tqdm

def xyxy_cwh(gts, ratio):
    cx, cy = (gts[:, 0] + gts[:, 2])//2, (gts[:, 1] + gts[:, 3])//2
    w, h = gts[:, 2] - gts[:, 0], gts[:, 3] - gts[:, 1] 
    return np.stack((cx, cy, w*ratio, h*ratio), 1)

class ImDataset(object):
    def __init__(self, dataset_name, im_size=(180, 240), is_train=True):
        root = f'./dataset/{dataset_name}/'
        self.image_file = root+'images.txt'
        self.events_file = root+'events.txt'
        self.gts_dir = root+'locations/'
        self.tau = 1.2
        self.im_size = im_size
        self.dt = 6.6*1e-3
        self.tslices = []
        self.images = []
        resize = Resize((64, 64))
        with open(self.image_file, 'r') as f:
            for line in f.readlines():
                time_e, file = line.split(' ')
                file = file.strip('\n').split('/')[-1]
                self.tslices.append(float(time_e))
                self.images.append(file)
        self.events = torch.from_numpy(np.loadtxt(self.events_file, dtype=np.float32, delimiter=" ")) # t, x, y, p
        self.prev_gts = []
        self.next_gts = []
        self.ids = []
        self.im_sequence = []
        self.TSLTD_patches = []
        # Deal with each TSLTD
        for item in tqdm(range(0, len(self.images)-1)):
            gts = np.loadtxt(os.path.join(self.gts_dir, self.images[item][:-4]+'.txt'), delimiter=',').reshape(-1, 5)
            gts1 = np.loadtxt(os.path.join(self.gts_dir, self.images[item+1][:-4]+'.txt'), delimiter=',').reshape(-1, 5)
            track_id = gts[:, -1]
            track_id1 = gts1[:, -1]
            gts_ = xyxy_cwh(gts[:, :-1], self.tau) # [N, 4] - xmin ymin xmax ymax
            ids, ids1 = np.nonzero(track_id[:, None] == track_id1[None, :])
            start_ind = torch.nonzero(self.events[:, 0] >= self.tslices[item], as_tuple=True)[0][0]
            end_ind = torch.nonzero(self.events[:, 0] >= self.tslices[item+1], as_tuple=True)[0][0]
            Ms = self.getEventFrames(self.events[start_ind:end_ind])
            for i, j in zip(ids, ids1):
                try:
                    half_w, half_h = gts_[i, 2]/2, gts_[i, 3]/2
                    xmin, ymin, xmax, ymax = max(0, round(gts_[i, 0]-half_w)), max(0, round(gts_[i, 1]-half_h)), \
                        min(self.im_size[1], round(gts_[i, 0]+half_w)), min(self.im_size[0], round(gts_[i, 1]+half_h))
                    patch = Ms[:, ymin:ymax, xmin:xmax]
                    patch = resize(patch.unsqueeze(0))
                    self.TSLTD_patches.append(patch.squeeze(0))
                    self.prev_gts.append(gts[i, :-1])
                    self.next_gts.append(gts1[j, :-1])
                    self.ids.append(track_id[i])
                    self.im_sequence.append(item+1)
                except Exception as e:
                    print(e)
                    print(gts[i, :-1], gts_[i], max(0, round(gts_[i, 1]-gts_[i, 3]/2)), min(self.im_size[0], round(gts_[i, 1]+gts_[i, 3]/2)),
                        max(0, round(gts_[i, 0]-gts_[i, 2]/2)), min(self.im_size[1], round(gts_[i, 0]+gts_[i, 2]/2)))
        if is_train:
            print(f'Train Dataset {dataset_name} is done.')
        else:
            print(f'Test Dataset {dataset_name} is done.')
        
    def getEventFrames(self, indices):
        '''
        Return [np.array([2*N, h, w])], N means the number of TSLTD frames.
        '''
        Ms = []
        prev_id = 0
        t_interval = torch.arange(indices[0, 0], indices[-1, 0], self.dt)
        for t in t_interval[1:-1]:
            ids = torch.nonzero(indices[:, 0] >= t, as_tuple=True)[0][0]
            F = torch.zeros((2, *self.im_size), dtype=torch.float32)
            ts = indices[prev_id, 0]
            F[indices[prev_id:ids, 3].long(), indices[prev_id:ids, 2].long(), indices[prev_id:ids, 1].long()] = torch.round(255*(indices[prev_id:ids, 0]-ts)/self.dt)
            prev_id = ids
            Ms.append(F)
        return torch.cat(Ms, 0)

    def __getitem__(self, item):
        return self.TSLTD_patches[item], torch.from_numpy(self.prev_gts[item]), torch.from_numpy(self.next_gts[item]), torch.tensor(self.ids[item], dtype=torch.long), torch.tensor(self.im_sequence[item], dtype=torch.long)

    def __len__(self):
        return len(self.TSLTD_patches)
    