# Event-RMRNet
This is an unofficial implementation of 2020 AAAI《End-to-End Learning of Object Motion Estimation from Retinal Events for Event-Based Object Tracking》using Pytorch.


It's a neural network estimating motion.
The origin bounding boxes of objects are needed.


## Prerequisites
python >= 3.0  
numpy  
pandas  
openCV  
tqdm
torch (my version is torch181)
torchvision
argparse
yacs
logging


## Training
```Python
python main.py  
```

## Testing
```Python
python test.py  
```


## Data Preparation
Train/Test datasets should be put in *'./dataset/{name_of_dataset}/events.txt'*, or you can change the file reading path in *data.py*.

```
dataset/
│
├── dataset1/
│   ├── groundtruth
│   │   ├── framexxxx.txt (xmin, xmax, ymin, ymax, track_id)   
│   │   ...
│   ├── events.txt  (event stream -- t, x, y, p)
│   └── images.txt  (t, framexxx.png)
│
└── ...

```

## Tracking results
The boxes after estimaing motion is saved as txt in *'./{name_of_dataset}_tracking_box/\*.txt'*, as well as the corresponding trajectory events in *'./{name_of_dataset}_tracking_res/\*.txt'*