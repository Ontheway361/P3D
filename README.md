this is the pytorch implementation for 'P3D'.

this project can be used as video understanding/recognition

paper : Learning Spatio-Temporal Representation with Pseudo-3D Residual,(2017-ICCV)

reference github : https://github.com/qijiezhao/pseudo-3d-pytorch

tensorboard check : tensorboard --logdir=./run --port=8008


refer result : - Action recognition(mean accuracy on UCF101):

modality/model | RGB | Flow | Fusion
---|---|---|---
P3D199 (Sports-1M) | 88.5%| -|-
P3D199 (Kinetics) | 91.2% | 92.4%| 98.3%

- Action localization(mAP on Thumos14):

#### steps: perframe+watershed
Step | perframe | localization
---|---|---
P3D199(Sports-1M | 0.451 | 0.25
P3D199(Kinetics) | 0.569(fused) | 0.307
