import torch
from collections import OrderedDict

checkpoint_path = 'checkpoints/resnet50/model_best.pth.tar'
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
state_dict2 = OrderedDict()
for k, v in state_dict.items():
    if 'tracked' in k:
        continue
    k = k.replace('module.', '')
    state_dict2[k] = v
checkpoint2_path = 'checkpoints/resnet50/model_best_fix.pth.tar'
checkpoint['state_dict'] = state_dict2
torch.save(checkpoint, checkpoint2_path)
