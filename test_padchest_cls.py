import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda().float()
    return tensor


class PatchDataset(Dataset):
    def __init__(self, path_to_images, fold='test', sample=0, transform=None):
        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv('../padchest/padchest_test_test.csv')
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(frac=sample, random_state=42)
            print('subsample the training set with ratio %f' % sample)
        self.df = self.df.set_index('ImageID')
        self.PRED_LABEL = self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.path_to_images, self.df.index[idx])
            )
        image = image.convert('RGB')
        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
        if self.transform:
            image = self.transform(image)

        return (image, label)


class Tester():
    def __init__(self, log_folder, conf, cls_conf, device='cuda:0'):
        self.log_folder = log_folder

        self.model = LitModel(conf)
        state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
        self.model.load_state_dict(state['state_dict'], strict=False)
        self.model.ema_model.to(device)

        self.cls_model = ClsModel(cls_conf)
        state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt',
                            map_location='cpu')
        print('latent step:', state['global_step'])
        self.cls_model.load_state_dict(state['state_dict'], strict=False)
        self.cls_model.to(device)

    def test(self, loader):
        label_list, pred_list = [], []
        with torch.no_grad():
            self.model.ema_model.eval()
            self.cls_model.ema_classifier.eval()
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                latent = self.model.ema_model.encoder(data)
                latent = self.cls_model.normalize(latent)
                output = self.cls_model.ema_classifier.forward(latent)                
                pred = torch.sigmoid(output)
                
                label_list.append(label.cpu().numpy())
                pred_list.append(pred.cpu().numpy())
        
        pred = np.squeeze(np.array(pred_list))
        label = np.squeeze(np.array(label_list))
        np.save(os.path.join(self.log_folder, 'y_pred.npy'), pred)
        np.save(os.path.join(self.log_folder, 'y_true.npy'), label)


if __name__ == '__main__':
    conf = padchest256_autoenc()
    cls_conf = padchest256_autoenc_cls()
    tester = Tester(conf=conf, cls_conf=cls_conf, log_folder=f'checkpoints/{cls_conf.name}/')
    te_dataset = PatchDataset(path_to_images='../padchest/test',
                            fold='test',
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)
    tester.test(te_loader)