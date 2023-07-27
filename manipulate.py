import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

class PatchDataset(Dataset):
    def __init__(self, path_to_images, fold='test', sample=0, transform=None):
        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv('../padchest/padchest_test_test.csv')
        self.df = self.df[self.df['normal']==1]
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

        return (image, label, self.df.index[idx])
    

if __name__ == '__main__':
    device = 'cuda:3'
    conf = padchest256_autoenc()
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    cls_conf = padchest256_autoenc_cls()
    cls_model = ClsModel(cls_conf)
    state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt',
                        map_location='cpu')
    print('latent step:', state['global_step'])
    cls_model.load_state_dict(state['state_dict'], strict=False)
    cls_model.to(device)

    te_dataset = PatchDataset(path_to_images='../padchest/test',
                        fold='test',
                        transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(256),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
    
    np.random.seed(0)
    idxs = np.random.choice(len(te_dataset), 20)
    for idx in idxs:
        batch, label, filename = te_dataset[idx][0], te_dataset[idx][1], te_dataset[idx][2]
        batch = batch[None]
        cond = model.encode(batch.to(device))
        xT = model.encode_stochastic(batch.to(device), cond, T=250)
        for cls_id, label in tqdm(enumerate(te_dataset.PRED_LABEL)):
            cond2 = cls_model.normalize(cond)
            cond2 = cond2 + 0.6 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
            cond2 = cls_model.denormalize(cond2)
            # torch.manual_seed(1)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            img = model.render(xT, cond2, T=1000)
            ori = (batch + 1) / 2
            ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
            ax[1].imshow(img[0].permute(1, 2, 0).cpu())
            ax[0].axis('off')
            ax[1].axis('off')
            path = os.path.join('./imgs_manipulated/padchests/', filename)
            Path(path).mkdir(parents=True, exist_ok=True)
            _ = fig.savefig(os.path.join(path, label+'.png'), bbox_inches='tight', dpi=500)
            plt.close()