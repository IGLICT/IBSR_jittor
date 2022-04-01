import os
import jittor.transform as transform
from jittor.dataset.dataset import Dataset
from PIL import Image
import pickle
import jittor as jt
if jt.has_cuda:
    jt.flags.use_cuda = 1 

class ShapeDataset(Dataset):
    def __init__(self, cfg):
        super(ShapeDataset, self).__init__()

        render_path = os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.render_path)
        # self.dicts = np.load(render_path, allow_pickle=True).item()
        with open(render_path, 'rb') as f:
            self.dicts = pickle.load(f)
        self.labels = self.make_dataset(self.dicts)

        self.transform = self.get_transform(cfg.data.pix_size, cfg.setting.is_aug)
        self.view_num = cfg.data.view_num


    def __getitem__(self, index):
        labels = self.labels[index]
        cat = labels['cat']
        idx = labels['instance']
        renderings = self.dicts[cat][idx] # 12x224x224
        # debug = self.transform(Image.fromarray(renderings[0]))
        render_img = jt.concat([self.transform(renderings[vi]) for vi in range(self.view_num)], dim=0)

        return {'rendering_img': render_img, 'labels': labels}

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def make_dataset(dicts):
        labels = []
        for cat in dicts.keys():
            for idx in dicts[cat].keys():
                labels.append({'cat': cat, 'instance': idx})
        return labels


    @staticmethod
    def get_transform(rsize=224, is_aug=False, method=Image.BICUBIC):
        transform_list = []
        # transform_list.append(transforms.Resize(rsize, method))
        if is_aug:
            transform_list.append(transform.RandomResizedCrop(rsize, scale=(0.65, 0.9)))
            transform_list.append(transform.RandomHorizontalFlip())


        transform_list += [transform.ToTensor()]
        transform_list += [transform.ImageNormalize((0.5, ), (0.5, ))] 
        return transform.Compose(transform_list)



if __name__ =='__main__':
    import yaml
    import argparse
    with open('./configs/pix3d.yaml', 'r') as f:
        config = yaml.load(f)
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    config = dict2namespace(config)

    dataset = ShapeDataset(cfg=config)
    retrieval_loader = ShapeDataset(cfg=config).set_attrs(batch_size=5, shuffle=True, num_workers=2, drop_last=True)

    # for batch in retrieval_loader:
    #     debug = 10
    #     debug = 20
    #     continue
    # debug = 10