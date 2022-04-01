import os
from jittor.misc import set_global_seed
import jittor.transform as transform
from jittor.dataset.dataset import Dataset
from PIL import Image
import pickle
import json
import time
import jittor as jt
if jt.has_cuda:
    jt.flags.use_cuda = 1 


class QueryDataset(Dataset):
    def __init__(self, cfg):
        super(QueryDataset, self).__init__()
        self.is_training = cfg.setting.is_training
        if cfg.setting.is_training:
            self.json_path = os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.training_json)
        else:
            self.json_path = os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.test_json)
 
        self.json_dict = self.read_json(self.json_path)
        self.data_dir = os.path.join(cfg.data.root_dir, cfg.data.name)

        crop_scale = (0.85, 0.95)
        self.aug = cfg.setting.is_aug
        self.query_transform = self.get_query_transform(cfg.data.pix_size, crop_scale, self.aug)
        self.mask_transform = self.get_mask_transform(cfg.data.pix_size, crop_scale, self.aug)
        self.rendering_transform = self.get_rendering_transform(cfg.data.pix_size, self.aug)
        self.view_num = cfg.data.view_num
        self.mask_dir = cfg.data.mask_dir

        render_path = os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.render_path)
        with open(render_path, 'rb') as f:
            self.dicts = pickle.load(f)

    def __getitem__(self, index):
        info = self.json_dict[index]
        cat = info['category']
        instance = info['model'].split('/')[-2]
        renderings = self.dicts[cat][instance]

        tmp_seed = int(time.time()) % 100000
        jt.set_global_seed(tmp_seed)
        query_img = self.query_transform(Image.open(os.path.join(self.data_dir, info['img'])).convert("RGB"))
        
        jt.set_global_seed(tmp_seed)
        mask_path_list = info['mask'].split('/')
        # if self.is_training:
        #     mask_path_list[0] = 'mask'
        # else:
        mask_path_list[0] = self.mask_dir
        mask_path = '/'.join(mask_path_list)
        mask_img = self.mask_transform(Image.open(os.path.join(self.data_dir, mask_path)))

        render_img = jt.concat([self.rendering_transform(renderings[vi]) for vi in range(self.view_num)], dim=0)

        # debug = 10
        # embeddings_name = info['category'] + '-' + info['model'].split('/')[-2]+'.npy'
        # embeddings = np.load(os.path.join(self.embeddings_dir, embeddings_name))
        return {'query_img': query_img, 'mask_img':mask_img, \
            'rendering_img':render_img, 'cat':cat, 'instance':instance }

    def __len__(self):
        return len(self.json_dict)

    @staticmethod
    def get_query_transform(rsize=(224, 224), crop_scale=(0.85, 0.95), is_aug=False):
        transform_list = []
        if is_aug:
            transform_list.append(transform.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=None, shear=None, resample=False, fillcolor=0))
            transform_list.append(transform.RandomResizedCrop(rsize, scale=crop_scale))
            transform_list.append(transform.RandomHorizontalFlip())

        transform_list += [transform.ToTensor()]
        # if not is_training:
        #     transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] 
        # we have add this 'Normalize' in train_retrieval.py
        # transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] 
        return transform.Compose(transform_list)

    @staticmethod
    def get_mask_transform(rsize=(224, 224), crop_scale=(0.85, 0.95), is_aug=False):
        transform_list = []
        if is_aug:
            transform_list.append(transform.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=None, shear=None, resample=False, fillcolor=0))
            transform_list.append(transform.RandomResizedCrop(rsize, scale=crop_scale))
            transform_list.append(transform.RandomHorizontalFlip())

        transform_list += [transform.ToTensor()]
        # transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] 
        return transform.Compose(transform_list)

    @staticmethod
    def get_rendering_transform(rsize=224, is_aug=False):
        transform_list = []
        # transform_list.append(transforms.Resize(rsize, method))
        if is_aug:
            transform_list.append(transform.RandomResizedCrop(rsize, scale=(0.65, 0.9)))
            transform_list.append(transform.RandomHorizontalFlip())

        transform_list += [transform.ToTensor()]
        transform_list += [transform.ImageNormalize((0.5, ), (0.5, ))] 
        return transform.Compose(transform_list)
    
    @staticmethod
    def read_json(mdir):
        with open(mdir, 'r') as f:
            tmp = json.loads(f.read())
        return tmp


class ImageDataset(Dataset):
    def __init__(self, cfg):
        super(ImageDataset, self).__init__()
        self.json_path = os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.test_json)
        self.json_dict = self.read_json(self.json_path)
        self.data_dir = os.path.join(cfg.data.root_dir, cfg.data.name)

        self.query_transform = transform.Compose([transform.ToTensor(), transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.mask_transform = transform.Compose([transform.ToTensor(), transform.ImageNormalize((0.5, ), (0.5, ))])


    def __getitem__(self, index):
        info = self.json_dict[index]
        cat = info['category']
        instance = info['model'].split('/')[-2]
        renderings = self.dicts[cat][instance]

        query_img = self.query_transform(Image.open(os.path.join(self.data_dir, info['img'])))
        mask_img = self.mask_transform(Image.open(os.path.join(self.data_dir, info['mask'])))
        
        return {'query_img': query_img, 'mask_img':mask_img, \
            'cat':cat, 'instance':instance }

    def __len__(self):
        return len(self.json_dict)
    
    @staticmethod
    def read_json(mdir):
        with open(mdir, 'r') as f:
            tmp = json.loads(f.read())
        return tmp
 

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

    # dataset = QueryDataset(cfg=config)
    # retrieval_loader = torch.utils.data.DataLoader(dataset=dataset, \
    #     batch_size=1, shuffle=True, \
    #         drop_last=True, num_workers=1)
    retrieval_loader = QueryDataset(cfg=config).set_attrs(batch_size=1, shuffle=True, num_workers=1, drop_last=True)

    for meta in retrieval_loader:
        
        ##### dataset debug ######
        with jt.no_grad():
            mask_img = meta['mask_img']
            embeddings = meta['rendering_img']
            cats = meta['cat']
            instances = meta['instance']
            query_img = meta['query_img']


            topil = transform.ToPILImage()
            masked_img = query_img*mask_img

    
            q_img = topil(jt.transpose(query_img[0], [1,2,0]))
            # s_img = topil(style_img[0])
            # tf_img = topil(transfer_img[0])
            
            mask_img_ = mask_img[0] 
            mask_img_[mask_img_>0] = 255
            m_img = topil(jt.transpose(mask_img_, [1,2,0])).convert('L')
            md_img = topil(jt.transpose(masked_img[0], [1,2,0]))
            # md_tf_img = topil(masked_tf_img[0])

            q_img.save('./debug/q_img.png')
            # s_img.save('./debug/s_img.png')
            # tf_img.save('./debug/tf_img.png')

            m_img.save('./debug/m_img.png')
            md_img.save('./debug/md_img.png')
            # md_tf_img.save('./debug/md_tf_img.png')
        ##### dataset debug ######

        debug = 10
        debug = 20
        continue