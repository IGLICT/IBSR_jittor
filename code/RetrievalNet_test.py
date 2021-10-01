from typing import Tuple
from numpy.core.records import record
import jittor as jt
from datasets.query_datasets import QueryDataset 
from datasets.shape_datasets import ShapeDataset
import tqdm
import jittor.transform as transform
import os
# from tensorboardX import SummaryWriter
import warnings
from utils import read_json
from PIL import Image
from RetrievalNet import RetrievalNet
warnings.filterwarnings('ignore')
from Models import RetrievalNet

import numpy as np
import binvox_rw
from sklearn.metrics.pairwise import pairwise_distances

os.environ["CUDA_VISIBLE_DEVICES"]="3"

import shutil
from sklearn import manifold
import matplotlib.pyplot as plt
import time



import multiprocessing
from contextlib import contextmanager


def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
     between two unordered sets of points (the function is symmetric).
     Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.

    from: https://github.com/HaipengXiong/weighted-hausdorff-loss/blob/master/object-locator/losses.py
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res/2




class Retrieval(object):
    '''
    ColorTransfer
    load
    save
    '''
    def __init__(self, config):
        self.cfg =config
        self.retrieval_net = RetrievalNet(self.cfg)

        self.normal_tf = transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.size = self.cfg.data.pix_size
        self.dim = self.cfg.models.z_dim
        self.view_num = self.cfg.data.view_num

        self.loading(self.cfg.models.pre_trained_path)
        

    def loading(self, paths=None):
        if paths == None or not os.path.exists(paths): 
            print('No ckpt!')
            exit(-1)
        else:
            # loading
            ckpt = jt.load(paths)
            self.retrieval_net.load_state_dict(ckpt)
            print('loading %s successfully' %(paths))


    def test_steps(self):
        # 缩小一下 model 的大小，然后再计算
        '''
        1. 先按照 json 中的cat来分类，把同一类的聚集起来
        2. 按照 类 来遍历，将Top1的结果保存
        3. 根据需求，是否需要计算IoU和Haus; 如果需要这两个数字，则遍历Top1结果
        '''
        cfg = self.cfg
        self.retrieval_net.eval()

        # datasets for no model embeddings repeating training
        is_aug = cfg.setting.is_aug
        cfg.setting.is_aug = False
        shape_dataset = ShapeDataset(cfg=cfg)
        # shape_loader = torch.utils.data.DataLoader(dataset=shape_dataset, \
        #     batch_size=cfg.data.batch_size, shuffle=False, \
        #         drop_last=False, num_workers=cfg.data.num_workers)
        shape_loader = ShapeDataset(cfg=cfg).set_attrs(batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, drop_last=False)
        cfg.setting.is_aug = is_aug


        name_src = 'compcars'
        name_tar = 'compcars'
        roots = '../data'
        is_iou_haus = True
        roots_src = os.path.join(roots, name_src)
        roots_tar = os.path.join(roots, name_tar)
        

        shape_cats_list = []
        shape_inst_list = []
        shape_ebd_list = []
        pbar = tqdm.tqdm(shape_loader)
        for meta in pbar:
            with jt.no_grad():
                rendering_img = meta['rendering_img']
                cats = meta['labels']['cat']
                instances = meta['labels']['instance']
            
                rendering = rendering_img.view(-1, 1, self.size, self.size)
                rendering_ebds = self.retrieval_net.get_rendering_ebd(rendering).view(-1, self.view_num, self.dim)
                shape_cats_list += cats
                shape_inst_list += instances
                shape_ebd_list.append(rendering_ebds)
        
        shape_ebd = jt.concat(shape_ebd_list, dim=0) # num, 12, dim

        # test image
        json_dict = read_json(os.path.join(cfg.data.data_dir, cfg.data.test_json))
        # json_dict = json_dict[:10]
        # json_lenth = len(json_dict)
        query_transformer = transform.Compose([transform.ToTensor(), transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        mask_transformer = transform.Compose([transform.ToTensor(), transform.Normalize((0.5, ), (0.5, ))])
        bs = shape_ebd.shape[0]

        # sorted as category 
        cats_dict = {}
        for items in json_dict:
            cat = items['category']
            try:
                cats_dict[cat].append(items)
            except:
                cats_dict[cat] = []
                cats_dict[cat].append(items)

        iou_dict = {}
        haus_dict = {}
        top1_dict = {}
        topk_dict = {}
        category_dict = {}
        record_dict = {}

        for cat in cats_dict.keys():
            cats_list = cats_dict[cat]
            iou_dict[cat] = [0 for i in range(len(cats_list))]
            haus_dict[cat] = [0 for i in range(len(cats_list))]
            top1_dict[cat] = [0 for i in range(len(cats_list))]
            topk_dict[cat] = [0 for i in range(len(cats_list))]
            category_dict[cat] = [0 for i in range(len(cats_list))]
            record_dict[cat] = []

            with jt.no_grad():
                pbar = tqdm.tqdm(cats_list)                
                for i, info in enumerate(pbar):
                    # info = json_dict[i]
                    query_img = query_transformer(Image.open(os.path.join(cfg.data.data_dir, info['img'])))
                    mask_img = mask_transformer(Image.open(os.path.join(cfg.data.data_dir, info['mask'])))

                    query = jt.concat((query_img, mask_img), dim=0)
                    query = query.unsqueeze(dim=0)
                    query_ebd = self.retrieval_net.get_query_ebd(query)

                    
                    query_ebd = query_ebd.repeat(bs, 1, 1)
                    _, weights = self.retrieval_net.attention_query(query_ebd, shape_ebd)
                    queried_rendering_ebd = jt.nn.bmm(weights, shape_ebd)
                    qr_ebd = queried_rendering_ebd
                    qi_ebd = query_ebd
                    prod_mat = (qi_ebd * qr_ebd).sum(dim=2)
                    max_idx = prod_mat.argmax(dim=0)


                    pr_cats = shape_cats_list[max_idx[0]]
                    pr_inst = shape_inst_list[max_idx[0]]
                    
                    gt_cats = info['category']
                    gt_inst = info['model'].split('/')[-2]
           
                    if gt_cats == pr_cats:
                        category_dict[cat][i] = 1

                        if gt_inst == pr_inst:
                            top1_dict[cat][i] = 1

                    record_dict[cat].append((pr_cats, pr_inst, gt_cats, gt_inst))

                     

                    max_idx = prod_mat.view(-1).topk(dim=0, k=10)[1]
                    for kk in range(10):
                        pr_cats = shape_cats_list[max_idx[kk]]
                        pr_inst = shape_inst_list[max_idx[kk]]
                        if gt_cats == pr_cats and gt_inst == pr_inst:
                            topk_dict[cat][i] = 1
                            break
                    

        # basic output: top1, top10, cats, total number
        out_info = []
        total_info = {}
        for cat in cats_dict.keys():
            length = len(top1_dict[cat])
            out_info.append('%s: top1: %d, top10: %d, cats: %d, top1_rt: %.3f, top10_rt: %.3f, cats_rt: %.3f, total num: %d\
                        ' %(cat, sum(top1_dict[cat]), sum(topk_dict[cat]), sum(category_dict[cat]),
                                sum(top1_dict[cat])/length, sum(topk_dict[cat])/length, 
                                sum(category_dict[cat])/length, length))
           
            total_info[cat] = []
            total_info[cat].append(sum(top1_dict[cat]))
            total_info[cat].append(sum(topk_dict[cat]))
            total_info[cat].append(sum(category_dict[cat]))
            total_info[cat].append(length)
        for msg in out_info:
            print(msg)
        
        total_top1 = sum([total_info[cat][0] for cat in cats_dict.keys()])
        total_top10 = sum([total_info[cat][1] for cat in cats_dict.keys()])
        total_cats = sum([total_info[cat][2] for cat in cats_dict.keys()]) 
        total_length = sum([total_info[cat][3] for cat in cats_dict.keys()])  
        print('%s: top1: %d, top10: %d, cats: %d, top1_rt: %.3f, top10_rt: %.3f, cats_rt: %.3f, total num: %d\n\
                        ' %('[Total]', total_top1, total_top10, total_cats,
                                total_top1/total_length, total_top10/total_length, 
                                total_cats/total_length, total_length))


        return record_dict
        if is_iou_haus:
            print('>>>>>>>>[calculating haus and iou]<<<<<<<<')
            # In order to speed up, we use multiprocessing here
            
            cal_iou_haus(record_dict, roots_src, roots_tar, iou_dict, haus_dict, cats_dict)

            out_info = []
            total_info = {}
            for cat in cats_dict.keys():
                length = len(iou_dict[cat])
                out_info.append('%s: haus: %.4f, iou: %.4f,\n\
                            ' %(cat, sum(haus_dict[cat])/length, sum(iou_dict[cat])/length, length))
            
                total_info[cat] = []
                total_info[cat].append(sum(haus_dict[cat]))
                total_info[cat].append(sum(iou_dict[cat]))
                total_info[cat].append(length)
            for msg in out_info:
                print(msg)
            
            total_haus = sum([total_info[cat][0] for cat in cats_dict.keys()])
            total_iou = sum([total_info[cat][1] for cat in cats_dict.keys()])
            total_length = sum([total_info[cat][2] for cat in cats_dict.keys()]) 
            print('%s:  haus: %.4f, iou: %.4f: %d\n\
                            ' %('[Total]', total_haus/total_length, 
                                    total_iou/total_length, total_length))



@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def func(zips):
    ki, record = zips
    name_src = 'stanfordcars'
    name_tar = 'stanfordcars'
    roots = '../data'
    roots_src = os.path.join(roots, name_src)
    roots_tar = os.path.join(roots, name_tar) 

    # pr_cats, pr_inst, gt_cats, gt_inst = record_dict['car'][ki] 
    pr_cats, pr_inst, gt_cats, gt_inst = record
    src_binvox_path = os.path.join(roots_src, 'model_std_bin128', gt_cats, '%s.binvox'%(gt_inst,))
    tar_binvox_path = os.path.join(roots_tar, 'model_std_bin128', pr_cats, '%s.binvox'%(pr_inst,))

    with open(src_binvox_path, 'rb') as f:
        src_bin = binvox_rw.read_as_3d_array(f).data
    
    with open(tar_binvox_path, 'rb') as f:
        tar_bin = binvox_rw.read_as_3d_array(f).data

    # IoU
    Iou_st = np.sum(src_bin & tar_bin) / np.sum((src_bin | tar_bin) + 1e-8)
    # Haus
    src_ptc_path = os.path.join(roots_src, 'model_std_ptc10k_npy', gt_cats, '%s.npy'%(gt_inst,))
    tar_ptc_path = os.path.join(roots_tar, 'model_std_ptc10k_npy', pr_cats, '%s.npy'%(pr_inst,))

    src_ptc = np.load(src_ptc_path)[:5000]
    tar_ptc = np.load(tar_ptc_path)[:5000]
    src_ptc = src_ptc/2
    tar_ptc = tar_ptc/2 

    Haus_st = averaged_hausdorff_distance(src_ptc, tar_ptc)
    if ki % 500 == 0:
        print(ki)

    return Haus_st, Iou_st 


if __name__ == '__main__':
    import yaml
    import argparse
    with open('./configs/stanfordcars.yaml', 'r') as f:
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


    config.models.pre_trained_path = './pre_trained/stanfordcars.pt'

    ret = Retrieval(config)
    record_dict = ret.test_steps()

    

    is_iou_haus = True
    if is_iou_haus:
        iou_dict = {}
        haus_dict = {}
        par_num = 20 
        for cat in record_dict.keys():
            nums = [i for i in range(len(record_dict[cat]))]
            records = [record_dict[cat][i] for i in range(len(record_dict[cat]))] 
            with poolcontext(processes=par_num) as pool:
                rt = pool.map(func, zip(nums, records))

            # print(rt)
            iou_dict[cat] = []
            haus_dict[cat] = []
            for haus, iou in rt:
                iou_dict[cat].append(iou) 
                haus_dict[cat].append(haus)

        
        out_info = []
        total_info = {}
        for cat in record_dict.keys():
            length = len(iou_dict[cat])
            out_info.append('%s: haus: %.4f, iou: %.4f,\
                        ' %(cat, sum(haus_dict[cat])/length, sum(iou_dict[cat])/length))
        
            total_info[cat] = []
            total_info[cat].append(sum(haus_dict[cat]))
            total_info[cat].append(sum(iou_dict[cat]))
            total_info[cat].append(length)
        for msg in out_info:
            print(msg)
        
        total_haus = sum([total_info[cat][0] for cat in record_dict.keys()])
        total_iou = sum([total_info[cat][1] for cat in record_dict.keys()])
        total_length = sum([total_info[cat][2] for cat in record_dict.keys()]) 
        print('%s:  haus: %.4f, iou: %.4f: %d\n\
                        ' %('[Total]', total_haus/total_length, 
                                total_iou/total_length, total_length))
    # pass



