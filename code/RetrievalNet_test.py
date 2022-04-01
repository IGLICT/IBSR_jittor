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

import yaml
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


def modified_averaged_hausdorff_distance(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    with jt.no_grad():
        xt = jt.float32(x.astype(np.float32)).unsqueeze(1)
        yt = jt.float32(y.astype(np.float32)).unsqueeze(0)
        differences = xt -yt 
        # differences = x.unsqueeze(1).cuda() - y.unsqueeze(0).cuda()
        distances = jt.sum(differences**2, -1).sqrt()

        num = distances.shape[0] + distances.shape[1]
        res = jt.min(distances, dim=0).sum() + jt.min(distances, dim=1).sum() 
        res = float(res)
    return res/num


def cal_IoU_and_Haus(cfg, record_dict):
    iou_dict = {}
    haus_dict = {}

    name_src = cfg.data.name
    roots = cfg.data.root_dir

    if cfg.mode == 'shapenet':
        if cfg.data.name == 'pix3d':
            name_tar = 'shapenet4'
        else:
            name_tar = 'shapenetcars'
    else:
        name_tar = cfg.data.name

    
    cnt = 0
    for cat in record_dict.keys():
        # nums = [i for i in range(len(record_dict[cat]))]
        records = [record_dict[cat][i] for i in range(len(record_dict[cat]))]
        iou_dict[cat] = []
        haus_dict[cat] = []

        for record in records:
            pr_cats, pr_inst, gt_cats, gt_inst = record

            roots_src = os.path.join(roots, name_src)
            roots_tar = os.path.join(roots, name_tar) 
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

            src_ptc = np.load(src_ptc_path)[:10000]/2
            tar_ptc = np.load(tar_ptc_path)[:10000]/2

            cnt += 1
            if cnt % 500 == 0:
                print(cnt)
            Haus_st = modified_averaged_hausdorff_distance(src_ptc, tar_ptc)

            iou_dict[cat].append(Iou_st) 
            haus_dict[cat].append(Haus_st)

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
    print('%s:  haus: %.4f, iou: %.4f || %d\n\
                    ' %('[Total]', total_haus/total_length, 
                            total_iou/total_length, total_length))



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


    def test_simple(self):
        # device = self.device
        cfg = self.cfg
        self.retrieval_net.eval()

        # datasets for no model embeddings repeating training
        cfg.setting.is_aug = False
        shape_loader = ShapeDataset(cfg=cfg).set_attrs(batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, drop_last=False)


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
        json_dict = read_json(os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.test_json))
        json_lenth = len(json_dict)
        query_transformer = transform.Compose([transform.ToTensor(), transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        mask_transformer = transform.Compose([transform.ToTensor(), transform.ImageNormalize((0.5, ), (0.5, ))])
        bs = shape_ebd.shape[0]

        total_dict = {}
        acc_cats_dict = {}
        acc_inst_dict = {}
        with jt.no_grad():
            pbar = tqdm.tqdm(range(json_lenth))
            for i in pbar:
                info = json_dict[i]
                query_img = query_transformer(Image.open(os.path.join(cfg.data.root_dir, cfg.data.name, info['img'])))
                mask_img = mask_transformer(Image.open(os.path.join(cfg.data.root_dir, cfg.data.name, info['mask'])))

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


                pr_cats = shape_cats_list[int(max_idx[0])]
                pr_inst = shape_inst_list[int(max_idx[0])]
                
                gt_cats = info['category']
                gt_inst = info['model'].split('/')[-2]

                try:
                    total_dict[gt_cats] = total_dict[gt_cats] + 1
                except:
                    total_dict[gt_cats] = 1

                if gt_cats == pr_cats:
                    try:
                        acc_cats_dict[gt_cats] = acc_cats_dict[gt_cats] + 1
                    except:
                        acc_cats_dict[gt_cats] = 1
                if gt_cats == pr_cats and gt_inst == pr_inst:
                    try:
                        acc_inst_dict[gt_cats] = acc_inst_dict[gt_cats] + 1
                    except:
                        acc_inst_dict[gt_cats] = 1

        total_num = 0
        total_acc = 0
        total_cat = 0
        out_info = []
        for keys in total_dict.keys():
            num = total_dict[keys]
            try:
                inst_num = acc_inst_dict[keys]
            except:
                inst_num = 0

            try:
                cats_num = acc_cats_dict[keys]
            except:
                cats_num = 0

            total_num += num
            total_acc += inst_num
            total_cat += cats_num
            out_info.append('%s: inst: %.3f, cats: %.3f, total: %d\n' %(keys, inst_num/num, cats_num/num, num))
        
        out_infos = ''.join(out_info)
        print(out_infos)

        print(total_acc/total_num)
        print(total_cat/total_num)


    def test_full(self):
        cfg = self.cfg
        self.retrieval_net.eval()

        # datasets for no model embeddings repeating training
        cfg.setting.is_aug = False
        shape_loader = ShapeDataset(cfg=cfg).set_attrs(batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, drop_last=False)
                                                                                                                                                                                                                                                    
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
        json_dict = read_json(os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.test_json))
        # json_dict = json_dict[:10]
        # json_lenth = len(json_dict)
        query_transformer = transform.Compose([transform.ToTensor(), transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        mask_transformer = transform.Compose([transform.ToTensor(), transform.ImageNormalize((0.5, ), (0.5, ))])
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


        top1_dict = {}
        topk_dict = {}
        category_dict = {}
        record_dict = {}

        for cat in cats_dict.keys():
            cats_list = cats_dict[cat]
            top1_dict[cat] = [0 for i in range(len(cats_list))]
            topk_dict[cat] = [0 for i in range(len(cats_list))]
            category_dict[cat] = [0 for i in range(len(cats_list))]
            record_dict[cat] = []

            with jt.no_grad():
                pbar = tqdm.tqdm(cats_list)                
                for i, info in enumerate(pbar):
                    # info = json_dict[i]
                    query_img = query_transformer(Image.open(os.path.join(cfg.data.root_dir, cfg.data.name, info['img'])))
                    mask_img = mask_transformer(Image.open(os.path.join(cfg.data.root_dir, cfg.data.name, info['mask'])))

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


                    pr_cats = shape_cats_list[int(max_idx[0])]
                    pr_inst = shape_inst_list[int(max_idx[0])]
                    
                    gt_cats = info['category']
                    gt_inst = info['model'].split('/')[-2]
           
                    if gt_cats == pr_cats:
                        category_dict[cat][i] = 1

                        if gt_inst == pr_inst:
                            top1_dict[cat][i] = 1

                    record_dict[cat].append((pr_cats, pr_inst, gt_cats, gt_inst))

                     

                    max_idx = prod_mat.view(-1).topk(dim=0, k=10)[1]
                    for kk in range(10):
                        pr_cats = shape_cats_list[int(max_idx[kk])]
                        pr_inst = shape_inst_list[int(max_idx[kk])]
                        if gt_cats == pr_cats and gt_inst == pr_inst:
                            topk_dict[cat][i] = 1
                            break
                    

        # basic output: top1, top10, cats, total number
        out_info = []
        total_info = {}
        for cat in cats_dict.keys():
            length = len(top1_dict[cat])
            out_info.append('%s: top1: %d, top10: %d, cats: %d, || top1_rt: %.3f, top10_rt: %.3f, cats_rt: %.3f, || total num: %d\
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
        print('%s: top1: %d, top10: %d, cats: %d, || top1_rt: %.3f, top10_rt: %.3f, cats_rt: %.3f, || total num: %d\n\
                        ' %('[Total]', total_top1, total_top10, total_cats,
                                total_top1/total_length, total_top10/total_length, 
                                total_cats/total_length, total_length))

        # self.cal_IoU_and_Haus(record_dict)
        return record_dict


    def _test_shapenet(self, ccat):
        cfg = self.cfg
        self.retrieval_net.eval()

        # datasets for no model embeddings repeating training
        cfg.setting.is_aug = False
        cats_ = ccat

        data_name = cfg.data.name
        data_render_path = cfg.data.render_path
        if data_name == 'pix3d':
            cfg.data.name = 'shapenet4'
        else:
            cfg.data.name = 'shapenetcars'
        cfg.data.render_path = 'rendering_shapenet%ss.pkl' %(cats_, )

        shape_loader = ShapeDataset(cfg=cfg).set_attrs(batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, drop_last=False)

        cfg.data.name = data_name
        cfg.data.render_path = data_render_path

        shape_cats_list = []
        shape_inst_list = []
        shape_ebd_list = []
        pbar = tqdm.tqdm(shape_loader)

        s_Flag = True
        for meta in pbar:
            with jt.no_grad():
                rendering_img = meta['rendering_img']
                cats = meta['labels']['cat']
                instances = meta['labels']['instance']
            
                rendering = rendering_img.view(-1, 1, self.size, self.size)
                rendering_ebds = self.retrieval_net.get_rendering_ebd(rendering).view(-1, self.view_num, self.dim)
                shape_cats_list += cats
                shape_inst_list += instances
                if s_Flag:
                    shape_ebd = rendering_ebds
                    s_Flag = False
                else:
                    shape_ebd = jt.concat([shape_ebd, rendering_ebds], dim=0) # num, 12, dim
                # shape_ebd_list.append(rendering_ebds)
                del rendering
                del rendering_img
                del rendering_ebds
                del meta
                jt.jittor_core.cleanup()
                jt.sync_all()
                jt.gc()

        
        # shape_ebd = jt.concat(shape_ebd_list, dim=0) # num, 12, dim

        # test image
        json_dict = read_json(os.path.join(cfg.data.root_dir, cfg.data.name, cfg.data.test_json))
        query_transformer = transform.Compose([transform.ToTensor(), transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        mask_transformer = transform.Compose([transform.ToTensor(), transform.ImageNormalize((0.5, ), (0.5, ))])
        bs = shape_ebd.shape[0]

        # sorted as category 
        cats_dict = {}
        for items in json_dict:
            cat = items['category']
            if not cat == ccat:
                continue
            try:
                cats_dict[cat].append(items)
            except:
                cats_dict[cat] = []
                cats_dict[cat].append(items)

        record_dict = {}

        for cat in cats_dict.keys():
            cats_list = cats_dict[cat]
            record_dict[cat] = []

            with jt.no_grad():
                pbar = tqdm.tqdm(cats_list)                
                for i, info in enumerate(pbar):
                    # info = json_dict[i]
                    query_img = query_transformer(Image.open(os.path.join(cfg.data.root_dir, cfg.data.name, info['img'])))
                    mask_img = mask_transformer(Image.open(os.path.join(cfg.data.root_dir, cfg.data.name, info['mask'])))

                    query = jt.concat((query_img, mask_img), dim=0)
                    query = query.unsqueeze(dim=0)
                    query_ebd = self.retrieval_net.get_query_ebd(query)

                    
                    query_ebd = query_ebd.repeat(bs, 1, 1)
                    _, weights = self.retrieval_net.attention_query(query_ebd, shape_ebd)
                    queried_rendering_ebd = jt.bmm(weights, shape_ebd)
                    qr_ebd = queried_rendering_ebd
                    qi_ebd = query_ebd
                    prod_mat = (qi_ebd * qr_ebd).sum(dim=2)
                    max_idx = prod_mat.argmax(dim=0)


                    pr_cats = shape_cats_list[int(max_idx[0])]
                    pr_inst = shape_inst_list[int(max_idx[0])]
                    
                    gt_cats = info['category']
                    gt_inst = info['model'].split('/')[-2]
           

                    record_dict[cat].append((pr_cats, pr_inst, gt_cats, gt_inst))
        
        return record_dict[ccat] 
                     

    def test_shapenet(self):
        record_dict = {}
        if self.cfg.data.name == 'pix3d':
            cat_list = ['bed', 'chair', 'sofa', 'table']
        else:
            cat_list = ['car', ]
        for cat in cat_list:
            record_dict[cat] = self._test_shapenet(cat)

        return record_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/stanfordcars.yaml", help="Path to (.yaml) config file."
    )
    parser.add_argument(
        "--mode", type=str, default="shapenet", help="testing mode: simple | full | shapenet."
    )

    configargs = parser.parse_args()

    with open(configargs.config, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
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
    setattr(config, 'mode', configargs.mode)
    Task = Retrieval(config)

    if config.mode == 'simple':
        Task.test_simple()
    elif config.mode == 'full':
        records = Task.test_full()
        cal_IoU_and_Haus(config, records)
    elif config.mode == 'shapenet':
        records = Task.test_shapenet()
        cal_IoU_and_Haus(config, records)
    else:
        pass


    


