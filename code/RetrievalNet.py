from utils import read_json
from PIL import Image
from Models import RetrievalNet
import yaml
import argparse
from ColorTransfer import color_tranfer
import tqdm
from datasets.shape_datasets import ShapeDataset
from datasets.query_datasets import QueryDataset
import jittor.transform as transform
import jittor as jt
from jittor import nn


# from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

import os


class Retrieval(object):
    '''
    training
    testing
    loading
    saving
    '''
    def __init__(self, config):
        self.cfg =config
        self.retrieval_net = RetrievalNet(self.cfg)
        lr = float(self.cfg.trainer.lr)
        beta1 = float(self.cfg.trainer.beta1)

        beta2 = float(self.cfg.trainer.beta2)
        self.opt = nn.Adam(self.retrieval_net.parameters(), lr=lr, betas=(beta1, beta2))

        self.normal_tf = transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.tau = self.cfg.data.tau
        self.size = self.cfg.data.pix_size
        self.dim = self.cfg.models.z_dim
        self.view_num = self.cfg.data.view_num

        self.epoch = 0
        self.it = 0
        self.loading(self.cfg.models.pre_trained_path)
        

        self.logs_root = './logs/'+self.cfg.data.name
        self.best_acc = 0
        # writer
        # writer = SummaryWriter(os.path.join(self.logs_root, 'curves'))
        

    def training(self):
        # device = self.device
        cfg = self.cfg
        query_loader = QueryDataset(cfg=cfg).set_attrs(batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True)
        shape_loader = ShapeDataset(cfg=cfg).set_attrs(batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True)
        
        total_epoch = cfg.trainer.epochs
        epoch = self.epoch
        it = self.it
        while epoch < total_epoch:
            # self.testing()
            self.retrieval_net.train()

            pbar = tqdm.tqdm(query_loader)
            # print(len(pbar))
            for meta in pbar:
                mask_img = meta['mask_img']
                rendering_img = meta['rendering_img']
                cats = meta['cat']
                instances = meta['instance']
                query_img = meta['query_img']
                
                # doing color transfer
                seq = jt.misc.randperm(query_img.shape[0])
                style_img = query_img[seq]
                transfer_img = color_tranfer(style_img, query_img)
                query_img = self.normal_tf(transfer_img).detach()


                # get Instance and Category level label
                inst_list = []  # record the unique idx for each model
                inst_index = [] # instance level idx
                idx_list = []   # using for torch.cat
                bs = len(cats)

                for ii in range(len(cats)):
                    tmp_cat = cats[ii]
                    tmp_inst = instances[ii]
                    try:
                        # model already existed
                        idx = inst_list.index((tmp_cat, tmp_inst))
                        inst_index.append(idx)
                    except ValueError:
                        inst_index.append(len(inst_list))
                        inst_list.append((tmp_cat, tmp_inst))
                        idx_list.append(ii)

                rendering_img = jt.concat([rendering_img[idx:idx+1] for idx in idx_list], dim=0)

                
                while not len(inst_list) == bs:
                    try:
                        shape_meta = next(iter_shape)
                    except:
                        iter_shape = iter(shape_loader)
                        shape_meta = next(iter_shape)
                    
                    tmp_cats = shape_meta['labels']['cat']
                    tmp_insts = shape_meta['labels']['instance']
                    tmp_reinderings = shape_meta['rendering_img']
                    # tmp_rendering_idx_list = [] # using this list to cat ebds in new datasets_iter
                    tmp_rendering_list = []

                    for ii in range(len(tmp_cats)):
                        tmp_cat = tmp_cats[ii]
                        tmp_inst = tmp_insts[ii]
                        try:
                            # model already existed
                            idx = inst_list.index((tmp_cat, tmp_inst))
                        except ValueError:
                            inst_list.append((tmp_cat, tmp_inst))
                            tmp_rendering_list.append(tmp_reinderings[ii:ii+1]) 

                        if len(inst_list) == bs:
                            break
                    if not len(tmp_rendering_list) == 0:
                        tmp_reindering = jt.concat(tmp_rendering_list, dim=0) 
                        rendering_img = jt.concat([rendering_img, tmp_reindering], dim=0)
                # bsx12x224x224

                cats_list = []
                shape_cats_index = []
                image_cats_index = []
                # cats_index = [] # category level idx
                for ii, items in enumerate(inst_list):
                    tmp_cat, tmp_inst = items
                    try:
                        idx = cats_list.index(tmp_cat)
                        shape_cats_index.append(idx)
                    except ValueError:
                        shape_cats_index.append(len(cats_list))
                        cats_list.append(tmp_cat)
                    tmp_cat, tmp_inst = inst_list[inst_index[ii]]
                    idx = cats_list.index(tmp_cat)
                    image_cats_index.append(idx)


                inst_label = jt.int(jt.array(inst_index).view(-1,1))
                InstsMat = jt.zeros((inst_label.shape[0], bs)).scatter_(1, inst_label, jt.ones((inst_label.shape[0], bs))).t()
        
                shape_cats_label = jt.int(jt.array(shape_cats_index))
                image_cats_label = jt.int(jt.array(image_cats_index))

                shape_cats_labels = shape_cats_label.unsqueeze(1).repeat(1, bs)
                image_cats_labels = image_cats_label.unsqueeze(0).repeat(bs, 1)
                CatsMat = jt.float(shape_cats_labels==image_cats_labels)

                # ######## [ end ] ######## [create no repeat rendering batch data ] ########
                # InstsMat      bs, bs  (shape, image)
                # CatsMat       bs, bs  (shape, image)

                # bsx4x224x224
                mquery = jt.concat((query_img, mask_img), dim=1)
                query_image_ebd, queried_rendering_ebd = self.retrieval_net(mquery, rendering_img)
                qi_ebd = query_image_ebd                # bs, bs, 128
                qr_ebd = queried_rendering_ebd          # bs, bs, 128     (shape, image, 128)


                prod_mat = (qi_ebd * qr_ebd).sum(dim=2)
                ProdMat = jt.exp(prod_mat * (1/self.tau))  # bs, bs   (shape, image)
                
                ProdMat_sum = ProdMat.sum(dim=0)
                # Instance Loss
                loss_inst_ = (ProdMat * InstsMat).sum(dim=0) / ProdMat_sum
                loss_inst = -jt.log(loss_inst_)
                loss_inst = loss_inst.mean() 

                # Category Loss
                if not (len(cats_list) == 1):
                    CatsMat_exc = CatsMat
                    pos_num = CatsMat_exc.sum(dim=0)

                    pos_num[pos_num==0]=1 # In some cases, pos_num = 0  -->> nan
                    SumMat = ProdMat.sum(dim=0).view(1, -1).repeat(bs, 1) # SumMat excluding InstMat
                    ExcProdMat = ProdMat * CatsMat_exc / SumMat

                    ExcProdMat[ExcProdMat==0] = 1
                    # ExcProdMat[ExcProdMat<1e-5] = 1e-5
                    loss_cats_ = -jt.log(ExcProdMat).sum(dim=0)/pos_num
                    loss_cats = loss_cats_[loss_cats_ != 0]
                    loss_cats = loss_cats.mean()

                    loss = loss_inst + 0.2*loss_cats
                    loss_cats_item = loss_cats.item()
                else:
                    loss = loss_inst
                    loss_cats_item = 0

                loss_item = loss.item()
                loss_inst_item = loss_inst.item()

                self.opt.step(loss)

                it += 1

                info_dict = {'loss': '%.3f' %(loss_item, ), 'loss_inst': '%.3f' %(loss_inst_item, ), 'loss_cats': '%.3f' %(loss_cats_item, ) }
                pbar.set_postfix(info_dict)
                pbar.set_description('Epoch: %d, Iter: %d ' % (epoch, it))
                

            epoch += 1
            self.it = it
            self.epoch = epoch
            # self.testing() 
            if epoch % 10 == 0:
                self.testing()
 

    def testing(self):
        # device = self.device
        cfg = self.cfg
        # writer = self.writer
        self.retrieval_net.eval()

        is_aug = cfg.setting.is_aug
        cfg.setting.is_aug = False

        shape_loader = ShapeDataset(cfg=cfg).set_attrs(batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, drop_last=False)
        cfg.setting.is_aug = is_aug


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
                query_img = query_transformer(Image.open(os.path.join(cfg.data.data_dir, info['img'])))
                mask_img = mask_transformer(Image.open(os.path.join(cfg.data.data_dir, info['mask'])))

                query = jt.concat((query_img, mask_img), dim=0)
                # query = query.unsqueeze(dim=0)
                query = jt.unsqueeze(query, dim=0)
                query_ebd = self.retrieval_net.get_query_ebd(query)

                
                query_ebd = query_ebd.repeat(bs, 1, 1)
                _, weights = self.retrieval_net.attention_query(query_ebd, shape_ebd)
                queried_rendering_ebd = jt.nn.bmm(weights, shape_ebd)
                qr_ebd = queried_rendering_ebd
                qi_ebd = query_ebd
                prod_mat = (qi_ebd * qr_ebd).sum(dim=2)
                max_idx = prod_mat.argmax(dim=0)

                # print(max_idx[0].data[0]) 
                pr_cats = shape_cats_list[max_idx[0].data[0]]
                pr_inst = shape_inst_list[max_idx[0].data[0]]
                
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
            out_info.append('%s: inst: %d, cats: %d, total: %d\n' %(keys, inst_num, cats_num, num))
        
        out_infos = ''.join(out_info)
        print(out_infos)

        final_acc = total_acc/total_num 
        print(final_acc)
        
        if final_acc > self.best_acc:
            self.best_acc = final_acc
            paths_list=self.cfg.models.pre_trained_path.split('/')[:-1]
            paths_list.append(self.cfg.data.name+'_best.pt')
            paths = '/'.join(paths_list)
            self.saving(paths=paths)
        
        print('best acc: %.3f' %(self.best_acc, ))


    def loading(self, paths=None):
        cfg = self.cfg
        if paths == None or not os.path.exists(paths):
            # init   
            model_dict =  self.retrieval_net.state_dict()
            res18_pre_path = os.path.join(cfg.models.pre_train_resnet_root, 'resnet18.pkl')
            res50_pre_path = os.path.join(cfg.models.pre_train_resnet_root, 'resnet50.pkl')
            save_model18 = jt.load(res18_pre_path)
            save_model50 = jt.load(res50_pre_path)

            # query encoder
            # conv1 and fc
            prefix = 'query_encoder.resnet'
            for keys in save_model50.keys():
                key_prefix = keys.split('.')[0]
                if key_prefix == 'conv1' or key_prefix == 'fc':
                    continue
                model_key = '%s.%s' %(prefix, keys)
                model_dict[model_key] = save_model50[keys]

            # rendering encoder
            # conv1 and fc
            prefix = 'rendering_encoder.resnet'
            for keys in save_model18.keys():
                key_prefix = keys.split('.')[0]
                if key_prefix == 'conv1' or key_prefix == 'fc':
                    continue
                model_key = '%s.%s' %(prefix, keys)
                model_dict[model_key] = save_model18[keys]

            self.retrieval_net.load_state_dict(model_dict)
            print('No ckpt! Init from ResNet18 for RenderingEncoder and ResNet50 for QueryEncoder')
      
        else:
            # loading
            ckpt = jt.load(paths)
            self.retrieval_net.load_state_dict(ckpt)
            print('loading %s successfully' %(paths))


    def saving(self, paths=None):
        cfg = self.cfg

        if paths == None:
            save_name = "epoch_{}_iter_{}.pt".format(self.epoch, self.it)
            save_path = os.path.join(cfg.save_dir, save_name)
            print('models %s saved!\n' %(save_name, ))
        else:
            save_path = paths      
            print('model paths %s saved!\n' %(paths, ))

        jt.save(self.retrieval_net.state_dict(), save_path)


if __name__ == '__main__':    
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

    retrieval = Retrieval(config)
    retrieval.training()

