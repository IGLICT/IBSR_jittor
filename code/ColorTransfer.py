# import the necessary packages
import jittor as jt
if jt.has_cuda:
    jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。

# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    # mask = (rgb > .04045).type(torch.float)
    mask = jt.float(rgb > .04045)

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = jt.concat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = jt.concat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = jt.maximum(rgb,jt.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    # mask = (rgb > .0031308).type(torch.float)
    mask = jt.float(rgb > .0031308)
    # if(rgb.is_cuda):
    #     mask = mask.cuda()
    # mask = mask.to(device)

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = jt.array((0.95047, 1., 1.08883))[None,:,None,None]
    # sc = jt.array((0.95047, 1., 1.08883))
    # if(xyz.is_cuda):
    #     sc = sc.cuda()
    # sc = sc.to(device)
    
    xyz_scale = xyz/sc

    # mask = (xyz_scale > .008856).type(torch.float)
    mask = jt.float(xyz_scale > .008856)

    # if(xyz_scale.is_cuda):
    #     mask = mask.cuda()
    # mask = mask.to(device)

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = jt.concat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def lab2xyz(lab):
    # device = lab.device
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)

    # z_int = torch.max(torch.Tensor((0,)).to(device), z_int)
    z_int = jt.maximum(jt.Var((0,)), z_int)

    out = jt.concat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    # mask = (out > .2068966).type(torch.float)
    mask = jt.float(out > .2068966)
    # if(out.is_cuda):
    #     mask = mask.cuda()
    # mask = mask.to(device)
    

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    # sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = jt.array((0.95047, 1., 1.08883))[None,:,None,None]
    # sc = sc.to(out.device)

    out = out*sc

    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2xyz')
        # embed()

    return out

def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-50)/100
    ab_rs = lab[:,1:,:,:]/100
    out = jt.concat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

def lab2rgb(lab_rs):
    l = lab_rs[:,[0],:,:]*100 + 50
    ab = lab_rs[:,1:,:,:]*100
    lab = jt.concat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2rgb')
        # embed()
    return out


# the testing function in main show that 
# clip = true  and  preserve_paper = False will get better results
def color_tranfer(source, target, clip=True, preserve_paper=False):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    source: NumPy array
        OpenCV image in BGR color space (the source image)
    target: NumPy array
        OpenCV image in BGR color space (the target image)
    clip: Should components of L*a*b* image be scaled by np.clip before 
        converting back to BGR codlor space?
        If False then components will be min-max scaled appropriately.
        Clipping will keep target image brightness truer to the input.
        Scaling will adjust image brightness to avoid washed out portions
        in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
        layed out in original paper? The method does not always produce
        aesthetically pleasing results.
        If False then L*a*b* components will scaled using the reciprocal of
        the scaling factor proposed in the paper.  This method seems to produce
        more consistently aesthetically pleasing results 

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)

    --------
    original source: https://github.com/jrosebr1/color_transfer/
    converted into pytorch version
    """
    source = source * 255
    target = target * 255

    source_ = source
    source = rgb2lab(source)
    target = rgb2lab(target)

     
    MeanSrc = source.mean(dims=(2, 3))
    src_size=1
    for i in source.shape[2:]:
        src_size *= i
    # jt.unsqueeze(jt.unsqueeze(source, -1), -1)
    # src_out = (jt.unsqueeze(jt.unsqueeze(MeanSrc, -1), -1) - source).sqr().sum(dims=(2, 3)) 
    src_out = (MeanSrc.unsqueeze(-1).unsqueeze(-1) - source).sqr().sum(dims=(2, 3)) 
    src_out = src_out/(src_size-1)
    StdSrc = src_out.maximum(1e-6).sqrt()
    
    # StdSrc = source.std(dims=(2, 3))

    MeanTar = target.mean(dims=(2, 3))
    tar_size=1
    for i in target.shape[2:]:
        tar_size *= i
    tar_out = (MeanTar.unsqueeze(-1).unsqueeze(-1) - target).sqr().sum(dims=(2, 3)) 
    tar_out = tar_out/(tar_size-1)
    StdTar = tar_out.maximum(1e-6).sqrt()

    # StdTar = target.std(dims=(2, 3))
    target -= MeanTar.unsqueeze(-1).unsqueeze(-1) 

    if preserve_paper:
        target =  (StdTar/StdSrc).unsqueeze(-1).unsqueeze(-1) * target
    else:
        target =  (StdSrc/StdTar).unsqueeze(-1).unsqueeze(-1) * target

    target += MeanSrc.unsqueeze(-1).unsqueeze(-1) 
    target = lab2rgb(target)

    if clip:
        transfers = jt.clamp(target, 0, 255)
        transfers = transfers/255
    else:
        bmin = target.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        bmax = target.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        transfers = (target - bmin) / (bmax - bmin)
    return transfers
    


if __name__ == '__main__':
    import yaml
    import argparse
    from datasets.query_datasets import QueryDataset
    import jittor.transform as transform

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
    cfg = dict2namespace(config)

 
    # query_dataset = QueryDataset(cfg=cfg)
    # query_loader = torch.utils.data.DataLoader(dataset=query_dataset, \
    #     batch_size=cfg.data.batch_size, shuffle=False, \
    #         drop_last=True, num_workers=cfg.data.num_workers)
    query_loader = QueryDataset(cfg=cfg).set_attrs(batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, drop_last=True)

    # device = torch.device('cuda:3')
    # cter = ColorTransfer(device)

    topil = transform.ToPILImage()
    with jt.no_grad():
        for meta in query_loader:
            mask_img = meta['mask_img']
            # rendering_img = meta['rendering_img']
            # cats = meta['cat']
            # instances = meta['instance']
            query_img = meta['query_img']

            # seq = torch.randperm(query_img.shape[0])
            seq = [i for i in range(mask_img.shape[0])][::-1]
            style_img = query_img[seq]
            style_mask_img = mask_img[seq]
            transfer_img = color_tranfer(style_img, query_img)


            # ##### dataset debug ######
            bs = query_img.shape[0]
            for ii in range(bs):
                
                jt.transpose(transfer_img[ii], [1,2,0])
                q_img = topil(jt.transpose(query_img[ii], [1,2,0]))
                s_img = topil(jt.transpose(style_img[ii], [1,2,0]))
                tf_img = topil(jt.transpose(transfer_img[ii], [1,2,0]))


                q_img.save('./debug/%d-q_img.png' %(ii, ))
                s_img.save('./debug/%d-s_img.png' %(ii, ))
                tf_img.save('./debug/%d-tf_img.png' %(ii, ))

            debug = 258
            break

