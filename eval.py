import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
import configargparse

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'phototourism'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'test_train'])
    parser.add_argument('--img_wh', nargs="+", type=int, default=[1368, 913],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    # for phototourism
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_cache', default=False, action="store_true",
                        help='whether to use ray cache (make sure img_downscale is the same)')

    # original NeRF parameters
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')

    # NeRF-W parameters
    parser.add_argument('--N_vocab', type=int, default=100,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance (NeRF-A)')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--encode_t', default=False, action="store_true",
                        help='whether to encode transient object (NeRF-U)')
    parser.add_argument('--N_tau', type=int, default=16,
                        help='number of embeddings for transient objects')
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='minimum color variance for each ray')
    parser.add_argument('--uw_model', default=False, action="store_true",
                        help='whether to use underwater model')
    parser.add_argument('--z_input', default=False, action="store_true",
                        help='whether to input z to BS and D in underwater model'),
    parser.add_argument('--uw_model_trans', default=False, action="store_true",
                        help='whether to use underwater model')
    parser.add_argument('--ndc', default=False, action="store_true",
                        help='whether to use ndc')
    parser.add_argument('--transient_uw', default=False, action="store_true",
                        help='whether to put the underwater model in the transient instead of the static')
    parser.add_argument('--uw_nerf', default=False, action="store_true",
                        help='whether to use uw_nerf model')
    parser.add_argument('--no_atten', default=False, action="store_true",
                        help='whether to remove attenuation')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--video_format', type=str, default='mp4',
                        choices=['gif', 'mp4'],
                        help='video format, gif or mp4')

    return parser.parse_args()


def check_image_weights(ret_dict,x,y,save_dir,i,h,w):

    x = np.long(x)
    y = np.long(y)
    z_vals = ret_dict['z_vals_fine'].view(h, w, -1).cpu().numpy()
    weights = ret_dict['weights_fine'].view(h, w, -1).cpu().numpy()
    pred_img = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
    alpha = ret_dict['alphas_fine'].view(h, w, -1).cpu().numpy()
    if 'BS' in ret_dict:
        fig, axBS = plt.subplots(2, 1)
        BS = ret_dict['BS'].view(h, w, -1,3).cpu().numpy()
        color = 'tab:red'
        axBS[0].set_xlabel('z vals')
        axBS[0].set_ylabel('BS', color=color)
        axBS[0].plot(z_vals[x, y, :], BS[x, y, :], color=color,label=['r', 'g','b'])
        axBS[0].tick_params(axis='y', labelcolor=color)
        axBS[1].plot(y, x, marker='v', color='red'), axBS[1].imshow(pred_img), fig.savefig(
            os.path.join(save_dir, f'BS_img_{i}_{x}_{y}.png'), bbox_inches='tight'), plt.close()
         # plt.figure(), plt.plot(z_vals[x,y, :], BS[x,y, :])  \
         #    ,plt.savefig(os.path.join(save_dir, f'BS_img_{i}_{x}_{y}.png'), bbox_inches='tight'),plt.close()
    density = ret_dict['sigma_fine'].view(h, w, -1).cpu().numpy()
    transmittance = ret_dict['transmittance_fine'].view(h, w, -1).cpu().numpy()
    fig, ax1 = plt.subplots(3,1)
    color = 'tab:red'
    ax1[0].set_xlabel('z vals')
    ax1[0].set_ylabel('density', color=color)
    ax1[0].plot(z_vals[x,y, :], density[x,y, :], color=color)
    ax1[0].tick_params(axis='y', labelcolor=color)

    ax2 = ax1[0].twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('weights', color=color)  # we already handled the x-label with ax1
    ax2.plot(z_vals[x,y, :], weights[x,y, :], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    # fig.savefig(os.path.join(save_dir, f'weights_img_{i}_{x}_{y}.png'), bbox_inches='tight')
    # plt.close()

    # fig, ax1 = plt.subplots()
    ax3 = ax1[1]
    color = 'tab:red'
    ax3.set_xlabel('z vals')
    ax3.set_ylabel('alpha', color=color)
    ax3.plot(z_vals[x,y, :], alpha[x,y, :], color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax3.set_ylabel('transmittance', color=color)  # we already handled the x-label with ax1
    ax3.plot(z_vals[x,y, :], transmittance[x,y, :], color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    # fig.savefig(os.path.join(save_dir, f'transmittance_img_{i}_{x}_{y}.png'), bbox_inches='tight')
    ax5 = ax1[2]


    ax5.plot(y, x, marker='v', color='red'), ax5.imshow(pred_img), fig.savefig(os.path.join(save_dir, f'target_img_{i}_{x}_{y}.png'), bbox_inches='tight'),plt.close(),\
    # plt.figure(), plt.plot(z_vals[x,y, :], density[x,y, :])  \
    #     ,plt.savefig(os.path.join(save_dir, f'density_img_{i}_{x}_{y}.png'), bbox_inches='tight'),plt.close(),
    # plt.figure(), plt.plot(z_vals[x,y, :], weights[x,y, :]) ,
    # plt.savefig(os.path.join(save_dir, f'weights_img_{i}_{x}_{y}.png'), bbox_inches='tight')
    # plt.close()
    # plt.figure(), plt.plot(z_vals[x, y, :], transmittance[x, y, :]),
    # plt.savefig(os.path.join(save_dir, f'transmittance_img_{i}_{x}_{y}.png'), bbox_inches='tight')
    # plt.close()
    # plt.figure(), plt.plot(z_vals[x,y, :], alpha[x,y, :])  \
    #     ,plt.savefig(os.path.join(save_dir, f'alpha_img_{i}_{x}_{y}.png'), bbox_inches='tight'),plt.close(),



@torch.no_grad()
def batched_inference(no_atten,uw_nerf,models, embeddings,
                      rays, ts, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,
                      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        ts[i:i+chunk] if ts is not None else None,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        uw_nerf= uw_nerf,
                        no_atten = no_atten,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    cuda = torch.device('cuda:1')
    kwargs = {'root_dir': args.root_dir,
              'split': args.split}
    if args.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(args.img_wh)
    else:
        kwargs['img_downscale'] = args.img_downscale
        kwargs['use_cache'] = args.use_cache
        kwargs['uw_model'] = args.uw_model
        # kwargs['uw_model_trans'] = args.uw_model_trans
    dataset = dataset_dict[args.dataset_name](**kwargs)
    scene = os.path.basename(args.root_dir.strip('/'))

    embedding_xyz = PosEmbedding(args.N_emb_xyz-1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir-1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if args.encode_a:
        embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).cuda(cuda) if args.uw_model else torch.nn.Embedding(args.N_vocab, args.N_a).cuda(cuda) #TODO: chang
        load_ckpt(embedding_a, args.ckpt_path, model_name='embedding_a')
        # zeroedEmbedding = torch.nn.parameter.Parameter(torch.zeros_like(embedding_a.weight))
        # zeroedEmbedding[:,0:12]= torch.nn.parameter.Parameter(torch.zeros_like(zeroedEmbedding[:,0:12]))
        # embedding_a.weight = zeroedEmbedding
        embeddings['a'] = embedding_a
    if args.encode_t:
        embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).cuda(cuda)
        load_ckpt(embedding_t, args.ckpt_path, model_name='embedding_t')
        # zeroedEmbeddingT = torch.nn.parameter.Parameter(torch.zeros_like(embedding_t.weight))
        # embedding_t.weight = zeroedEmbeddingT
        embeddings['t'] = embedding_t
    if args.uw_model_trans:
        embedding_b = torch.nn.Embedding(args.N_vocab, 6).cuda(cuda)
        load_ckpt(embedding_b, args.ckpt_path, model_name='embedding_b')
        embeddings['b'] = embedding_b
    # nerf_coarse = NeRF('coarse',
    #                     in_channels_xyz=6*args.N_emb_xyz+3,
    #                     in_channels_dir=6*args.N_emb_dir+3).cuda()
    if args.uw_model or args.transient_uw:
        nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6 * args.N_emb_xyz + 3,
                                in_channels_dir=6 * args.N_emb_dir + 3,
                                encode_appearance=args.encode_a,
                                in_channels_a=args.N_a,
                                encode_transient=args.encode_t,
                                in_channels_t=args.N_tau,
                                beta_min=args.beta_min,
                                uw_model=args.uw_model,uw_model_trans=args.uw_model_trans,transient_uw=args.transient_uw).cuda(cuda)
    elif args.uw_nerf:
        nerf_coarse = NeRFUw('coarse',
                                in_channels_xyz=6 * args.N_emb_xyz + 3,
                                in_channels_dir=6 * args.N_emb_dir + 3,
                                encode_appearance=args.encode_a,
                                in_channels_a=args.N_a,
                                encode_transient=args.encode_t,
                                in_channels_t=args.N_tau,
                                beta_min=args.beta_min,input_z=args.z_input).cuda(cuda)


    else:
        # nerf_coarse = NeRF('coarse',
        #                 in_channels_xyz=6*args.N_emb_xyz+3,
        #                 in_channels_dir=6*args.N_emb_dir+3,ndc=args.ndc).cuda(cuda)

        nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6 * args.N_emb_xyz + 3,
                                in_channels_dir=6 * args.N_emb_dir + 3,
                                encode_appearance=args.encode_a,
                                in_channels_a=args.N_a,
                                encode_transient=args.encode_t,
                                in_channels_t=args.N_tau,
                                beta_min=args.beta_min,
                                uw_model=args.uw_model,uw_model_trans=args.uw_model_trans,transient_uw=args.transient_uw,ndc=args.ndc).cuda(cuda)


    # models = {'coarse': nerf_coarse}
    if args.uw_nerf:
        nerf_fine = NeRFUw('fine',
                     in_channels_xyz=6*args.N_emb_xyz+3,
                     in_channels_dir=6*args.N_emb_dir+3,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau,
                     beta_min=args.beta_min,input_z=args.z_input,ndc=args.ndc).cuda(cuda)
    else:
        nerf_fine = NeRF('fine',
                     in_channels_xyz=6*args.N_emb_xyz+3,
                     in_channels_dir=6*args.N_emb_dir+3,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau,
                     beta_min=args.beta_min,uw_model=args.uw_model,uw_model_trans=args.uw_model_trans,transient_uw=args.transient_uw).cuda(cuda)

    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}


    imgs, psnrs = [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    kwargs = {}
    # define testing poses and appearance index for phototourism
    # if args.dataset_name == 'phototourism' and args.split == 'test':
    #     # define testing camera intrinsics (hard-coded, feel free to change)
    #     # dataset.test_img_w, dataset.test_img_h = args.img_wh
    #     # dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6) # fov=60 degrees
    #     # dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
    #     #                            [0, dataset.test_focal, dataset.test_img_h/2],
    #     #                            [0,                  0,                    1]])
    #     dataset.test_img_w, dataset.test_img_h = args.img_wh
    #     # dataset.test_focal = dataset.test_img_w / 2   # fov=60 degrees
    #     dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6) # fov=60 degrees
    #     dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
    #                                [0, dataset.test_focal, dataset.test_img_h/2],
    #                                [0,                  0,                    1]])
    #     if scene == 'brandenburg_gate':
    #         # select appearance embedding, hard-coded for each scene
    #         dataset.test_appearance_idx = 1123 # 85572957_6053497857.jpg
    #         N_frames = 30*3
    #         dx = np.linspace(0, 0.03, N_frames)
    #         dy = np.linspace(0, -0.1, N_frames)
    #         dz = np.linspace(0, 0.5, N_frames)
    #         # define poses
    #         dataset.poses_test = np.tile(dataset.poses_dict[1123], (N_frames, 1, 1))
    #         for i in range(N_frames):
    #             dataset.poses_test[i, 0, 3] += dx[i]
    #             dataset.poses_test[i, 1, 3] += dy[i]
    #             dataset.poses_test[i, 2, 3] += dz[i]
    #     elif scene == 'katzaa':
    #         # select appearance embedding, hard-coded for each scene
    #         dataset.test_appearance_idx = 8  # 85572957_6053497857.jpg
    #         N_frames = 30 * 3
    #         dx = np.linspace(0, 0.03, N_frames)
    #         dy = np.linspace(0, -0.1, N_frames)
    #         dz = np.linspace(0, 0.5, N_frames)
    #         # define poses
    #         dataset.poses_test = np.tile(dataset.poses_dict[8], (N_frames, 1, 1))
    #         for i in range(N_frames):
    #             dataset.poses_test[i, 0, 3] += dx[i]
    #             dataset.poses_test[i, 1, 3] += dy[i]
    #             dataset.poses_test[i, 2, 3] += dz[i]
    #     else:
    #         raise NotImplementedError
    #     kwargs['output_transient'] = False

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']
        results = batched_inference( args.no_atten,args.uw_nerf,models, embeddings, rays.cuda(cuda), ts.cuda(cuda),
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    **kwargs)

        if args.dataset_name == 'blender':
            w, h = args.img_wh
        else:
            w, h = sample['img_wh']
        std_weights = torch.std(results['weights_fine'].view(h, w,args.N_samples+ args.N_importance),dim=2).cpu().numpy()
        if args.uw_nerf:
            BS = results['rgbBS_fine'].view(h, w, 3).cpu().numpy()
            BS_ = (BS * 255).astype(np.uint8)

            imageio.imwrite(os.path.join(dir_name, f'BS_{i:03d}.png'), BS_)
            direct = results['direct_fine'].view(h, w, 3).cpu().numpy()
            direct_ = (direct * 255).astype(np.uint8)

            imageio.imwrite(os.path.join(dir_name, f'direct_{i:03d}.png'), direct_)
        if args.encode_t and not (args.uw_nerf):
            transient_rgb_map = results['transient_rgb_map'].view(h, w, 3).cpu().numpy()
            transient_rgb_map_ = (transient_rgb_map * 255).astype(np.uint8)

            imageio.imwrite(os.path.join(dir_name, f'transient_{i:03d}.png'), transient_rgb_map_)
            transient_depth = results['depth_fine_transient'].view(h,w).cpu().numpy()

            imageio.imwrite(os.path.join(dir_name, f'transient_depth_{i:03d}.png'), transient_depth)
            bs_betas = results['beta'].view(h, w).cpu().numpy()
            imageio.imwrite(os.path.join(dir_name, f'betas{i:03d}.png'), (bs_betas * 255))
        if args.uw_nerf:
            density_var = results['sigma_var_fine'].view(h,w).cpu().numpy()
            imageio.imwrite(os.path.join(dir_name, f'density_var{i:03d}.png'), (density_var * 255))
            bs_betas = results['bs_betas_fine'].view(h,w).cpu().numpy()
            imageio.imwrite(os.path.join(dir_name, f'bs_betas_fine{i:03d}.png'), (bs_betas * 255))




        density_var = results['sigma_var_fine'].view(h, w).cpu().numpy()
        imageio.imwrite(os.path.join(dir_name, f'density_var{i:03d}.png'), (density_var * 255))
        depth = results['depth_fine'].view(h,w).cpu().numpy()
        imageio.imwrite(os.path.join(dir_name, f'weights_std{i:03d}.png'), (std_weights * 255))
        imageio.imwrite(os.path.join(dir_name, f'depth{i:03d}.png'), (depth))
        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred*255).astype(np.uint8)

        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            fig, ax = plt.subplots()
            ax.imshow(img_pred_)

            # select point
            yroi = plt.ginput(0, 0)

            [check_image_weights(results, y, x, dir_name, i,h,w) for [x, y] in yroi]
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
    if args.dataset_name == 'blender' or \
      (args.dataset_name == 'phototourism' and args.split == 'test'):
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.{args.video_format}'),
                        imgs, fps=30)
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')