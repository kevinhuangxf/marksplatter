import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
import trimesh
from kiui.lpips import LPIPS

from core.unet import UNet
from core.options import Options
from core.gs import GaussianRenderer
from core.utils import get_disp_candidates, warp_with_pose_depth_candidates

# from dust3r.inference import inference
# from dust3r.model import AsymmetricCroCo3DStereo, AsymmetricCroCo3DStereoGaussian
# from dust3r.utils.image import load_images
# from dust3r.image_pairs import make_pairs
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# from dust3r.utils.device import to_cuda

from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from einops import rearrange, repeat
# from models.encoder.dino_wrapper import DinoWrapper
# from models.decoder.transformer import TriplaneTransformer
# from models.renderer.utils.renderer import generate_planes, sample_from_planes
# from src.tgs.models.renderer import GSLayer
# from kiui.cam import orbit_camera

# import mast3r.utils.path_to_dust3r
# from mast3r.model import AsymmetricMASt3R
# from dust3r.inference import inference
# from dust3r.utils.image import load_images
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# from dust3r.post_process import estimate_focal_knowing_depth  # noqa
# from croco.models.dpt_block import Interpolate

import cv2
# from dust3r.utils.geometry import inv, geotrf

# from core.provider_objaverse_dust3r import normalize_point_cloud
# from loss.depth_loss import get_depth_grad_loss, depth_grad_loss_func
from torchmetrics.image import StructuralSimilarityIndexMeasure

import einops
import sys
sys.path.append('/workspace/code/LGM')
# from core.splatt3r_utils.geometry import build_covariance
# from core.splatt3r_utils.sh_utils import RGB2SH

# from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

# from pytorch3d.loss import chamfer_distance

# Initialize the SSIM metric
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

# encoder
encoder_freeze: bool = False
encoder_model_name: str = 'facebook/dino-vitb16'

# triplane
encoder_feat_dim: int = 768
transformer_dim: int = 1024 
transformer_layers: int = 12
transformer_heads: int = 16
triplane_low_res: int = 32
triplane_high_res: int = 64
triplane_dim: int = 40

# # Mast3r
# device = 'cuda'
# model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
# mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

minmax_norm = lambda x: (x - x.min()) / (x.max() - x.min())

def calculate_model_size_millions(model):
    # Count the total number of parameters and convert to millions
    param_count = sum(p.numel() for p in model.parameters())
    param_count_millions = param_count / 1e6  # Parameters in millions
    return param_count_millions

class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        if self.opt.data_mode == 'dust3r':
            in_channels = 33
        else:
            in_channels = 9

        # unet
        self.unet = UNet(
            in_channels, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
            num_frames=self.opt.num_input_views, # new added
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        # # Dust3r
        # model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        # self.dust3r = AsymmetricCroCo3DStereoGaussian.from_pretrained(model_name)

        # # Mast3r
        # device = 'cuda'
        # model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        # self.mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        # self.mast3r_model.eval()

        # for param in self.mast3r_model.parameters():
        #     param.requires_grad = False

        if self.opt.use_depth_net:
            # Unet definition
            # down_channels = (64, 128, 256)
            # down_attention = (False, False, True)
            # mid_attention = True
            # up_channels = (256, 128, 64)
            # up_attention = (True, False, False)
            # in_channel = 17
            # out_channel = 1

            # self.unet_depth = UNet(
            #     in_channel, out_channel, 
            #     down_channels=down_channels,
            #     down_attention=down_attention,
            #     mid_attention=mid_attention,
            #     up_channels=up_channels,
            #     up_attention=up_attention
            # ).cuda()

            # The "DPTDepthModel" head
            feature_dim = 256
            last_dim = 32
            out_channels = 1
            self.depth_feat_head_1 = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(last_dim, out_channels, kernel_size=1, stride=1, padding=0)
            ).cuda()
            self.depth_feat_head_2 = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(last_dim, out_channels, kernel_size=1, stride=1, padding=0)
            ).cuda()


        if self.opt.use_geometry_net:
            num_depth_candidates = 128
            input_size: int = 128
            # Unet definition
            down_channels = (64, 128, 256)
            down_attention = (False, True, True)
            mid_attention = True
            up_channels = (256, 128, 64)
            up_attention = (True, True, False)
            in_channel = 256

            self.unet_depth = UNet(
                in_channel, 3, 
                down_channels=down_channels,
                down_attention=down_attention,
                mid_attention=mid_attention,
                up_channels=up_channels,
                up_attention=up_attention
            ).cuda()

            # self.downsample = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1).to(device)
            # self.downsample = nn.Sequential(
            #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
            # )
            # self.depth_head_lowres = nn.Sequential(
            #     nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            #     nn.GELU(),
            #     nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
            # )
            # self.head = nn.Sequential(
            #     nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            #     nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(last_dim, 3, kernel_size=1, stride=1, padding=0)
            # )

            # feature_dim = num_depth_candidates
            # self.position_head = nn.Sequential(
            #     nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(feature_dim // 2, feature_dim // 4, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(feature_dim // 4, 3, kernel_size=1, stride=1, padding=0)
            # )

        # # dino
        # self.dino_encoder = DinoWrapper(
        #     model_name=encoder_model_name,
        #     freeze=encoder_freeze,
        # )
                
        # self.transformer = TriplaneTransformer(
        #     inner_dim=transformer_dim, 
        #     num_layers=transformer_layers, 
        #     num_heads=transformer_heads,
        #     image_feat_dim=encoder_feat_dim,
        #     triplane_low_res=triplane_low_res, 
        #     triplane_high_res=triplane_high_res, 
        #     triplane_dim=triplane_dim,
        # )
        
        # cfg={'in_channels': 120, 'xyz_offset': True, 'restrict_offset': True, 'use_rgb': True, 'feature_channels': {'xyz': 3, 'scaling': 3, 'rotation': 4, 'opacity': 1, 'shs': 3}, 'clip_scaling': 0.2}
        # self.gs_net = GSLayer(cfg)
        # self.plane_axes = generate_planes().cuda()

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def load_triplane_state_dict(self, path="/workspace/code/InstantMesh/ckpts/instant_nerf_base.ckpt"):
        state_dict = torch.load(path, map_location='cpu')['state_dict']
        keys = list(state_dict.keys())
        # Load the weights into the dino_encoder model
        for name, param in self.dino_encoder.named_parameters():
            if name in keys:
                param.data.copy_(keys[name])
        # Load the weights into the transformer model
        for name, param in self.transformer.named_parameters():
            if name in keys:
                param.data.copy_(keys[name])

    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings

    def get_pos_from_network_output(self, ray_dirs_xy, depth, offset):

        pos = ray_dirs_xy * depth + offset

        return pos

    def forward_gaussians(self, images, vis_feat=False, return_all=False):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images) # [B*4, 14, h, w]
        x = self.conv(x) # [B*4, 14, h, w]

        x = x.reshape(B, self.opt.num_input_views, 14, self.opt.splat_size, self.opt.splat_size) # b, 4, 14, 64, 64
        
        ## visualize multi-view gaussian features for plotting figure
        if vis_feat or return_all:
            tmp_alpha = self.opacity_act(x[:, :, 3:4])
            tmp_img_rgb = self.rgb_act(x[:, :, 11:]) # * tmp_alpha + (1 - tmp_alpha)
            tmp_img_pos = self.pos_act(x[:, :, 0:3]) * 0.5 + 0.5
            tmp_img_rotation = (self.rot_act(x[:, :, 7:10]) * 0.5 + 0.5) / ((self.rot_act(x[:, :, 7:10]) * 0.5 + 0.5).max() - (self.rot_act(x[:, :, 7:10]) * 0.5 + 0.5).min())
            tmp_img_scale = (self.scale_act(x[:, :, 4:7]) * 0.5 + 0.5) / ((self.scale_act(x[:, :, 4:7]) * 0.5 + 0.5).max() - (self.scale_act(x[:, :, 4:7]) * 0.5 + 0.5).min())
            # kiui.vis.plot_image(tmp_alpha[0]>0.01, save=True)
            # kiui.vis.plot_image(tmp_alpha[0], save=True)
            # kiui.vis.plot_image(tmp_img_rgb[0], save=True)
            # kiui.vis.plot_image(tmp_img_pos[0], save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14) # B, 16384, 14
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        if vis_feat:
            return gaussians, tmp_img_rgb, tmp_alpha, tmp_img_pos

        if return_all:
            return gaussians, tmp_img_rgb, tmp_alpha, tmp_img_pos, tmp_img_rotation, tmp_img_scale

        return gaussians

    def forward_dust3r_gaussians(self, images):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        device = 'cuda'
        batch_size = 1
        schedule = 'cosine'
        lr = 0.01
        niter = 300

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        # save images
        grid = make_grid(images[:, :3, :, :])
        to_pil_image = ToPILImage()
        img = to_pil_image(grid)

        # Save the image
        img.save('image_grid.png')

        imgs_dust3r = []
        for img in images:
            imgs_dust3r.append(dict(img=img[None][:, :3, :, :], 
                            true_shape=torch.tensor(img.shape)[1:].unsqueeze(0),
                            idx=len(imgs_dust3r), 
                            instance=str(len(imgs_dust3r))))

        out01 = self.dust3r(imgs_dust3r[0], imgs_dust3r[1])
        out23 = self.dust3r(imgs_dust3r[2], imgs_dust3r[3])

        pairs = make_pairs(imgs_dust3r, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, 'cuda', batch_size=1)

        scene = global_aligner(output, device='cuda', mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # retrieve useful values from scene:
        imgs = [torch.from_numpy(img).cuda() for img in scene.imgs]
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        pts = torch.cat([p[m] for p, m in zip(pts3d, confidence_masks)], dim=0)
        col = torch.cat([p[m] for p, m in zip(imgs, confidence_masks)], dim=0)
        pct = trimesh.PointCloud(pts.reshape(-1, 3).detach().cpu(), colors=col.reshape(-1, 3).detach().cpu())
        pct.export('/workspace/code/LGM/global_align.ply')

        min_conf_thr = 2
        pts3d = [out01[0]['pts3d'], out01[1]['pts3d_in_other_view'], out23[0]['pts3d'], out23[1]['pts3d_in_other_view']]
        mask = [out01[0]['conf'] > min_conf_thr, out01[1]['conf'] > min_conf_thr, out23[0]['conf'] > min_conf_thr, out23[1]['conf'] > min_conf_thr]
        conf = [out01[0]['conf'].unsqueeze(-1), out01[1]['conf'].unsqueeze(-1), out23[0]['conf'].unsqueeze(-1), out23[1]['conf'].unsqueeze(-1)]
        color = [imgs_dust3r[0]['img'].permute((0, 2, 3, 1)), imgs_dust3r[1]['img'].permute((0, 2, 3, 1)), imgs_dust3r[2]['img'].permute((0, 2, 3, 1)), imgs_dust3r[3]['img'].permute((0, 2, 3, 1))]
        rot = [out01[0]['rotations'], out01[1]['rotations'], out23[0]['rotations'], out23[1]['rotations']]
        scale = [out01[0]['scales'], out01[1]['scales'], out23[0]['scales'], out23[1]['scales']]

        # mask = [torch.ones_like(m) for m in mask]

        pts = torch.cat([p[m] for p, m in zip(pts3d, mask)], dim=0)
        col = torch.cat([p[m] for p, m in zip(color, mask)], dim=0)
        rots = torch.cat([p[m] for p, m in zip(rot, mask)], dim=0)
        scls = torch.cat([p[m] for p, m in zip(scale, mask)], dim=0)
        opa = torch.cat([p[m] for p, m in zip(conf, mask)], dim=0)
        # opa = torch.ones_like(opa)

        pct = trimesh.PointCloud(pts.reshape(-1, 3).detach().cpu(), colors=torch.flip(col.reshape(-1, 3).detach().cpu(), [1]))
        pct.export('/workspace/code/LGM/pct.ply')
        gaussians = torch.cat([pts, opa, scls, rots, col], dim=1).unsqueeze(0)

        return gaussians

    def forward_depth_gaussian(self, data):

        images = data['input'] # [B, 4, 9, h, W], input features
        B, V, C, H, W = images.shape

        # triplane gaussian
        # input_images = images[:, :, :3, :, :]
        input_images = images
        input_camera = data['cam_view'].flatten(-2)[:, :4, :]
        image_feats = self.dino_encoder(input_images, input_camera)
        image_feats = rearrange(image_feats, '(b v) l d -> b (v l) d', b=B)
        planes = self.transformer(image_feats)

        sample_coordinates = data['point_cloud'] # [B, N, 3]

        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=2.0)
        sampled_features = rearrange(sampled_features, 'b l n c -> b n (l c)')

        # gs_fout = query_triplane(sample_coordinates, planes)
        # gsfout2 = self.mlp_net(gs_fout)
        
        gs = self.gs_net(sampled_features, sample_coordinates)
        gaussians = torch.cat([gs[0], 1 + gs[1], gs[3], gs[2], gs[4]], dim=-1)

        return gaussians

    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        # images = data['input'] # [B, 4, 9, h, W], input features

        if self.opt.data_mode == 'dust3r':
            # data['images'][0]['img'] = data['images'][0]['img'][0]
            # data['images'][1]['img'] = data['images'][1]['img'][0]
            B, V, C, H, W = data['images'][0]['img'].shape
            data['images'][0]['img'] = data['images'][0]['img'].view(B*V, C, H, W)
            data['images'][1]['img'] = data['images'][1]['img'].view(B*V, C, H, W)
            data['images'][0]['true_shape'] = data['images'][0]['true_shape'][0]
            data['images'][1]['true_shape'] = data['images'][1]['true_shape'][0]

            images = data['images'][:2]
            cat_images = torch.cat([images[0]['img'].view(B, V, C, H, W), images[1]['img'].view(B, V, C, H, W)], dim=1)

            # Get current device ID
            device_id = torch.cuda.current_device()
            output = inference([tuple(images)], self.mast3r_model.to(f"cuda:{device_id}"), f"cuda:{device_id}", batch_size=V, verbose=False)

            # at this stage, you have the raw dust3r predictions
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']
            N, H, W, C = pred1['desc'].shape
            pred1['desc'] = pred1['desc'].view(B, V, H, W, C)
            pred2['desc'] = pred2['desc'].view(B, V, H, W, C)

            # depth prediction
            if self.opt.use_geometry_net:
                # get_disp_candidates()
                
                # depth_feat_1 = pred1['depth_feat'].cuda() # .view(B, V, 256, 128, 128) # .contiguous() # (1, 256, 128, 128) # depth_feat[:, :128, :64, :64]
                # depth_feat_2 = pred2['depth_feat'].cuda() # .view(B, V, 256, 128, 128) # .contiguous() # (1, 256, 128, 128) # depth_feat[:, :128, :64, :64]
                # cat_depth_feat = torch.cat([depth_feat_1, depth_feat_2], dim=0)
                # pos_depth = self.unet_depth(cat_depth_feat)
                # pos_depth = self.pos_act(pos_depth) # [B, N, 3]
                # pos_depth = pos_depth.view(B, -1, 3)

                extrinsics = data['extrinsics'][:, :self.opt.num_input_views, :, :]
                ext1 = extrinsics[:, 0::2, :, :]
                ext2 = extrinsics[:, 1::2, :, :]
                ext1_inv = torch.inverse(ext1)
                ext_origin = torch.eye(4)
                ext_origin[2][3] = self.opt.cam_radius * -1
                ext_origin = ext_origin.unsqueeze(0).unsqueeze(0).repeat(self.opt.batch_size,self.opt.num_input_views//2,1,1).cuda()
                Trf = torch.matmul(ext1_inv, ext_origin).view(self.opt.batch_size*self.opt.num_input_views//2, 4, 4)
                
                ptd3d_1 = pred1['pts3d']#.view(B, V, 3, H, W)
                ptd3d_2 = pred2['pts3d_in_other_view']#.view(B, V, 3, H, W)
                ptd3d_1_trf = geotrf(Trf, ptd3d_1.cuda())
                ptd3d_2_trf = geotrf(Trf, ptd3d_2.cuda())
                ptd3d_1_trf[..., 0] = ptd3d_1_trf[..., 0] *-1
                ptd3d_2_trf[..., 0] = ptd3d_2_trf[..., 0] *-1

                conf_1 = pred1['conf']
                conf_2 = pred2['conf']
                def get_mask(conf, min_conf_thr=3):
                    return (conf > min_conf_thr)
                msk1 = get_mask(conf_1, 1.05)
                msk2 = get_mask(conf_2, 1.05)
                cat_msk = torch.cat([msk1, msk2], dim=0).cuda()
                cat_ptd3d = torch.cat([ptd3d_1_trf, ptd3d_2_trf], dim=0)#.permute(0, 3, 1, 2)

                # # visualization
                # point_np = cat_ptd3d[cat_msk].view(-1, 3).cpu().numpy()
                # color_np = np.array([0, 255, 0])[None, ...].repeat(point_np.shape[0], axis=0)
                # pct = trimesh.PointCloud(point_np, colors=color_np)
                # pct.export('point_cloud2332323.ply')

                cat_msk = cat_msk[:, ::2, ::2]
                cat_ptd3d = cat_ptd3d[:, ::2, ::2, ...]
                cat_ptd3d = cat_ptd3d.reshape(self.opt.batch_size, -1, 3)
                cat_msk = cat_msk.reshape(self.opt.batch_size, -1)
                # cat_ptd3d = normalize_point_cloud(cat_ptd3d, box_scale=2.0)
                
                def cal_cost_volume(depth_feat_0, depth_feat_1, data):

                    depth_feat_01_list = []
                    for i in range(len(depth_feat_0)):
                        depth_feat_01_list.append(depth_feat_0[i:i+1])
                        depth_feat_01_list.append(depth_feat_1[i:i+1])
                    depth_feat_01 = torch.cat(depth_feat_01_list, dim=0)

                    depth_feat_10_list = []
                    for i in range(len(depth_feat_0)):
                        depth_feat_10_list.append(depth_feat_1[i:i+1])
                        if i % 2 == 0:
                            depth_feat_10_list.append(depth_feat_0[i+1:i+2])
                        else:
                            depth_feat_10_list.append(depth_feat_0[i-1:i])
                    depth_feat_10 = torch.cat(depth_feat_10_list, dim=0)

                    # downsample spatial dimensions
                    depth_feat_01 = self.downsample(depth_feat_01.cuda())
                    depth_feat_10 = self.downsample(depth_feat_10.cuda())

                    N, C, H, W = depth_feat_01.shape
                    # disp_candi_curr = get_disp_candidates(num_samples=128, feat=depth_feat_01)

                    intrinsics = data['intrinsics'][:, :self.opt.num_input_views, :, :].reshape(N, 3, 3)
                    extrinsics = data['extrinsics'][:, :self.opt.num_input_views, :, :].reshape(N, 4, 4)
                    raw_correlation_in_lists = []

                    for i in range(self.opt.batch_size):
                        depth_feat_10_single_batch = depth_feat_10[i:i+self.opt.num_input_views, ...]
                        intrinsics_single_batch = intrinsics[i:i+self.opt.num_input_views, ...]
                        extrinsics_single_batch = extrinsics[i:i+self.opt.num_input_views, ...]
                        extrinsics_single_batch[[0,1,2,3]] = extrinsics_single_batch[[1,2,3,0]]
                        
                        disp_candi_curr = get_disp_candidates(num_samples=32, feat=depth_feat_10_single_batch)
                        warped_feature = warp_with_pose_depth_candidates(depth_feat_10_single_batch, intrinsics_single_batch, extrinsics_single_batch, 1 / disp_candi_curr.repeat(1, 1, H, W), warp_padding_mode="zeros")

                        raw_correlation_in = (depth_feat_10_single_batch.unsqueeze(2) * warped_feature).sum(
                            1
                        ) / (
                            C**0.5
                        )  # [vB, D, H, W]

                    raw_correlation_in_lists.append(raw_correlation_in)
                    # average all cost volumes
                    raw_correlation_in = torch.mean(
                        torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
                    )  # [vxb d, h, w]
                    raw_correlation_in = torch.cat((raw_correlation_in, depth_feat_01), dim=1) # 4 256 128 128

                    raw_correlation_in = F.interpolate(raw_correlation_in, scale_factor=2, mode='bilinear', align_corners=False)

                    N, C, H, W = raw_correlation_in.shape
                    corr = raw_correlation_in.view(self.opt.batch_size, N // self.opt.batch_size, C, H, W)

                    # out = self.unet_depth(raw_correlation_in)
                    # pos_geometry = self.position_head(out)
                    # pos_geometry = pos_geometry.reshape(B, self.opt.num_input_views, 3, self.opt.splat_size, self.opt.splat_size) # b, 4, 3, 64, 64
                    # pos_geometry = pos_geometry.permute(0, 1, 3, 4, 2).reshape(B, -1, 3) # B, 16384, 3
                    # pos_geometry = self.pos_act(pos_geometry[..., 0:3]) # [B, N, 3]

                    # # softmax to get coarse depth and density
                    # pdf = F.softmax(
                    #     self.depth_head_lowres(out), dim=1
                    # )  # [2xB, D, H, W]
                    # coarse_disps = (disp_candi_curr * pdf).sum(
                    #     dim=1, keepdim=True
                    # )  # (vb, 1, h, w)
                    # depths = 1.0 / coarse_disps

                    return corr

            # concat feat and rays
            feat1, feat2 = pred1['desc'].permute(0, 1, 4, 2, 3), pred2['desc'].permute(0, 1, 4, 2, 3)
            feat_mast3r = torch.cat([feat1, feat2], dim=1)
            cat_input = torch.cat([cat_images, feat_mast3r.cuda(), data['rays']], dim=2)
            gaussians = self.forward_gaussians(cat_input) # [B, N, 14]

            if self.opt.use_geometry_net:
                gaussians[:, :, :3] += cat_ptd3d
                # gaussians[:, :, 3] *= cat_msk # opacity = self.opacity_act(x[..., 3:4])
                # gaussians[:, :, :3] += pos_depth
            
            if self.opt.use_depth_net:

                # add for depth prediction
                ptd3d_1 = pred1['pts3d']
                ptd3d_2 = pred2['pts3d_in_other_view']
                conf_1 = pred1['conf']
                conf_2 = pred2['conf']

                extrinsics = data['extrinsics'][:, :self.opt.num_input_views, :, :]
                ext1 = extrinsics[:, 0::2, :, :]
                ext2 = extrinsics[:, 1::2, :, :]
                ext1_inv = torch.inverse(ext1)
                # TODO: check if this is correct
                Trf = torch.matmul(ext1_inv, ext2).view(self.opt.batch_size*self.opt.num_input_views//2, 4, 4)
                
                def get_mask(conf, min_conf_thr=3):
                    return (conf > min_conf_thr)

                rel_poses = []
                depth1_list =[]
                depth2_list = []
                pp = torch.tensor((W/2, H/2))
                pixels = np.mgrid[:W, :H].T.astype(np.float32)
                
                mask = data['masks_output'][:, :self.opt.num_input_views, :, :]
                mask1 = mask[:, ::2, :, :].reshape(conf_1.shape[0], conf_1.shape[1], conf_1.shape[2])
                mask2 = mask[:, 1::2, :, :].reshape(conf_2.shape[0], conf_2.shape[1], conf_2.shape[2])
                
                for i in range(N):

                    # trf
                    depth1 = ptd3d_1[i].numpy()[..., 2]
                    # depth2 = geotrf(Trf[i], ptd3d_2[i].cuda())[..., 2]
                    # depth2 = depth2.cpu().numpy()
                    depth2 = ptd3d_2[i].numpy()[..., 2]
                    depth1 = (depth1 - depth1.min()) / (depth1.max()-depth1.min())
                    depth2 = (depth2 - depth2.min()) / (depth2.max()-depth2.min())
                    msk1 = get_mask(conf_1[i], 1.0005).numpy()
                    msk2 = get_mask(conf_2[i], 1.0005).numpy()
                    msk1 = mask1[i].cpu() > 0.5
                    msk2 = mask2[i].cpu() > 0.5
                    depth1[~msk1] = 0
                    depth2[~msk2] = 0
                    depth1_list.append(torch.from_numpy(depth1).unsqueeze(0))
                    depth2_list.append(torch.from_numpy(depth2).unsqueeze(0))

                    # cv2.imwrite(f'/workspace/code/LGM/depth1_trf_new_{i}.png', (depth1 * 255).astype(np.uint8))
                    # cv2.imwrite(f'/workspace/code/LGM/depth2_trf_new_{i}.png', (depth2 * 255).astype(np.uint8))
                    continue

                    # compute all parameters directly from raw input

                    focal = float(estimate_focal_knowing_depth(ptd3d_1[i:i+1], pp, focal_mode='weiszfeld'))
                    
                    # estimate the pose of pts1 in image 2
                    # msk = self.get_masks()[i].numpy()
                    msk = get_mask(conf_2[i]).numpy()
                    K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])
                    ptd3d = ptd3d_2[i].numpy()

                    try:
                        res = cv2.solvePnPRansac(ptd3d[msk], pixels[msk], K, None,
                                                iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
                        success, R, T, inliers = res
                        assert success

                        R = cv2.Rodrigues(R)[0]  # world to cam
                        pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
                    except Exception as e:
                        print(e)
                        pose = np.eye(4)

                    rel_poses.append(torch.from_numpy(pose.astype(np.float32)))

                    depth1 = ptd3d_1[i].numpy()[..., 2]
                    depth2 = geotrf(inv(pose), ptd3d_2[i].numpy())[..., 2]
                    depth1_list.append(torch.from_numpy(depth1).unsqueeze(0))
                    depth2_list.append(torch.from_numpy(depth2).unsqueeze(0))
                    # depth1 = ptd3d_1[i][..., 2].unsqueeze(-1).cpu().numpy()
                    # depth2 = ptd3d_2[i][..., 2].unsqueeze(-1).cpu().numpy()
                    cv2.imwrite(f'/workspace/code/LGM/depth1_{i}.png', ((depth1 - depth1.min()) / (depth1.max()-depth1.min()) * 255).astype(np.uint8))
                    cv2.imwrite(f'/workspace/code/LGM/depth2_{i}.png', ((depth2 - depth2.min()) / (depth2.max()-depth2.min()) * 255).astype(np.uint8))
                    # cv2.imwrite(f'/workspace/code/LGM/img1_{i}.png', ((img1+1)/2 * 255).astype(np.uint8))
                    # cv2.imwrite(f'/workspace/code/LGM/img2_{i}.png', ((img2+1)/2 * 255).astype(np.uint8))
                    # self.depth = [self.pred_i['0_1'][..., 2], geotrf(inv(rel_poses[1]), self.pred_j['0_1'])[..., 2]]

                # get depth
                # depth_1 = ptd3d_1[:, :, 2:3, ...]
                # depth_1 = depth_1 / depth_1.max()
                # depth_2 = ptd3d_2[:, :, 2:3, ...]
                # depth_1 = depth_2 / depth_2.max()
                depth_1 = torch.cat(depth1_list, dim=0).cuda()
                depth_2 = torch.cat(depth2_list, dim=0).cuda()
                depth_1 = depth_1.reshape(self.opt.batch_size, self.opt.num_input_views//2, 1, H, W)
                depth_2 = depth_2.reshape(self.opt.batch_size, self.opt.num_input_views//2, 1, H, W)

                # depth + offset
                depth = torch.cat([depth_1, depth_2], dim=1)
                # depth = depth * 2 - 1

                # refine depth
                # depth feat
                depth_feat_1 = pred1['depth_feat'].cuda() # .view(B, V, 256, 128, 128) # .contiguous() # (1, 256, 128, 128) # depth_feat[:, :128, :64, :64]
                depth_feat_2 = pred2['depth_feat'].cuda() # .view(B, V, 256, 128, 128) # .contiguous() # (1, 256, 128, 128) # depth_feat[:, :128, :64, :64]

                depth_1_residual = self.depth_feat_head_1(depth_feat_1)
                depth_2_residual = self.depth_feat_head_2(depth_feat_2)
                depth_1_residual = depth_1_residual.reshape(self.opt.batch_size, self.opt.num_input_views//2, 1, H, W)
                depth_2_residual = depth_2_residual.reshape(self.opt.batch_size, self.opt.num_input_views//2, 1, H, W)
                depth_residual = torch.cat([depth_1_residual, depth_2_residual], dim=1)

                # cat_depth_feat = torch.cat([depth_feat_1, depth_feat_2], dim=0)
                # cat_depth_feat = self.depth_feat_head(cat_depth_feat)
                # cat_depth_feat = torch.cat([cat_depth_feat, depth], dim=1)
                # depth_residual = self.pos_act(self.unet_depth(cat_depth_feat))

                depth_mast3r = depth.clone()
                depth = depth_mast3r + depth_residual
                depth = (depth - depth.min()) / (depth.max()-depth.min())

                depth = depth.reshape(self.opt.batch_size, self.opt.num_input_views, 1, H, W)
                ray_dirs = data['rays'][:, :, 3:6, ...]

                # use gt_depth for testing
                # depth = data['depths_input'].reshape(self.opt.batch_size, self.opt.num_input_views, 1, H, W)

                pos = depth * ray_dirs
                # pos = F.interpolate(pos, size=(3, self.opt.splat_size, self.opt.splat_size), mode='trilinear', align_corners=False)
                downsample_rate = self.opt.input_size // self.opt.splat_size
                pos = pos[:, :, :, ::downsample_rate, ::downsample_rate]

                # depth_1 = F.interpolate(depth, size=(3, self.opt.splat_size, self.opt.splat_size), mode='trilinear', align_corners=False)
                # ray_dirs_1 = F.interpolate(ray_dirs, size=(3, self.opt.splat_size, self.opt.splat_size), mode='trilinear', align_corners=False)
                offset = gaussians[:, :, :3].reshape(B, 4, 3, self.opt.splat_size, self.opt.splat_size)
                # pos = self.get_pos_from_network_output(ray_dirs, depth, offset).reshape(B, -1, 3)
                pos = self.pos_act(pos + offset)
                # TODO: use opt to control this
                if self.opt.use_depth_offset:
                    gaussians[:, :, :3] = pos.reshape(B, -1, 3)
        else:
            # use the first view to predict gaussians
            images = data['input'] # [B, 4, 9, h, W], input features
            gaussians = self.forward_gaussians(images) # [B, N, 14]

        if self.opt.data_mode == 'shapesplat':
            # dust3r gaussian

            # always use white bg
            bg_color = torch.ones(3, dtype=torch.float32, device=data['gaussians'].device)

            gt_output = self.gs.render(data['gaussians'], data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color) # [B, V, C, output_size, output_size]
            image = gt_output['image']
            data['images_output'] = image
            B, V, C, H, W = image.shape
            image0 = {"img": image[0, 0::2], "true_shape": np.array([[self.opt.input_size, self.opt.input_size]]), "idx": 0, "instance": "0"}
            image1 = {"img": image[0, 1::2], "true_shape": np.array([[self.opt.input_size, self.opt.input_size]]), "idx": 1, "instance": "1"}
            images = [image0, image1]
            cat_images = torch.cat([images[0]['img'].view(B, V//2, C, H, W), images[1]['img'].view(B, V//2, C, H, W)], dim=1)

            # Get current device ID
            device_id = torch.cuda.current_device()
            output = inference([tuple(images)], self.mast3r_model.to(f"cuda:{device_id}"), f"cuda:{device_id}", batch_size=V//2, verbose=False)

            # at this stage, you have the raw dust3r predictions
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']
            N, H, W, C = pred1['desc'].shape
            pred1['desc'] = pred1['desc'].view(B, V//2, H, W, C)
            pred2['desc'] = pred2['desc'].view(B, V//2, H, W, C)

            # concat feat and rays
            feat1, feat2 = pred1['desc'].permute(0, 1, 4, 2, 3), pred2['desc'].permute(0, 1, 4, 2, 3)
            feat_mast3r = torch.cat([feat1, feat2], dim=1)
            cat_input = torch.cat([cat_images, feat_mast3r.cuda(), data['rays']], dim=2)
            gaussians = self.forward_gaussians(cat_input) # [B, N, 14]

        # dust3r gaussian
        # gaussians = self.forward_dust3r_gaussians(images)

        # depth gaussian
        # gaussians = self.forward_depth_gaussian(data)

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]
        pred_depths = results['depth'] # [B, V, 1, output_size, output_size]

        pred_depths = minmax_norm(pred_depths)
        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['depths_pred'] = pred_depths
        
        if self.opt.use_depth_net:
            # results['depths_pred'] = depth
            results['depths_mast3r'] = depth_mast3r.reshape(self.opt.batch_size, self.opt.num_input_views, 1, H, W)
            results['depths_refined'] = depth

        if self.opt.data_mode == 'shapesplat':
            gt_images = gt_output['image'] # [B, V, 3, output_size, output_size], ground-truth novel views
            gt_masks = gt_output['alpha'] # [B, V, 1, output_size, output_size], ground-truth masks
        else:
            gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
            gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.use_depth:
            gt_depths = data['depths_output'] # [B, V, 1, output_size, output_size], ground-truth depths
            masks_out = data['masks_output'] > 0.5
            loss_mse_depth = F.mse_loss(pred_depths[masks_out], gt_depths[masks_out])
            loss = loss + loss_mse_depth

            # depth grad loss
            pred_depths = pred_depths.reshape(-1, self.opt.output_size, self.opt.output_size)
            gt_depths = gt_depths.reshape(-1, self.opt.output_size, self.opt.output_size)
            masks_out = masks_out.reshape(-1, self.opt.output_size, self.opt.output_size)
            loss_grad_depth = get_depth_grad_loss(pred_depths, gt_depths, masks_out, inverse_depth_loss=False)
            loss = loss + loss_grad_depth

        # TODO: update this
        if self.opt.use_depth_net:
            gt_depths = data['depths_input'] # [B, V, 1, output_size, output_size], ground-truth depths
            masks_inp = data['masks_input'] > 0
            loss_mse_depth = F.mse_loss(depth[masks_inp], gt_depths[masks_inp])
            loss = loss + loss_mse_depth

            # depth grad loss
            depth = depth.reshape(-1, H, W)
            gt_depths = gt_depths.reshape(-1, H, W)
            masks_inp = masks_inp.reshape(-1, H, W)
            loss = loss + get_depth_grad_loss(depth, gt_depths, masks_inp, inverse_depth_loss=False)

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss
        # results['loss_mse'] = loss_mse
        # results['loss_mse_depth'] = loss_mse_depth
        # results['loss_grad_depth'] = loss_grad_depth

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            ssim_score = ssim(pred_images.reshape(-1, 3, self.opt.output_size, self.opt.output_size).cpu(), gt_images.reshape(-1, 3, self.opt.output_size, self.opt.output_size).cpu())
            # psnr_score = psnr_metric(pred_images.detach(), gt_images)

            results['lpips'] = loss_lpips
            results['ssim'] = ssim_score.item()
            results['psnr'] = psnr
            results['gaussians'] = gaussians
            # results['uid'] = data['uid']

        return results

class TriplaneModel(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x) # 0.1
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        # self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again
        self.rgb_act = lambda x: torch.sigmoid(x)

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        # # Dust3r
        # model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        # self.dust3r = AsymmetricCroCo3DStereoGaussian.from_pretrained(model_name)

        # dino
        self.dino_encoder = DinoWrapper(
            model_name=self.opt.encoder_model_name,
            freeze=self.opt.encoder_freeze,
        )
                
        # transformer
        self.transformer = TriplaneTransformer(
            inner_dim=self.opt.transformer_dim, 
            num_layers=self.opt.transformer_layers, 
            num_heads=self.opt.transformer_heads,
            image_feat_dim=self.opt.encoder_feat_dim,
            triplane_low_res=self.opt.triplane_low_res, 
            triplane_high_res=self.opt.triplane_high_res, 
            triplane_dim=self.opt.triplane_dim,
        )

        cfg={'in_channels': 120, 'xyz_offset': True, 'restrict_offset': True, 'use_rgb': True, 'feature_channels': {'xyz': 3, 'scaling': 3, 'rotation': 4, 'opacity': 1, 'shs': 3}, 'clip_scaling': 0.2}
        self.gs_net = GSLayer(cfg)
        self.plane_axes = generate_planes().cuda()

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def load_triplane_state_dict(self, path="/workspace/code/InstantMesh/ckpts/instant_nerf_base.ckpt"):
        state_dict = torch.load(path, map_location='cpu')['state_dict']
        keys = list(state_dict.keys())
        # Load the weights into the dino_encoder model
        for name, param in self.dino_encoder.named_parameters():
            if name in keys:
                param.data.copy_(keys[name])
        # Load the weights into the transformer model
        for name, param in self.transformer.named_parameters():
            if name in keys:
                param.data.copy_(keys[name])

    def forward_dust3r_gaussians(self, images):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        device = 'cuda'
        batch_size = 1
        schedule = 'cosine'
        lr = 0.01
        niter = 300

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        # save images
        grid = make_grid(images[:, :3, :, :])
        to_pil_image = ToPILImage()
        img = to_pil_image(grid)

        # Save the image
        img.save('image_grid.png')

        imgs_dust3r = []
        for img in images:
            imgs_dust3r.append(dict(img=img[None][:, :3, :, :], 
                            true_shape=torch.tensor(img.shape)[1:].unsqueeze(0),
                            idx=len(imgs_dust3r), 
                            instance=str(len(imgs_dust3r))))

        out01 = self.dust3r(imgs_dust3r[0], imgs_dust3r[1])
        out23 = self.dust3r(imgs_dust3r[2], imgs_dust3r[3])

        pairs = make_pairs(imgs_dust3r, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, 'cuda', batch_size=1)

        scene = global_aligner(output, device='cuda', mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # retrieve useful values from scene:
        imgs = [torch.from_numpy(img).cuda() for img in scene.imgs]
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        pts = torch.cat([p[m] for p, m in zip(pts3d, confidence_masks)], dim=0)
        col = torch.cat([p[m] for p, m in zip(imgs, confidence_masks)], dim=0)
        pct = trimesh.PointCloud(pts.reshape(-1, 3).detach().cpu(), colors=col.reshape(-1, 3).detach().cpu())
        pct.export('/workspace/code/LGM/global_align.ply')

        min_conf_thr = 2
        pts3d = [out01[0]['pts3d'], out01[1]['pts3d_in_other_view'], out23[0]['pts3d'], out23[1]['pts3d_in_other_view']]
        mask = [out01[0]['conf'] > min_conf_thr, out01[1]['conf'] > min_conf_thr, out23[0]['conf'] > min_conf_thr, out23[1]['conf'] > min_conf_thr]
        conf = [out01[0]['conf'].unsqueeze(-1), out01[1]['conf'].unsqueeze(-1), out23[0]['conf'].unsqueeze(-1), out23[1]['conf'].unsqueeze(-1)]
        color = [imgs_dust3r[0]['img'].permute((0, 2, 3, 1)), imgs_dust3r[1]['img'].permute((0, 2, 3, 1)), imgs_dust3r[2]['img'].permute((0, 2, 3, 1)), imgs_dust3r[3]['img'].permute((0, 2, 3, 1))]
        rot = [out01[0]['rotations'], out01[1]['rotations'], out23[0]['rotations'], out23[1]['rotations']]
        scale = [out01[0]['scales'], out01[1]['scales'], out23[0]['scales'], out23[1]['scales']]

        # mask = [torch.ones_like(m) for m in mask]

        pts = torch.cat([p[m] for p, m in zip(pts3d, mask)], dim=0)
        col = torch.cat([p[m] for p, m in zip(color, mask)], dim=0)
        rots = torch.cat([p[m] for p, m in zip(rot, mask)], dim=0)
        scls = torch.cat([p[m] for p, m in zip(scale, mask)], dim=0)
        opa = torch.cat([p[m] for p, m in zip(conf, mask)], dim=0)
        # opa = torch.ones_like(opa)

        pct = trimesh.PointCloud(pts.reshape(-1, 3).detach().cpu(), colors=torch.flip(col.reshape(-1, 3).detach().cpu(), [1]))
        pct.export('/workspace/code/LGM/pct.ply')
        gaussians = torch.cat([pts, opa, scls, rots, col], dim=1).unsqueeze(0)

        return gaussians

    def forward_triplane_model(self, data):

        images = data['input'] # [B, 4, 9, h, W], input features
        B, V, C, H, W = images.shape

        # triplane gaussian
        # input_images = images[:, :, :3, :, :]
        # input_camera = data['cam_view'].flatten(-2)[:, :4, :]
        input_images = images
        input_camera = data['cam_view'].flatten(-2)[:, :self.opt.num_input_views, :]
        image_feats = self.dino_encoder(input_images, input_camera)
        image_feats = rearrange(image_feats, '(b v) l d -> b (v l) d', b=B)
        planes = self.transformer(image_feats)

        sample_coordinates = data['point_cloud'] # [B, N, 3]

        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=2.0)
        sampled_features = rearrange(sampled_features, 'b l n c -> b n (l c)')

        gs = self.gs_net(sampled_features, sample_coordinates)

        pos = gs[0] # + self.pos_act(gs[0]) # [B, N, 3]
        opacity = gs[1] + 1 # self.opacity_act(gs[1])
        scale = gs[3] # self.scale_act(gs[3])
        rotation = gs[2] # self.rot_act(gs[2])
        rgbs = gs[4] # self.rgb_act(gs[4])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        # gaussians = torch.cat([gs[0], 1 + gs[1], gs[3], gs[2], gs[4]], dim=-1)

        return gaussians

    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        images = data['input'] # [B, 4, 9, h, W], input features

        # dust3r gaussian
        # gaussians = self.forward_dust3r_gaussians(images)

        # depth gaussian
        gaussians = self.forward_triplane_model(data)

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results

class MASt3RGS(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        # Mast3r
        # device = 'cuda'
        # model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        # self.mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        # self.mast3r_model.eval()
        # for param in self.mast3r_model.parameters():
        #     param.requires_grad = False
        
        self.encoder = AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='PatchEmbedDust3R', # 'ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='gaussian_head',
            output_mode='pts3d+gaussian+desc24',
            depth_mode=('exp', -float('inf'), float('inf')),
            conf_mode=('exp', 1, float('inf')),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=True, # True
            sh_degree=1
        )
        # self.encoder.requires_grad_(False)
        # self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        # self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)
        # self.encoder.mv_head.gaussian_dpt.dpt.requires_grad_(True)

        print('Loading Model')
        ckpt = torch.load("/workspace/code/splatt3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        _ = self.encoder.load_state_dict(ckpt['model'], strict=False)
        del ckpt

        in_channels = 14 + 3 # + 24        

        # unet
        self.unet = UNet(
            in_channels, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
            num_frames=self.opt.num_input_views, # new added
        )

        size_in_millions = calculate_model_size_millions(self.unet)
        print(f"Unet size: {size_in_millions:.2f} M")
        size_in_millions = calculate_model_size_millions(self.encoder)
        print(f"Encoder size: {size_in_millions:.2f} M")

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings

    def get_pos_from_network_output(self, ray_dirs_xy, depth, offset):

        pos = ray_dirs_xy * depth + offset

        return pos

    def depthmap_to_absolute_camera_coordinates(self, depthmaps, camera_intrinsics, camera_poses, **kw):
        """
        Args:
            - depthmaps (BxHxW tensor): Batch of depth maps
            - camera_intrinsics: a 3x3 matrix
            - camera_poses: a batch of 4x3 or 4x4 cam2world matrices (Bx4x4 or Bx4x3)
        Returns:
            pointmap of absolute coordinates (BxHxWx3 tensor), and a mask specifying valid pixels.
        """

        def depthmap_to_camera_coordinates(depthmaps, camera_intrinsics, pseudo_focal=None):
            """
            Args:
                - depthmaps (BxHxW tensor): Batch of depth maps
                - camera_intrinsics: a 3x3 matrix
            Returns:
                pointmap of absolute coordinates (BxHxWx3 tensor), and a mask specifying valid pixels.
            """
            camera_intrinsics = torch.tensor(camera_intrinsics, dtype=torch.float32)
            B, H, W = depthmaps.shape

            # Compute 3D ray associated with each pixel
            # Strong assumption: there are no skew terms
            assert camera_intrinsics[0, 1] == 0.0
            assert camera_intrinsics[1, 0] == 0.0
            if pseudo_focal is None:
                fu = camera_intrinsics[0, 0]
                fv = camera_intrinsics[1, 1]
            else:
                assert pseudo_focal.shape == (H, W)
                fu = fv = pseudo_focal
            cu = camera_intrinsics[0, 2]
            cv = camera_intrinsics[1, 2]
            device = camera_intrinsics.device

            u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
            u = u.unsqueeze(0).expand(B, -1, -1)
            v = v.unsqueeze(0).expand(B, -1, -1)
            z_cam = depthmaps
            x_cam = (u - cu) * z_cam / fu
            y_cam = (v - cv) * z_cam / fv
            X_cam = torch.stack((x_cam, y_cam, z_cam), dim=-1).to(torch.float32)

            # Mask for valid coordinates
            valid_mask = (depthmaps > 0.0)
            return X_cam, valid_mask

        X_cam, valid_mask = depthmap_to_camera_coordinates(depthmaps, camera_intrinsics)

        B, H, W, _ = X_cam.shape
        R_cam2world = camera_poses[:, :3, :3]
        t_cam2world = camera_poses[:, :3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = torch.einsum("bij,bhwj->bhwi", R_cam2world, X_cam) + t_cam2world[:, None, None, :]
        return X_world

    def forward_splatt3r(self, view1, view2, data):

        extrinsics = data['extrinsics']

        # Freeze the encoder and decoder
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # Train the downstream heads
        pred1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)

        pred1['covariances'] = build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = build_covariance(pred2['scales'], pred2['rotations'])

        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1['sh'])
            new_sh2 = torch.zeros_like(pred2['sh'])
            new_sh1[..., 0] = RGB2SH(einops.rearrange(view1['img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = RGB2SH(einops.rearrange(view2['img'], 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            pred2['sh'] = pred2['sh'] + new_sh2

        # Update the keys to make clear that pts3d and means are in view1's frame
        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')

        return pred1, pred2

    def forward_mast3r(self, view1, view2, data):

        extrinsics = data['extrinsics']

        # Freeze the encoder and decoder
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # Train the downstream heads
        pred1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)

        return pred1, pred2

    def forward_mast3r_mv(self, view1, view2, data):

        V = len(data['images'])

        # Freeze the encoder and decoder
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        dec = [torch.cat((d1, d2), dim=0) for d1, d2 in zip(dec1, dec2)]

        # shape = torch.cat((shape1, shape2), dim=1)
        H, W = int(shape1.min()), int(shape1.max())
        pred = self.encoder.mv_head(dec, (H, W, V))

        return pred

    def random_sample_points(self, tensor, mask, num_samples):
        """
        在mask为1的区域内随机采样num_samples个点，并提取这些点在tensor中的数值，
        最终结果存储在一个形状为(B, num_samples)的张量中。
        
        Args:
            tensor (torch.Tensor): 形状为(B, L, C)的张量
            mask (torch.Tensor): 形状为(B, L)的掩码张量
            num_samples (int): 每个批次中要采样的点的数量
            
        Returns:
            sampled_values (torch.Tensor): 形状为(B, num_samples)的张量，包含采样点的数值
        """
        B, L, C = tensor.shape
        
        # # 重塑tensor和mask
        # tensor = tensor.view(-1, channels, height, width).permute(0, 2, 3, 1)  # 形状为(BN, H, W, C)
        # mask = mask.squeeze(2).view(-1, height, width)  

        # flatten_tensor = tensor.view(-1, channels)  # 形状为(L, C)
        # flatten_mask = mask.view(-1)  # 形状为(BN, H, W, C)
        # flatten_tensor = tensor
        # flatten_mask = mask

        # 初始化存储采样值的张量
        # sampled_values = torch.zeros((batch_size, num_samples))
        batch_sampled_values = []
        
        for i in range(B):

            flatten_tensor = tensor[i]
            flatten_mask = mask[i]

            # 找到combined_mask为1的位置 (H, W)，返回二维索引(x, y)
            # valid_indices = torch.nonzero(faltten_mask == 1)
            valid_indices = torch.where(flatten_mask == 1)[0]
            
            if len(valid_indices) == 0:
                return None

            # 如果valid_indices的数量小于num_samples，则重复valid_indices直到其长度至少为num_samples
            if len(valid_indices) < num_samples:
                valid_indices = valid_indices.repeat((num_samples // len(valid_indices)) + 1)
                # repeated_valid_indices = valid_indices.repeat((num_samples // len(valid_indices)) + 1)
                # valid_indices = repeated_valid_indices[:num_samples]

            if len(valid_indices) < num_samples:
                raise ValueError(f"Batch has {len(valid_indices)} fewer than {num_samples} points where mask is 1.")
            
            # 随机选择num_samples个索引
            selected_indices = torch.randperm(len(valid_indices))[:num_samples]
            
            # 获取这些索引对应的坐标 (x, y)
            sampled_points = valid_indices[selected_indices]
            
            sampled_values = []
            # 从第一个通道中提取对应位置的值
            for _, sample_idx in enumerate(sampled_points):
                # 提取当前批次的所有子张量中对应位置的值
                values_at_point = flatten_tensor[sample_idx]  # 形状为(N,)
                sampled_values.append(values_at_point)
            
            batch_sampled_values.append(torch.stack(sampled_values, dim=0))
        
        batch_sampled_values = torch.stack(batch_sampled_values, dim=0)

        return batch_sampled_values

    def chamfer_loss(self, pred, gt):
        dist = torch.cdist(pred, gt)  # [B, N, M]
        min_pred_to_gt = torch.min(dist, dim=2).values  # [B, N]
        min_gt_to_pred = torch.min(dist, dim=1).values  # [B, M]
        
        # 按点数量平均（与PyTorch3D的point_reduction="mean"对齐）
        loss = min_pred_to_gt.mean() + min_gt_to_pred.mean()
        return loss

    def export_ply(self, pts3d, img, valid_mask):
        pts = np.concatenate([p[m] for p, m in zip(pts3d, valid_mask)])
        col = np.concatenate([p[m] for p, m in zip(img, valid_mask)])
        pct = trimesh.PointCloud(pts, colors=col)

        # Save to a PLY file
        print('saving to local file.')
        pct.export('point_cloud.ply')
        print('saving to local file done.')

    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        if self.opt.data_mode == 'dust3r':
            B, V, C, H, W = data['images'][0]['img'].shape
            data['images'][0]['img'] = data['images'][0]['img'].view(B*V, C, H, W)
            data['images'][1]['img'] = data['images'][1]['img'].view(B*V, C, H, W)
            data['images'][0]['true_shape'] = data['images'][0]['true_shape'][0]
            data['images'][1]['true_shape'] = data['images'][1]['true_shape'][0]

            images = data['images'][:2]

            masks_input = data['masks_output'][:, :4, ...]
            masks_input = (masks_input > 0.5).float()
            mask1 = masks_input[:, :self.opt.num_input_views][:, ::2]
            mask2 = masks_input[:, :self.opt.num_input_views][:, 1::2]

            pose1 = data['extrinsics'][:, :self.opt.num_input_views][:, ::2]
            pose2 = data['extrinsics'][:, :self.opt.num_input_views][:, 1::2]
            pose1 = pose1.reshape(B*V, 4, 4)
            pose2 = pose2.reshape(B*V, 4, 4)
            # depth1 = pred1['means'].view(B*V, H, W, 3)[..., 2]
            # depth2 = pred2['means_in_other_view'].view(B*V, H, W, 3)[..., 2]

            # self.opt.mv_head = True
            if self.opt.mv_head:
                pred = self.forward_mast3r_mv(images[0], images[1], data)

                # unet start
                # images_inp = torch.cat([images[0]['img'], images[1]['img']], dim=0)
                # unet_inp = torch.cat([pred, images_inp], dim=1)
                # pred = pred + self.unet(unet_inp)
                # unet end

                pos = pred[:, :3, ...].reshape(B*V, -1, 3)
                opacity = pred[:, 3:4, ...].reshape(B*V, -1, 1)
                scale = pred[:, 4:7, ...].reshape(B*V, -1, 3)
                rotation = pred[:, 7:11, ...].reshape(B*V, -1, 4)
                rgbs = pred[:, 11:14, ...].reshape(B*V, -1, 3)

                mask1_flat = mask1.squeeze(2).reshape(-1, H, W).reshape(B, -1)
                mask2_flat = mask2.squeeze(2).reshape(-1, H, W).reshape(B, -1)
                pos_input_cat = pos.reshape(B, -1, 3)
                mask_input_cat = torch.cat([mask1_flat, mask2_flat], dim=1)

            else:
                pred1, pred2 = self.forward_mast3r(images[0], images[1], data)

                # depthhead start
                # depth1 = pred1[:, :3, ...].reshape(B*V, H, W, 3)[..., 2]
                # depth2 = pred2[:, :3, ...].reshape(B*V, H, W, 3)[..., 2]
                # pts1 = self.depthmap_to_absolute_camera_coordinates(depth1, data['intrinsics'][0][0], pose1)
                # pts2 = self.depthmap_to_absolute_camera_coordinates(depth2, data['intrinsics'][0][0], pose2)
                # pts1 = normalize_point_cloud(pts1.reshape(B*V, -1, 3), 2.0)
                # pts2 = normalize_point_cloud(pts2.reshape(B*V, -1, 3), 2.0)
                # pos = torch.cat([pts1, pts2], dim=0)
                # pos = pos.view(2*B*V, -1, 3)
                # depthhead end

                # pointhead start
                pts1 = pred1[:, :3, ...].reshape(B*V, -1, 3)
                pts2 = pred2[:, :3, ...].reshape(B*V, -1, 3)
                pos = torch.cat([pts1, pts2], dim=0)
                # pointhead end

                pts1_flat = pts1.reshape(B, -1, 3)
                pts2_flat = pts2.reshape(B, -1, 3)
                mask1_flat = mask1.squeeze(2).reshape(-1, H, W).reshape(B, -1)
                mask2_flat = mask2.squeeze(2).reshape(-1, H, W).reshape(B, -1)
                pos_input_cat = torch.cat([pts1_flat, pts2_flat], dim=1)
                mask_input_cat = torch.cat([mask1_flat, mask2_flat], dim=1)
                # pos_input_cat = torch.cat([pred1[:, :3, ...], pred2[:, :3, ...]], dim=0).reshape(B, self.opt.num_input_views, 3, H, W)
                # mask_input_cat = torch.cat([mask1, mask2], dim=1)

                opacity = torch.cat([pred1[:, 3:4, ...].reshape(B*V, -1, 1), pred2[:, 3:4, ...].reshape(B*V, -1, 1)])
                # opacity = torch.cat([mask1.reshape(B*V, -1, 1), mask2.reshape(B*V, -1, 1)])
                scale = torch.cat([pred1[:, 4:7, ...].reshape(B*V, -1, 3), pred2[:, 4:7, ...].reshape(B*V, -1, 3)])
                rotation = torch.cat([pred1[:, 7:11, ...].reshape(B*V, -1, 4), pred2[:, 7:11, ...].reshape(B*V, -1, 4)])
                rgbs = torch.cat([pred1[:, 11:14, ...].reshape(B*V, -1, 3), pred2[:, 11:14, ...].reshape(B*V, -1, 3)])

            pos = self.pos_act(pos)
            opacity = self.opacity_act(opacity) # masks_input # 
            scale = self.scale_act(scale)
            rotation = self.rot_act(rotation)
            rgbs = self.pos_act(rgbs) # color 

            gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
            gaussians = gaussians.view(B, -1, 14)

            sampled_points = self.random_sample_points(pos_input_cat, mask_input_cat, 10000)
            if sampled_points is None:
                chamfer_loss = 0
            else:
                chamfer_loss = self.chamfer_loss(sampled_points, data['point_cloud'])
                # chamfer_loss, _ = chamfer_distance(sampled_points, data['point_cloud'])

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]
        pred_depths = results['depth'] # [B, V, 1, output_size, output_size]

        pred_depths = minmax_norm(pred_depths)
        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['depths_pred'] = pred_depths
        
        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse
        if self.opt.lambda_chamfer > 0:
            loss = loss + self.opt.lambda_chamfer * chamfer_loss

        if self.opt.use_depth:
            depth = minmax_norm(pred[:, 2:3, ...])
            depth_input = data['depths_output'][:, :4, ...]
            depth_input1 = depth_input[:, :self.opt.num_input_views][:, ::2]
            depth_input2 = depth_input[:, :self.opt.num_input_views][:, 1::2]
            depth_input_gt = torch.cat([depth_input1, depth_input2], dim=1)
            depth_input_gt = depth_input_gt.reshape(-1, self.opt.output_size, self.opt.output_size)
            mask_input_gt = torch.cat([mask1, mask2], dim=1)
            mask_input_gt = mask_input_gt.reshape(-1, self.opt.output_size, self.opt.output_size)
            depth = depth.reshape(-1, self.opt.output_size, self.opt.output_size)
            loss_grad_depth = get_depth_grad_loss(depth, depth_input_gt, mask_input_gt, inverse_depth_loss=False)
            loss = loss + self.opt.lambda_depth * loss_grad_depth

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            ssim_score = ssim(pred_images.reshape(-1, 3, self.opt.output_size, self.opt.output_size).cpu(), gt_images.reshape(-1, 3, self.opt.output_size, self.opt.output_size).cpu())
            # psnr_score = psnr_metric(pred_images.detach(), gt_images)
            if self.opt.lambda_lpips > 0:
                results['lpips'] = loss_lpips
            results['ssim'] = ssim_score.item()
            results['psnr'] = psnr
            results['gaussians'] = gaussians

        return results
