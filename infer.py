
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline

import time

from PIL import Image

# WAM watermarking imports
# from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.metrics import bit_accuracy_inference
from watermark_anything.augmentation.augmenter import Augmenter
from einops import rearrange
from skimage.metrics import peak_signal_noise_ratio
from utils import init_model

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load image dream
pipe = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe = pipe.to(device)

# load rembg
bg_remover = rembg.new_session()

# Load WAM watermarking model
# exp_dir = "/workspace/code/watermark-anything/checkpoints"
# json_path = os.path.join(exp_dir, "params_cross_att.json")
json_path = "ckpts/params_cross_att.json"
# ckpt_path = os.path.join(exp_dir, 'wam_mit.pth') 
wam = init_model(json_path).to(device)
wam.scaling_w = 0.3

# Resume WAM checkpoint
#wam_ckpt = load_file("/workspace/code/LGM/workspace_debug/workspace_wam_debug_250212/model.safetensors", device='cpu')

if os.path.exists(opt.marksplatter_ckpt_path):
    wam_ckpt = load_file(opt.marksplatter_ckpt_path, device='cpu')
    print('Loading WAM checkpoint...')
    wam_state_dict = wam.state_dict()
    for k, v in wam_ckpt.items():
        if k in wam_state_dict: 
            if wam_state_dict[k].shape == v.shape:
                wam_state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {wam_state_dict[k].shape}, ignored.')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')
else:
    print(f'[WARN] WAM checkpoint path does not exist: {opt.marksplatter_ckpt_path}')
    raise FileNotFoundError(f"WAM checkpoint path does not exist: {opt.marksplatter_ckpt_path}")

# Generate random watermark message
wm_msgs = torch.randint(0, 2, (1, 32))

# Setup augmenter for robustness testing
augs = {
    'perspective': 1,
}
augs_params = {
    'perspective': {'min_distortion_scale': 0.1, "max_distortion_scale": 0.3},
}
masks = {
    "kind": "full",
}
augmenter = Augmenter(
    masks=masks,
    augs=augs,
    augs_params=augs_params
).eval()

# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    input_image = kiui.read_image(path, mode='uint8')

    # bg removal
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    
    # generate mv
    image = image.astype(np.float32) / 255.0

    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

    tik = time.time()
    mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    tok = time.time()
    print(f'[INFO] MVDream took {tok - tik:.2f}s')

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            tik = time.time()
            gaussians, rgb, alpha, pos, rotation, scale = model.forward_gaussians(input_image, return_all=True)
            tok = time.time()
            print(f'[INFO] LGM took {tok - tik:.2f}s')

            # Apply WAM watermarking to RGB features
            print('[INFO] Applying WAM watermarking...')
            B, V, C, H, W = rgb.shape
            rgb_feat = rgb.reshape(B*V, C, H, W) # B*V, 3, H, W
            opacity = alpha.reshape(B*V, 1, H, W) # B*V, 1, H, W  
            position = pos.reshape(B*V, C, H, W) # B*V, 3, H, W

            # Prepare watermark messages for all views
            wm_msgs_all = torch.cat([wm_msgs] * V * B, dim=0).cuda()
            
            # Embed watermarks
            wam_outputs = wam.embed(rgb_feat, wm_msgs_all, opacity, position)
            rgbs_w = wam_outputs['imgs_w']  # Watermarked RGB features
            
            # Test watermark detection on embedded features
            preds = wam.detect(rgbs_w)["preds"]
            mask_preds = F.sigmoid(preds[:, 0, :, :])
            mask_preds_ = F.sigmoid(preds[:, 0:1, :, :])
            bit_preds = preds[:, 1:, :, :]
            bit_accuracy_ = bit_accuracy_inference(
                bit_preds, 
                wam_outputs["msgs"],
                mask_preds_
            ).nanmean().item()
            print(f"[INFO] Watermark bit accuracy on gaussians: {bit_accuracy_:.4f}")

            # Replace RGB features in gaussians with watermarked versions
            rgbs_w_flat = rgbs_w.permute(0, 2, 3, 1).reshape(B, -1, 3) # B, H*W, 3
            
            # Save original gaussians before watermarking
            gaussians_original = gaussians.clone()
            gaussians[:, :, 11:] = rgbs_w_flat  # Replace RGB channels (11:14) with watermarked RGB

            # Test watermark on rendered images (similar to eval_wam.py)
            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
            
            # Render a few test views
            test_azimuths = [0, 90, 180, 270]
            for test_azi in test_azimuths:
                cam_poses = torch.from_numpy(orbit_camera(0, test_azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
                cam_poses[:, :3, 1:3] *= -1
                cam_view = torch.inverse(cam_poses).transpose(1, 2)
                cam_view_proj = cam_view @ proj_matrix
                cam_pos = - cam_poses[:, :3, 3]
                
                # Render watermarked image
                rendered_results = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)
                rendered_images = rendered_results['image'] # [1, 1, 3, H, W]
                
                # Reshape for detection
                reshaped_rendered = rearrange(rendered_images, 'b v c h w -> (b v) c h w')
                
                # Test watermark detection on rendered images
                preds_rendered = wam.detect(reshaped_rendered)["preds"]
                mask_preds_rendered = F.sigmoid(preds_rendered[:, 0, :, :])
                mask_preds_rendered_ = F.sigmoid(preds_rendered[:, 0:1, :, :])
                bit_preds_rendered = preds_rendered[:, 1:, :]
                
                bit_accuracy_rendered = bit_accuracy_inference(
                    bit_preds_rendered, 
                    wm_msgs.cuda(),
                    mask_preds_rendered_
                ).nanmean().item()
                print(f"[INFO] Watermark bit accuracy on rendered view {test_azi}°: {bit_accuracy_rendered:.4f}")
                
                # Test augmentation robustness
                imgs_aug, mask_targets, selected_aug = augmenter(reshaped_rendered, reshaped_rendered, mask_preds_rendered_)
                
                preds_aug = wam.detect(imgs_aug)["preds"]
                mask_preds_aug = F.sigmoid(preds_aug[:, 0, :, :])
                mask_preds_aug_ = F.sigmoid(preds_aug[:, 0:1, :, :])
                bit_preds_aug = preds_aug[:, 1:, :]
                
                bit_accuracy_aug = bit_accuracy_inference(
                    bit_preds_aug, 
                    wm_msgs.cuda(),
                    mask_preds_aug_
                ).nanmean().item()
                print(f"[INFO] Watermark bit accuracy after augmentation {test_azi}°: {bit_accuracy_aug:.4f}")

            # save splat images for debugging
            # kiui.vis.plot_image(rgb[0], save=True, prefix='splatter_images/rgb')
            # kiui.vis.plot_image(alpha[0], save=True, prefix='splatter_images/alpha')
            # kiui.vis.plot_image(pos[0], save=True, prefix='splatter_images/pos')
            # kiui.vis.plot_image(rotation[0], save=True, prefix='splatter_images/rotation')
            # kiui.vis.plot_image(scale[0], save=True, prefix='splatter_images/scale')
            # print("rotation[0]: ", rotation[0].shape)
            # print("scale[0]: ", scale[0].shape)

        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '_watermarked.ply'), prune=True)
        model.gs.save_ply(gaussians_original, os.path.join(opt.workspace, name + '_original.ply'), prune=True)

        # Save some comparison images (original vs watermarked)
        os.makedirs(os.path.join(opt.workspace, 'comparison'), exist_ok=True)
        
        # Set elevation for camera positioning
        elevation = 0
        
        # Render a few comparison views and compute PSNR
        comparison_azimuths = [0, 90, 180, 270]
        psnr_scores = []
        
        for i, azi in enumerate(comparison_azimuths):
            cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1
            cam_view = torch.inverse(cam_poses).transpose(1, 2)
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = - cam_poses[:, :3, 3]
            
            # Render original image
            orig_image = model.gs.render(gaussians_original, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            orig_image_np = (orig_image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)[0]
            
            # Render watermarked image
            wm_image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            wm_image_np = (wm_image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)[0]
            
            # Get mask prediction for watermarked image
            rendered_img = wm_image.squeeze(1)  # [1, 3, H, W]
            preds_rendered = wam.detect(rendered_img)["preds"]
            mask_preds_rendered = F.sigmoid(preds_rendered[:, 0:1, :, :])
            mask_vis = mask_preds_rendered.squeeze().cpu().numpy()  # [H, W]
            mask_vis = (mask_vis * 255).astype(np.uint8)  # Convert to 0-255
            mask_vis = np.stack([mask_vis, mask_vis, mask_vis], axis=-1)  # Convert to RGB
            
            # Compute difference
            diff_image = np.abs(wm_image_np.astype(np.float32) - orig_image_np.astype(np.float32))
            diff_image = np.clip(diff_image * 10, 0, 255).astype(np.uint8)  # Amplify differences
            
            # Compute PSNR
            psnr_val = peak_signal_noise_ratio(orig_image_np, wm_image_np)
            psnr_scores.append(psnr_val)
            
            # Save comparison images
            imageio.imwrite(os.path.join(opt.workspace, 'comparison', f'original_view_{azi}.png'), orig_image_np)
            imageio.imwrite(os.path.join(opt.workspace, 'comparison', f'watermarked_view_{azi}.png'), wm_image_np)
            imageio.imwrite(os.path.join(opt.workspace, 'comparison', f'difference_view_{azi}.png'), diff_image)
            imageio.imwrite(os.path.join(opt.workspace, 'comparison', f'mask_prediction_view_{azi}.png'), mask_vis)
            
            # Create side-by-side comparison (4-way: original | watermarked | difference | mask)
            comparison = np.concatenate([orig_image_np, wm_image_np, diff_image, mask_vis], axis=1)
            imageio.imwrite(os.path.join(opt.workspace, 'comparison', f'comparison_view_{azi}.png'), comparison)
            
            print(f'[INFO] Saved comparison view at azimuth {azi}, PSNR: {psnr_val:.2f} dB')
        
        avg_psnr = np.mean(psnr_scores)
        print(f'[INFO] Average PSNR across comparison views: {avg_psnr:.2f} dB')

        # render 360 video 
        images_original = []
        images_watermarked = []
        images_difference = []
        images_mask_pred = []
        elevation = 0
        
        # Track watermark detection accuracy across all frames
        all_bit_accuracies = []
        video_psnr_scores = []

        if opt.fancy_video:

            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                # Render original and watermarked images
                orig_image = model.gs.render(gaussians_original, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                wm_image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                
                # Convert to numpy
                orig_image_np = (orig_image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                wm_image_np = (wm_image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                
                # Compute difference
                diff_image_np = np.abs(wm_image_np.astype(np.float32) - orig_image_np.astype(np.float32))
                diff_image_np = np.clip(diff_image_np * 10, 0, 255).astype(np.uint8)  # Amplify differences
                
                # Compute PSNR for this frame
                frame_psnr = peak_signal_noise_ratio(orig_image_np[0], wm_image_np[0])
                video_psnr_scores.append(frame_psnr)
                
                images_original.append(orig_image_np)
                images_watermarked.append(wm_image_np)
                images_difference.append(diff_image_np)
                
                # Test watermark detection on rendered images
                rendered_img = wm_image.squeeze(1)  # [1, 3, H, W]
                preds_rendered = wam.detect(rendered_img)["preds"]
                mask_preds_rendered = F.sigmoid(preds_rendered[:, 0:1, :, :])
                bit_preds_rendered = preds_rendered[:, 1:, :, :]
                bit_accuracy_rendered = bit_accuracy_inference(
                    bit_preds_rendered, 
                    wm_msgs.cuda(),
                    mask_preds_rendered
                ).nanmean().item()
                all_bit_accuracies.append(bit_accuracy_rendered)
                
                # Convert mask prediction to visualization
                mask_vis = mask_preds_rendered.squeeze().cpu().numpy()  # [H, W]
                mask_vis = (mask_vis * 255).astype(np.uint8)  # Convert to 0-255
                mask_vis = np.stack([mask_vis, mask_vis, mask_vis], axis=-1)  # Convert to RGB
                mask_vis = mask_vis[np.newaxis, ...]  # Add batch dimension [1, H, W, 3]
                images_mask_pred.append(mask_vis)
                
                if azi % 90 == 0:  # Print accuracy every 90 degrees
                    print(f"[INFO] Watermark bit accuracy at azimuth {azi}: {bit_accuracy_rendered:.4f}, PSNR: {frame_psnr:.2f} dB")
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                # Render original and watermarked images
                orig_image = model.gs.render(gaussians_original, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                wm_image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                
                # Convert to numpy
                orig_image_np = (orig_image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                wm_image_np = (wm_image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                
                # Compute difference
                diff_image_np = np.abs(wm_image_np.astype(np.float32) - orig_image_np.astype(np.float32))
                diff_image_np = np.clip(diff_image_np * 10, 0, 255).astype(np.uint8)  # Amplify differences
                
                # Compute PSNR for this frame
                frame_psnr = peak_signal_noise_ratio(orig_image_np[0], wm_image_np[0])
                video_psnr_scores.append(frame_psnr)
                
                images_original.append(orig_image_np)
                images_watermarked.append(wm_image_np)
                images_difference.append(diff_image_np)
                
                # Test watermark detection on rendered images
                rendered_img = wm_image.squeeze(1)  # [1, 3, H, W]
                preds_rendered = wam.detect(rendered_img)["preds"]
                mask_preds_rendered = F.sigmoid(preds_rendered[:, 0:1, :, :])
                bit_preds_rendered = preds_rendered[:, 1:, :, :]
                bit_accuracy_rendered = bit_accuracy_inference(
                    bit_preds_rendered, 
                    wm_msgs.cuda(),
                    mask_preds_rendered
                ).nanmean().item()
                all_bit_accuracies.append(bit_accuracy_rendered)
                
                # Convert mask prediction to visualization
                mask_vis = mask_preds_rendered.squeeze().cpu().numpy()  # [H, W]
                mask_vis = (mask_vis * 255).astype(np.uint8)  # Convert to 0-255
                mask_vis = np.stack([mask_vis, mask_vis, mask_vis], axis=-1)  # Convert to RGB
                mask_vis = mask_vis[np.newaxis, ...]  # Add batch dimension [1, H, W, 3]
                images_mask_pred.append(mask_vis)
                
                if azi % 90 == 0:  # Print accuracy every 90 degrees
                    print(f"[INFO] Watermark bit accuracy at azimuth {azi}: {bit_accuracy_rendered:.4f}, PSNR: {frame_psnr:.2f} dB")

        tok = time.time()
        print(f'[INFO] LGM total took {tok - tik:.2f}s')

        # Concatenate images and save videos
        images_original = np.concatenate(images_original, axis=0)
        images_watermarked = np.concatenate(images_watermarked, axis=0)
        images_difference = np.concatenate(images_difference, axis=0)
        images_mask_pred = np.concatenate(images_mask_pred, axis=0)
        
        # Save individual videos
        imageio.mimwrite(os.path.join(opt.workspace, name + '_original.mp4'), images_original, fps=30)
        imageio.mimwrite(os.path.join(opt.workspace, name + '_watermarked.mp4'), images_watermarked, fps=30)
        imageio.mimwrite(os.path.join(opt.workspace, name + '_difference.mp4'), images_difference, fps=30)
        imageio.mimwrite(os.path.join(opt.workspace, name + '_mask_prediction.mp4'), images_mask_pred, fps=30)
        
        # Create side-by-side comparison video (4-way: original | watermarked | difference | mask)
        images_comparison = np.concatenate([images_original, images_watermarked, images_difference, images_mask_pred], axis=2)
        imageio.mimwrite(os.path.join(opt.workspace, name + '_comparison.mp4'), images_comparison, fps=30)
        
        # Compute and report statistics
        avg_video_psnr = np.mean(video_psnr_scores)
        avg_bit_accuracy = np.mean(all_bit_accuracies)
        min_bit_accuracy = np.min(all_bit_accuracies)
        max_bit_accuracy = np.max(all_bit_accuracies)
        std_bit_accuracy = np.std(all_bit_accuracies)
        
        print(f'[INFO] Watermarked 3D content generation complete!')
        print(f'[INFO] Saved original PLY: {name}_original.ply')
        print(f'[INFO] Saved watermarked PLY: {name}_watermarked.ply')
        print(f'[INFO] Saved original video: {name}_original.mp4')
        print(f'[INFO] Saved watermarked video: {name}_watermarked.mp4')
        print(f'[INFO] Saved difference video: {name}_difference.mp4')
        print(f'[INFO] Saved mask prediction video: {name}_mask_prediction.mp4')
        print(f'[INFO] Saved comparison video: {name}_comparison.mp4')
        print(f'[INFO] Watermark message: {wm_msgs.squeeze().tolist()}')
        print(f'[INFO] ==================== STATISTICS ====================')
        print(f'[INFO] Average PSNR across all frames: {avg_video_psnr:.2f} dB')
        print(f'[INFO] Average watermark bit accuracy: {avg_bit_accuracy:.4f}')
        print(f'[INFO] Min watermark bit accuracy: {min_bit_accuracy:.4f}')
        print(f'[INFO] Max watermark bit accuracy: {max_bit_accuracy:.4f}')
        print(f'[INFO] Std watermark bit accuracy: {std_bit_accuracy:.4f}')
        print(f'[INFO] Total frames analyzed: {len(all_bit_accuracies)}')
        print(f'[INFO] ====================================================')
        
        # Save statistics to file
        stats = {
            'watermark_message': wm_msgs.squeeze().tolist(),
            'avg_psnr': avg_video_psnr,
            'avg_bit_accuracy': avg_bit_accuracy,
            'min_bit_accuracy': min_bit_accuracy,
            'max_bit_accuracy': max_bit_accuracy,
            'std_bit_accuracy': std_bit_accuracy,
            'total_frames': len(all_bit_accuracies),
            'all_bit_accuracies': all_bit_accuracies,
            'all_psnr_scores': video_psnr_scores
        }
        
        import json
        with open(os.path.join(opt.workspace, name + '_watermark_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

if __name__ == '__main__':
    assert opt.test_path is not None
    if os.path.isdir(opt.test_path):
        file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    else:
        file_paths = [opt.test_path]
    for path in file_paths:
        process(opt, path)
