import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 49.1 # 67.38 # 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 8
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # 2.0 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8
    # total views
    total_views_number: int = 32

    ### training
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 1
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'fp16'
    # learning rate
    lr: float = 4e-4
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None
    local_models_path_json: Optional[str] = None
    local_views_path_json: str = ''
    eval_views_path_json: str = ''
    camera_augmentation: bool = False

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

    ### additional settings
    # use_depth_net: bool = False
    # use_depth_offset: bool = False
    # use_depth: bool = False
    # use_geometry_net: bool = False

    # log_steps
    log_steps: int = 100

    # scaling dataset length
    dataset_scalar: float = 1.0

    # MarkSplatter
    marksplatter_ckpt_path: str = ''
    
# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    # fovy=67.38/2,
    fovy=49.1,
    output_size=256, # render & supervise Gaussians at a higher resolution.
    batch_size=1,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

config_doc['big_a40'] = 'big model with higher resolution Gaussians'
config_defaults['big_a40'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=480, # render & supervise Gaussians at a higher resolution.
    batch_size=1,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    log_steps=10,
    num_epochs=60,
    # local_views_path_json="/workspace/NGC-workspace/files.json",
    local_views_path_json='/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_eval_gso_finetune.json',
    # eval_views_path_json='/workspace/NGC-workspace/files.json'
    # eval_views_path_json='/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241107.json'
    # eval_views_path_json='/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_eval_gso_finetune.json',
)

config_doc['wam'] = 'big model with higher resolution Gaussians'
config_defaults['wam'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=480, # render & supervise Gaussians at a higher resolution.
    batch_size=1,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    log_steps=10,
    num_epochs=60,
    total_views_number=8,
    # local_views_path_json="/workspace/NGC-workspace/files.json",
    local_views_path_json='/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_eval_gso_finetune.json',
    # eval_views_path_json='/workspace/NGC-workspace/files.json'
    # eval_views_path_json='/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241107.json'
    # eval_views_path_json='/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_eval_gso_finetune.json',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
