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
    local_views_path_json: str = '/workspace/code/objaverse-rendering/valid_views.json'
    eval_views_path_json: str = '/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241107.json'
    camera_augmentation: bool = False

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

    ### additional settings
    use_depth_net: bool = False
    use_depth_offset: bool = False
    use_depth: bool = False
    use_geometry_net: bool = False

    # log_steps
    log_steps: int = 100

    # scaling dataset length
    dataset_scalar: float = 1.0
    
@dataclass
class Options_Triplane:
    ### model
    model_type = "Triplane"
    # Unet image input size
    input_size: int = 256
    # Unet output size, dependent on the input_size and U-Net structure!
    # gaussian render size
    output_size: int = 256
    # dino_encoder
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

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 67.38 # 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 8
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8
    # num_sample_points
    num_sample_points: int = 32768

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

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False
    

@dataclass
class Options_Mast3r:
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
    # geometry net
    use_geometry_net: bool = False
    # depth data
    use_depth: bool = False
    # use offset
    use_depth_offset: bool = True
    # use_depth_net
    use_depth_net: bool = False

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 'dust3r'
    # fovy of the dataset
    fovy: float = 67.38 # 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 8
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8
    # local_views_path
    local_views_path_json: str = '/workspace/code/objaverse-rendering/valid_views.json'
    # local_views_path
    eval_views_path_json: str = '/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241107.json'
    # local_models_path
    local_models_path_json: str = '/workspace/code/objaverse-rendering/valid_views_glb.json'
    # shape_splat_path
    ply_files_path: str = '/workspace/datasets/ShapeSplatsV1/ply_files'
    # camera augnmentation
    camera_augmentation: bool = False

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
    # log_steps
    log_steps: int = 100

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False
    

@dataclass
class Options_MASt3RGS:
    ### model
    model_type = "Mast3RGS"
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
    # geometry net
    use_geometry_net: bool = False
    # depth data
    use_depth: bool = False
    # use offset
    use_depth_offset: bool = True
    # use_depth_net
    use_depth_net: bool = False
    # use mv_head
    mv_head: bool = False

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 'dust3r'
    # fovy of the dataset
    fovy: float = 67.38 # 49.1
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 8
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8
    # local_views_path
    local_views_path_json: str = '/workspace/code/objaverse-rendering/valid_views.json'
    # local_views_path
    eval_views_path_json: str = '/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241107.json'
    # local_models_path
    local_models_path_json: str = '/workspace/code/objaverse-rendering/valid_views_glb.json'
    # shape_splat_path
    ply_files_path: str = '/workspace/datasets/ShapeSplatsV1/ply_files'
    # camera augnmentation
    camera_augmentation: bool = False

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
    # log_steps
    log_steps: int = 100
    # chamfer loss weight
    lambda_chamfer: float = 1.0
    # depth loss weight
    lambda_depth: float = 1.0

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

config_doc['lpm'] = 'the default settings for lpm'
config_defaults['lpm'] = Options_Triplane(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

config_doc['lpm_a40'] = 'the default settings for lpm a40'
config_defaults['lpm_a40'] = Options_Triplane(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=2,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=100,
)

config_doc['mast3r'] = 'the default settings for lpm a40'
config_defaults['mast3r'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    eval_views_path_json='/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241120_4v.json'
)

config_doc['mast3r_depth'] = 'the default settings for lpm a40'
config_defaults['mast3r_depth'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=1,
    use_depth=True,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/WIP/views_lvis_depth_normal_241106.json",
    local_models_path_json=None
)

config_doc['mast3r_splat_128'] = 'the default settings for lpm a40'
config_defaults['mast3r_splat_128'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=1,
    use_depth=True,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=60,
    up_channels = (1024, 1024, 512, 256, 128),
    up_attention = (True, True, True, False, False),
    splat_size=128,
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/WIP/views_lvis_depth_normal_241107.json",
    local_models_path_json=None
)

config_doc['mast3r_splat_128_a40'] = 'the default settings for lpm a40'
config_defaults['mast3r_splat_128_a40'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=6,
    use_depth=True,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=60,
    up_channels = (1024, 1024, 512, 256, 128),
    up_attention = (True, True, True, False, False),
    splat_size=128,
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/WIP/views_lvis_depth_normal_241113.json",
    local_models_path_json=None
)

config_doc['mast3r_splat_128_gso_finetune_gso'] = 'the default settings for lpm a40'
config_defaults['mast3r_splat_128_gso_finetune_gso'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=3,
    use_depth=False,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=10,
    up_channels = (1024, 1024, 512, 256, 128),
    up_attention = (True, True, True, False, False),
    splat_size=128,
    local_views_path_json="/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_eval_gso_finetune.json",
    # local_views_path_json="/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241107.json",
    local_models_path_json=None
)

config_doc['mast3r_miku'] = 'the default settings for lpm a40'
config_defaults['mast3r_miku'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=4,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json"
)

config_doc['mast3r_gso'] = 'the default settings for lpm a40'
config_defaults['mast3r_gso'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=2,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    log_steps=10,
    local_views_path_json="/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241103_2.json",
    local_models_path_json=None
)

config_doc['mast3r_miku_65536'] = 'the default settings for lpm a40'
config_defaults['mast3r_miku_65536'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=4,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    up_channels = (1024, 1024, 512, 256, 128),
    up_attention = (True, True, True, False, False),
    splat_size=128,
    local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json"
)

config_doc['mast3r_geometry'] = 'the default settings for lpm a40'
config_defaults['mast3r_geometry'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=2,
    gradient_accumulation_steps=1,
    mixed_precision=None,
    use_geometry_net=True,
    num_epochs=100,
    up_channels = (1024, 1024, 512, 256, 128),
    up_attention = (True, True, True, False, False),
    splat_size=128,
    local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json"
)

config_doc['mast3r_geometry_splat_128'] = 'the default settings for lpm a40'
config_defaults['mast3r_geometry_splat_128'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=3,
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
    use_geometry_net=True,
    num_epochs=100,
    up_channels = (1024, 1024, 512, 256, 128),
    up_attention = (True, True, True, False, False),
    splat_size=128,
    local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json"
)

config_doc['mast3r_depth_splat_128'] = 'the default settings for lpm a40'
config_defaults['mast3r_depth_splat_128'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
    use_depth=True,
    num_epochs=100,
    up_channels = (1024, 1024, 512, 256, 128),
    up_attention = (True, True, True, False, False),
    splat_size=128,
    log_steps=50,
    # local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    # local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json"
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/lvis_models_path_46000_46207.json",
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/valid_views_miku_depth_random_lightning.json",
    local_models_path_json=None
)

config_doc['mast3r_depth_splat_128_a40'] = 'the default settings for lpm a40'
config_defaults['mast3r_depth_splat_128_a40'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=6,
    gradient_accumulation_steps=1,
    mixed_precision="bf16",
    use_depth=True,
    num_epochs=120,
    up_channels = (1024, 1024, 512, 256, 128),
    up_attention = (True, True, True, False, False),
    splat_size=128,
    log_steps=100,
    # local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    # local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json"
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/lvis_models_path_46000_46207.json",
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/lvis_models_path_45000_46000.json",
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/valid_views_miku_depth_random_lightning.json",
    lr=1e-4,
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/WIP/views_lvis_depth_normal_241105.json",
    local_models_path_json=None
)

config_doc['mast3r_a40'] = 'the default settings for lpm a40'
config_defaults['mast3r_a40'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=60,
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lgm_lvis_241010.json",
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lgm_lvis_241101_2.json",
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lgm_lvis_241113.json",
    local_models_path_json=None
)

config_doc['mast3r_a40_depth'] = 'the default settings for lpm a40'
config_defaults['mast3r_a40_depth'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=4,
    num_views=8,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=60,
    use_depth=True,
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lgm_lvis_241010.json",
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lgm_lvis_241101_2.json",
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/WIP/views_lvis_depth_normal_241106.json",
    local_models_path_json=None
)

config_doc['mast3r_6v'] = 'the default settings for lpm a40'
config_defaults['mast3r_6v'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=6,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    # local_views_path_json="/workspace/code/objaverse-rendering/valid_views.json",
    # local_models_path_json="/workspace/code/objaverse-rendering/valid_views_glb.json"
    local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json"
)

config_doc['mast3r_6v_eval'] = 'the default settings for lpm a40'
config_defaults['mast3r_6v_eval'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=6,
    num_views=10,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    # local_views_path_json="/workspace/code/objaverse-rendering/valid_views.json",
    # local_models_path_json="/workspace/code/objaverse-rendering/valid_views_glb.json"
    local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json",
    eval_views_path_json='/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_test_241120_6v.json'
)

config_doc['mast3r_8v'] = 'the default settings for lpm a40'
config_defaults['mast3r_8v'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=8,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    # local_views_path_json="/workspace/code/objaverse-rendering/valid_views.json",
    # local_models_path_json="/workspace/code/objaverse-rendering/valid_views_glb.json"
    local_views_path_json="/workspace/code/objaverse-rendering/valid_views_miku.json",
    local_models_path_json="/workspace/code/objaverse-rendering/local_models_path_miku.json"
)

config_doc['mast3r_6v_long'] = 'the default settings for lpm a40'
config_defaults['mast3r_6v_long'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=6,
    num_views=8,
    batch_size=3,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_241008.json",
    # local_views_path_json="/workspace/code/objaverse-rendering/valid_views.json",
    local_models_path_json=None
)

config_doc['mast3r_a40_6v'] = 'the default settings for lpm a40'
config_defaults['mast3r_a40_6v'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=6,
    num_views=8,
    batch_size=7,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    camera_augmentation = True,
    num_epochs=100,
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lgm_lvis_241010.json",
    # local_views_path_json="/workspace/code/objaverse-rendering/valid_views.json",
    local_models_path_json=None
)

config_doc['mast3r_6v_shapesplat'] = 'the default settings for lpm a40'
config_defaults['mast3r_6v_shapesplat'] = Options_Mast3r(
    input_size=256,
    output_size=256,
    num_input_views=6,
    num_views=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    num_epochs=100,
    data_mode="shapesplat",
    ply_files_path="/workspace/datasets/ShapeSplatsV1/ply_files"
)

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
    use_depth=True,
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


config_doc['tiny'] = 'tiny model for ablation'
config_defaults['tiny'] = Options(
    input_size=256, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    splat_size=64,
    output_size=256,
    batch_size=16,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

config_doc['mast3rgs'] = 'the default settings for lpm a40'
config_defaults['mast3rgs'] = Options_MASt3RGS(
    input_size=256,
    output_size=128,
    num_input_views=4,
    num_views=8,
    num_workers=2,
    batch_size=1,
    use_depth=True,
    mv_head=True,
    lambda_lpips=1,
    lambda_chamfer=0,
    lambda_depth=0,
    down_channels=(64, 128, 256, 512, 1024),
    down_attention=(True, True, True, True, True),
    mid_attention=True,
    up_channels=(1024, 512, 256, 128, 64),
    up_attention=(True, True, True, True, True),
    gradient_accumulation_steps=1,
    mixed_precision='no',
    log_steps=10,
    num_epochs=100,
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/valid_views_miku_depth.json",
    # local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/WIP/views_lvis_depth_normal_241101.json",
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal/lvis_models_path_0_1000.json",
    # local_views_path_json="/workspace/datasets/Google_Scanned_Objects/json_files/gso_lgm_eval_gso_finetune.json",
    # local_views_path_json='/workspace/code/objaverse-rendering/views_depth_fixed/views_depth_fixed.json',
    local_models_path_json=None
)

config_doc['lgm_wam_v100_fulldata'] = 'big model with higher resolution Gaussians'
config_defaults['lgm_wam_v100_fulldata'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=256, # render & supervise Gaussians at a higher resolution.
    batch_size=2,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
    log_steps=10,
    num_epochs=10,
    local_views_path_json="/workspace/code/objaverse-rendering/json_files/views_lvis_depth_normal_241115.json",
    dataset_scalar=0.1
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
