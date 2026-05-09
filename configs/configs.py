import torch

cfg = dict(
    segment_model = dict(
        type ='MFFEnet',  # ['Unet','Unet++','MFFEnet','RIPF_Unet','Attention_deeplabV3plus','RFA_ResUnet','DC_light_bifpn_cat']
    ),

    dataset = dict(
        set          = ['train', 'val', 'test'],
        dataset_name = 'Bijie/Nepal',
        batch_size   = 4,
        in_channels= 3,
        Class=2,
        # clip_grad_value_ = 5.0  # 模型梯度裁剪
    ),

    optimizer = dict(
        type         = 'AdamW',  # ['SGD','AdamW']
       base_lr       = 2e-4,
        min_lr       = 1e-6,
        step_size    = 10,
        gamma        = 0.9,
        weight_decay = 5e-4,
        momentum     = 0.99
    ),
    train = dict(
        loss_function='Tversky_loss_lovasz',  # ['celoss','Tversky_loss_lovasz']
    ),

    scheduler=dict(
        #['linear', 'step', 'CosineAnnealingLR'] 内置
        epoch        = 100,
        type         = 'WarmupCosineAnnealingLR',
        warmup_epoch = 10
    )
)
