from experiment import *


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def ffhq64_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.scale_up_gpus(4)
    return conf


def ffhq64_autoenc():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.make_model_conf()
    return conf


def celeba64d2c_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_ddpm'
    return conf


def celeba64d2c_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_autoenc'
    return conf


def ffhq128_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq128_autoenc_base():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.scale_up_gpus(4)
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq256_autoenc():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc'
    return conf


def chexpert256_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'chexpert'
    conf.scale_up_gpus(3)
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 16
    conf.make_model_conf()
    conf.name = 'chexpert256_autoenc'
    return conf


def mimic256_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'mimic'
    conf.scale_up_gpus(3)
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 12
    # conf.eval_path = 'checkpoints/chexpert256_autoenc/last.ckpt'
    # conf.pretrain = PretrainConfig(
    #     name='76M',
    #     path=f'checkpoints/{chexpert256_autoenc().name}/last.ckpt')
    conf.make_model_conf()
    conf.name = 'mimic256_autoenc'
    return conf

def padchest256_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'padchest'
    conf.scale_up_gpus(4)
    conf.num_workers = 24
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 100_000_000
    conf.eval_ema_every_samples = 100_000_000
    conf.total_samples = 20_000_000
    conf.batch_size = 24
    # conf.eval_path = 'checkpoints/chexpert256_autoenc/last.ckpt'
    # conf.pretrain = PretrainConfig(
    #     name='90M',
    #     path=f'checkpoints/{mimic256_autoenc().name}/last.ckpt')
    conf.make_model_conf()
    conf.name = 'padchest256_autoenc'
    return conf


def ukachest256_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'uka_chest'
    conf.scale_up_gpus(1)
    conf.num_workers = 8
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 100_000_000
    conf.eval_ema_every_samples = 100_000_000
    conf.total_samples = 20_000_000
    conf.batch_size = 16
    # conf.eval_path = 'checkpoints/padchest256_autoenc/last.ckpt'
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/{padchest256_autoenc().name}/last.ckpt')
    conf.make_model_conf()
    conf.name = 'ukachest256_autoenc'
    return conf


def ffhq256_autoenc_eco():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc_eco'
    return conf


def ffhq128_ddpm_72M():
    conf = ffhq128_ddpm()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_ddpm_72M'
    return conf


def ffhq128_autoenc_72M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_autoenc_72M'
    return conf


def ffhq128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_ddpm_130M'
    return conf


def ffhq128_autoenc_130M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_autoenc_130M'
    return conf


def horse128_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_ddpm'
    return conf


def horse128_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_autoenc'
    return conf


def bedroom128_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom128_ddpm'
    return conf


def bedroom128_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom128_autoenc'
    return conf


def pretrain_celeba64d2c_72M():
    conf = celeba64d2c_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{celeba64d2c_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{celeba64d2c_autoenc().name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc72M():
    conf = ffhq128_autoenc_base()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{ffhq128_autoenc_72M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_72M().name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc130M():
    conf = ffhq128_autoenc_base()
    conf.pretrain = PretrainConfig(
        name='130M',
        path=f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    return conf


def pretrain_ffhq256_autoenc():
    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'
    return conf


def pretrain_chexpert256_autoenc():
    conf = chexpert256_autoenc()
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/{chexpert256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{chexpert256_autoenc().name}/latent.pkl'
    return conf


def pretrain_mimic256_autoenc():
    conf = mimic256_autoenc()
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/{chexpert256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{mimic256_autoenc().name}/latent.pkl'
    return conf


def pretrain_horse128():
    conf = horse128_autoenc()
    conf.pretrain = PretrainConfig(
        name='82M',
        path=f'checkpoints/{horse128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{horse128_autoenc().name}/latent.pkl'
    return conf


def pretrain_bedroom128():
    conf = bedroom128_autoenc()
    conf.pretrain = PretrainConfig(
        name='120M',
        path=f'checkpoints/{bedroom128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{bedroom128_autoenc().name}/latent.pkl'
    return conf
