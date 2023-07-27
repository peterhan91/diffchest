from templates import *


def ffhq128_autoenc_cls():
    conf = ffhq128_autoenc_130M()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    conf.name = 'ffhq128_autoenc_cls'
    return conf


def ffhq256_autoenc_cls():
    '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = ffhq256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.name = 'ffhq256_autoenc_cls'
    return conf


def chexpert256_autoenc_cls():
    conf = chexpert256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.chexpert_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{chexpert256_autoenc().name}/latent.pkl'  # we train on chexpert dataset, not mimic
    conf.batch_size = 16
    conf.lr = 1e-3
    conf.total_samples = int(15 * 192000) # 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '90M',
        f'checkpoints/{chexpert256_autoenc().name}/last.ckpt',
    )
    conf.name = 'chexpert256_autoenc_cls'
    return conf


def chexpert256_autoenc_cls_19200k():
    conf = chexpert256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.chexpert_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{chexpert256_autoenc().name}/latent.pkl'  # we train on chexpert dataset, not mimic
    conf.batch_size = 16
    conf.lr = 1e-3
    conf.total_samples = int(15 * 192000)
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '90M',
        f'checkpoints/{chexpert256_autoenc().name}/last.ckpt',
    )
    conf.name = 'chexpert256_autoenc_cls_19200k'
    return conf


def chexpert256_autoenc_cls_subsample():
    conf = chexpert256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.chexpert_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{chexpert256_autoenc().name}/latent.pkl'  # we train on chexpert dataset, not mimic
    conf.do_subsample = True
    conf.subsample_ratio = 0.07
    conf.batch_size = 16
    conf.num_workers = 8
    conf.lr = 1e-3
    conf.total_samples = int(15 * 192000 * conf.subsample_ratio) # 191,000 one epoch 15 epoch in total
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '90M',
        f'checkpoints/{chexpert256_autoenc().name}/last.ckpt',
    )
    conf.name = 'chexpert256_autoenc_cls_subsample_%f' % conf.subsample_ratio
    return conf


def padchest256_autoenc_cls():
    conf = padchest256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.padchest_train
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{padchest256_autoenc().name}/latent.pkl'  # we train on chexpert dataset, not mimic
    conf.batch_size = 16
    conf.lr = 1e-3
    conf.total_samples = 600_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '90M',
        f'checkpoints/{padchest256_autoenc().name}/last.ckpt',
    )
    conf.name = 'padchest256_autoenc_cls'
    return conf


def ukachest256_autoenc_cls():
    conf = ukachest256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.uka_chest
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ukachest256_autoenc().name}/latent.pkl'  # we train on padchest dataset, not uka
    conf.batch_size = 16
    conf.lr = 1e-3
    conf.total_samples = 600_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '90M',
        f'checkpoints/{padchest256_autoenc().name}/last.ckpt',
    )
    conf.name = 'ukachest256_autoenc_cls'
    return conf


def mimic256_autoenc_cls():
    conf = mimic256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.mimic_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{mimic256_autoenc().name}/latent.pkl'  # we train on chexpert dataset, not mimic
    conf.batch_size = 16
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '90M',
        f'checkpoints/{mimic256_autoenc().name}/last.ckpt',
    )
    conf.name = 'mimic256_autoenc_cls'
    return conf


def mimic256_autoenc_cls_24000k():
    conf = mimic256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.mimic_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{mimic256_autoenc().name}/latent.pkl'  # we train on chexpert dataset, not mimic
    conf.batch_size = 16
    conf.lr = 1e-3
    conf.total_samples = 24_000_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '90M',
        f'checkpoints/{mimic256_autoenc().name}/last.ckpt',
    )
    conf.name = 'mimic256_autoenc_cls_24000k'
    return conf


def mimic256_autoenc_cls_finetune():
    conf = mimic256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.mimic_finetune
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{mimic256_autoenc().name}/latent.pkl'  # we train on chexpert dataset, not mimic
    conf.batch_size = 16
    conf.lr = 1e-4
    conf.total_samples = 24_000_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '90M',
        f'checkpoints/{mimic256_autoenc().name}/last.ckpt',
    )
    conf.name = 'mimic256_autoenc_cls_finetune'
    return conf
