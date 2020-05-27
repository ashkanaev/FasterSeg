import cv2
cv2.setNumThreads(0)
from torch.utils import data
from utils.img_utils import random_scale, random_mirror, normalize, generate_random_crop_pos, random_crop_pad_to_shape

from albumentations import (
    RandomSizedCrop,
    HorizontalFlip,
    Transpose,
    Compose,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    Blur,
    MotionBlur,
    MedianBlur,
    IAAEmboss,
    HueSaturationValue,
    Resize
)

def strong_aug(config, aug_prob):
    return Compose([
        # Resize(config.image_height, config.image_width, always_apply=True),

        RandomSizedCrop(p=config.random_sized_crop_prob, min_max_height=(int(config.image_height * config.min_max_height), config.image_height), height=config.image_height, width=config.image_width, w2h_ratio = config.image_width/config.image_height),

        HorizontalFlip(p=config.horizontal_flip_prob),

        RandomGamma(p=config.random_gamma_prob),
        RandomContrast(p=config.random_contrast_prob, limit=config.random_contrast_limit),
        RandomBrightness(p=config.random_brightness_prob, limit=config.random_brightness_limit),

        OneOf([
            MotionBlur(p=config.motion_blur_prob),
            MedianBlur(blur_limit=config.median_blur_limit, p=config.median_blur_prob),
            Blur(blur_limit=config.blur_limit, p=config.blur_prob),
        ], p=config.one_of_blur_prob),

        CLAHE(clip_limit=config.clahe_limit, p=config.clahe_prob),
        IAAEmboss(p=config.iaaemboss_prob),

        HueSaturationValue(p=config.hue_saturation_value_prob, hue_shift_limit = config.hue_shift_limit, sat_shift_limit = config.sat_shift_limit, val_shift_limit = config.val_shift_limit)

    ], p=aug_prob)


class TrainPre(object):
    def __init__(self, config, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std
        self.config = config
        self.augmenter = strong_aug(config, 0.9)

    def __call__(self, img, gt):
        # img, gt = random_mirror(img, gt)
        # if self.config.train_scale_array is not None:
        #     img, gt, scale = random_scale(img, gt, self.config.train_scale_array)


        # crop_size = (self.config.image_height, self.config.image_width)
        # crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)
        # p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        # p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        # p_gt = cv2.resize(p_gt, (self.config.image_width // self.config.gt_down_sampling, self.config.image_height // self.config.gt_down_sampling), interpolation=cv2.INTER_NEAREST)
        augment = self.augmenter(image=img, mask=gt)
        p_img, p_gt = augment['image'], augment['mask']
        img = normalize(p_img, self.img_mean, self.img_std)

        p_img = p_img.transpose(2, 0, 1)
        extra_dict = None

        return p_img, p_gt, extra_dict


def get_train_loader(config, dataset, portion=None, worker=None, test=False):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'down_sampling': config.down_sampling,
                    'portion': portion}
    if test:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_eval_source,
                        'eval_source': config.eval_source,
                        'down_sampling': config.down_sampling,
                        'portion': portion}
    train_preprocess = TrainPre(config, config.image_mean, config.image_std)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    is_shuffle = True
    batch_size = config.batch_size

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers if worker is None else worker,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True)

    return train_loader
