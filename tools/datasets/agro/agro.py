import numpy as np

from datasets.BaseDataset import BaseDataset


class Agro(BaseDataset):
    trans_labels = [0, 1, 2, 3, 4, 5, 6, 7]

    @classmethod
    def get_class_colors(*args):
        return [[128, 64, 128], [244, 35, 232], [70, 70, 70],
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0]]

    @classmethod
    def get_class_names(*args):
        # class counting(gtFine)
        # 2953 2811 2934  970 1296 2949 1658 2808 2891 1654 2686 2343 1023 2832
        # 359  274  142  513 1646
        return ['bg', 'crop', 'windrow', 'egovehicle', 'mowedfield', 'car', 'pedestrian',
                'pole']

    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        print('Trans', name, 'to', new_name, '    ',
              np.unique(np.array(pred, np.uint8)), ' ---------> ',
              np.unique(np.array(label, np.uint8)))
        return label, new_name