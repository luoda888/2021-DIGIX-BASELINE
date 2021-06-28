import os


class OCR_DatasetCatalog(object):
    def __init__(self, root_path=''):
        super(OCR_DatasetCatalog, self).__init__()
        self.root_path = root_path
        self.general_datasets = {
            'menu_train': {
                'root_path': 'train_image_common/',
                'gt_path': 'train_label_common.json',
            }
        }

    def get(self, name):
        if name in self.general_datasets.keys():
            return self.general_datasets[name]
        else:
            raise RuntimeError('Dataset not available: {}.'.format(name))


if __name__ == '__main__':
    ocr_dataset_catalog = OCR_DatasetCatalog()
    print(ocr_dataset_catalog.general_datasets.keys())
