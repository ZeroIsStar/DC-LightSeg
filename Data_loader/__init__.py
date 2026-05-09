import importlib
import torch.utils.data as data_utils
from Data_loader.Bijie_dataset import Bijie_Dataset
from Data_loader.Nepal_dataset import Nepal_Dataset


class Struct(dict):
    def __getattr__(self, item):
        try:
            value = self[item]
            if type(value) == type({}):
                return Struct(value)
            return value
        except KeyError:
            raise AttributeError(item)

    def set_cd_cfg_from_file(cfg_path=r'configs/configs.py'):
        module_spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        cfg = module.cfg
        cfg = Struct(cfg)
        return cfg


cfg = Struct.set_cd_cfg_from_file()


class DataLoader:
    def __init__(self, dataset_name):
        self.dataset       = dataset_name
        self.Bijie_Dataset = Bijie_Dataset
        self.Nepal = Nepal_Dataset
        self.Train_dataset = None
        self.Val_dataset   = None
        self.Test_dataset  = None
        if dataset_name == 'Bijie':
            landslide_dir = r'Bijie'
            self.Train_dataset = self.Bijie_Dataset(dir=landslide_dir, set=cfg.dataset.set[0])
            self.Val_dataset   = self.Bijie_Dataset(dir=landslide_dir, set=cfg.dataset.set[1])
            self.Test_dataset  = self.Bijie_Dataset(dir=landslide_dir, set=cfg.dataset.set[2])
        elif dataset_name == 'Nepal':
            landslide_dir = r'Nepal_landslide_dataset'
            self.Train_dataset = self.Nepal(dir=landslide_dir, set=cfg.dataset.set[0])
            self.Val_dataset   = self.Nepal(dir=landslide_dir, set=cfg.dataset.set[1])
            self.Test_dataset  = self.Nepal(dir=landslide_dir, set=cfg.dataset.set[2])

    def get_dataloader(self, batch_size=cfg.dataset.batch_size):
        train_loader = data_utils.DataLoader(self.Train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = data_utils.DataLoader(self.Val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader  = data_utils.DataLoader(self.Test_dataset, batch_size=1, shuffle=False)
        return train_loader, val_loader, test_loader



