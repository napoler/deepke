import os
import torch
from torch import optim
import torch.nn as nn
import hydra
from hydra import utils
import logging
from torch.utils.data import DataLoader
# self
import models
from preprocess import preprocess
from dataset import CustomDataset, collate_fn
from trainer import train
from utils import manual_seed, load_pkl

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    # print(cfg.pretty())
    # todo
    # cwd api 目前还没发布，只能自己修改
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd

    manual_seed(cfg.seed)

    __Model__ = {
        'cnn': models.PCNN,
    }


    # device
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')
    logger.info(f'device: {device}')

    # preprocess(cfg)

    train_data_path = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_data_path = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_data_path = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    vocab_path = os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl')

    if cfg.model_name == 'lm':
        vocab_size = None
    else:
        vocab = load_pkl(vocab_path)
        vocab_size = vocab.count
    cfg.vocab_size = vocab_size

    train_dataset = CustomDataset(train_data_path)
    valid_dataset = CustomDataset(valid_data_path)
    test_dataset = CustomDataset(test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    model = __Model__[cfg.model_name](cfg)
    model.to(device)
    logger.info(f'\n {model}')

    optimizer = optim.Adam(model.parameters(),lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_macro_f1, best_macro_epoch = 0, 1
    best_micro_f1, best_micro_epoch = 0, 1
    best_macro_model, best_micro_model = '', ''
    logger.info('=' * 10 + ' Start training ' + '=' * 10)

    for epoch in range(1, cfg.epoch+1):
        pass
        # train(epoch, model, train_dataloader, optimizer, criterion, device, cfg)

if __name__ == '__main__':
    main()
