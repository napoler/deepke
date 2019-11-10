import os
import hydra
from hydra import utils
import torch
from utils import load_pkl
from serializer import Serializer
from preprocess import _serialize_sentence, _convert_tokens_into_index,_add_pos_seq

sentence = '《香椿爱情》这是一部非常棒的电视剧，由知名导演赵本山所拍摄的乡村爱情故事。'
head = '香椿爱情'
tail = '赵本山'
head_type = '影视作品'
tail_type = '人物'

instance = {'sentence': sentence, 'head': head, 'head_type': head_type, 'tail': tail, 'tail_type': tail_type}

print(instance)

@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    vocab = load_pkl(os.path.join(cwd, 'data/out/vocab.pkl'),verbose=False)
    serializer = Serializer(do_chinese_split=cfg.chinese_split)
    serial = serializer.serialize


    data = list()
    data.append(instance)
    _serialize_sentence(data,serial,cfg)
    _convert_tokens_into_index(data,vocab)
    _add_pos_seq(data,cfg)
    print(data[0]['tokens'])



if __name__ =='__main__':
    main()
    # python predict.py --help
    # python predict.py -c
    # python predict.py chinese_split=0,1 replace_entity_with_type=0,1 -m
