import os
import logging
from typing import List, Dict, Tuple
from transformers import BertTokenizer
# self
from serializer import Serializer
from vocab import Vocab
from utils import save_pkl, load_csv

logger = logging.getLogger(__name__)


def _handle_pos_limit(pos: List[int], limit: int) -> List[int]:
    for i, p in enumerate(pos):
        if p > limit:
            pos[i] = limit
        if p < -limit:
            pos[i] = -limit
    return [p + limit + 1 for p in pos]


def _add_pos_seq(train_data: List[Dict], cfg):
    for d in train_data:
        entities_idx = [d['head_idx'], d['tail_idx']] if d['head_idx'] < d['tail_idx'] else [d['tail_idx'], d['head_idx']]

        d['head_pos'] = list(map(lambda i: i - d['head_idx'], list(range(d['seq_len']))))
        d['head_pos'] = _handle_pos_limit(d['head_pos'], int(cfg.pos_limit))

        d['tail_pos'] = list(map(lambda i: i - d['tail_idx'], list(range(d['seq_len']))))
        d['tail_pos'] = _handle_pos_limit(d['tail_pos'], int(cfg.pos_limit))

        d['entities_pos'] = [1] * (entities_idx[0] + 1) + [2] * (entities_idx[1] - entities_idx[0] - 1) + [3] * (d['seq_len'] - entities_idx[1])


def _convert_tokens_into_index(data: List[Dict], vocab):
    unk_str = '[UNK]'
    unk_idx = vocab.word2idx[unk_str]

    for d in data:
        d['token2idx'] = [vocab.word2idx.get(i, unk_idx) for i in d['tokens']]
        d['seq_len'] = len(d['token2idx'])


def _serialize_sentence(data: List[Dict], serial, cfg):
    for d in data:
        sent = d['sentence'].strip()

        if cfg.replace_entity_with_type:
            if cfg.chinese_split:
                sent = sent.replace(d['head'], 'HEAD', 1).replace(d['tail'], 'TAIL', 1)
                d['tokens'] = serial(sent, never_split=['HEAD', 'TAIL'])
                head_idx, tail_idx = d['tokens'].index('HEAD'), d['tokens'].index('TAIL')
                d['tokens'][head_idx], d['tokens'][tail_idx] = d['head_type'], d['tail_type']
            else:
                sent = sent.replace(d['head'], d['head_type'], 1).replace(d['tail'], d['tail_type'], 1)
                d['tokens'] = serial(sent)
                head_idx, tail_idx = sent.index(d['head_type']), sent.index(d['tail_type'])
            d['head_idx'], d['tail_idx'] = head_idx, tail_idx
        else:
            if cfg.chinese_split:
                sent = sent.replace(d['head'], 'HEAD', 1).replace(d['tail'], 'TAIL', 1)
                d['tokens'] = serial(sent, never_split=['HEAD', 'TAIL'])
                head_idx, tail_idx = d['tokens'].index('HEAD'), d['tokens'].index('TAIL')
                d['tokens'][head_idx], d['tokens'][tail_idx] = d['head'], d['tail']
            else:
                d['tokens'] = serial(sent)
                head_idx, tail_idx = sent.index(d['head']), sent.index(d['tail'])
            d['head_idx'], d['tail_idx'] = head_idx, tail_idx


def _lm_serialize(data: List[Dict], cfg):
    logger.info('use bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm_file)
    for d in data:
        sent = d['sentence'].strip()
        sent = sent.replace(d['head'], d['head_type'], 1).replace(d['tail'], d['tail_type'], 1)
        sent += '[SEP]' + d['head'] + '[SEP]' + d['tail']
        d['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)


def _handle_relation_data(relation_data: List[Dict]) -> Tuple:
    rels = dict()
    for d in relation_data:
        rels[d['relation']] = d['index']

    heads = [d['head'] for d in relation_data][1:]
    tails = [d['tail'] for d in relation_data][1:]

    return rels, heads, tails


def preprocess(cfg):

    logger.info('===== start preprocess data =====')
    train_fp = os.path.join(cfg.cwd, cfg.data_path, 'train.csv')
    valid_fp = os.path.join(cfg.cwd, cfg.data_path, 'valid.csv')
    test_fp = os.path.join(cfg.cwd, cfg.data_path, 'test.csv')
    relation_fp = os.path.join(cfg.cwd, cfg.data_path, 'relation.csv')

    logger.info('load raw files...')
    train_data = load_csv(train_fp)
    valid_data = load_csv(valid_fp)
    test_data = load_csv(test_fp)
    relation_data = load_csv(relation_fp)

    logger.info('convert relation into index...')
    rels, heads, tails = _handle_relation_data(relation_data)
    for d in train_data:
        d['rel2idx'] = rels[d['relation']]
    for d in valid_data:
        d['rel2idx'] = rels[d['relation']]
    for d in test_data:
        d['rel2idx'] = rels[d['relation']]

    logger.info('verify whether use pretrained language models...')
    if cfg.model_name == 'lm':
        logger.info('use pretrained language models serialize sentence...')
        _lm_serialize(train_data, cfg)
        _lm_serialize(valid_data, cfg)
        _lm_serialize(test_data, cfg)
    else:
        logger.info('serialize sentence into tokens...')
        serializer = Serializer(do_chinese_split=cfg.chinese_split, never_split=[*heads, *tails] if cfg.chinese_split else None)
        serial = serializer.serialize
        _serialize_sentence(train_data, serial, cfg)
        _serialize_sentence(valid_data, serial, cfg)
        _serialize_sentence(test_data, serial, cfg)

        logger.info('build vocabulary...')
        vocab = Vocab('word')
        train_tokens = [d['tokens'] for d in train_data]
        valid_tokens = [d['tokens'] for d in valid_data]
        test_tokens = [d['tokens'] for d in test_data]
        sent_tokens = [*train_tokens, *valid_tokens, *test_tokens]
        for sent in sent_tokens:
            vocab.add_words(sent)
        vocab.trim(min_freq=cfg.min_freq)

        logger.info('convert tokens into index...')
        _convert_tokens_into_index(train_data, vocab)
        _convert_tokens_into_index(valid_data, vocab)
        _convert_tokens_into_index(test_data, vocab)

        logger.info('build position sequence...')
        _add_pos_seq(train_data, cfg)
        _add_pos_seq(valid_data, cfg)
        _add_pos_seq(test_data, cfg)

    logger.info('save data for backup...')
    os.makedirs(os.path.join(cfg.cwd, 'data/out'), exist_ok=True)
    train_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    save_pkl(valid_data, valid_save_fp)
    save_pkl(test_data, test_save_fp)

    if cfg.model_name != 'lm':
        vocab_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl')
        vocab_txt = os.path.join(cfg.cwd, cfg.out_path, 'vocab.txt')
        save_pkl(vocab, vocab_save_fp)
        logger.info('save vocab in txt file, for watching...')
        with open(vocab_txt, 'w', encoding='utf-8') as f:
            f.write(os.linesep.join(vocab.word2idx.keys()))

    logger.info('===== end preprocess data =====')


if __name__ == '__main__':
    pass
