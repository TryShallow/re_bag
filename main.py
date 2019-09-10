import preprocess
import os

from dataset import MyDataset
from train import Trainer
from nltk.tokenize import TweetTokenizer


if __name__ == '__main__':
    in_file = 'train.json'
    dev_file = 'dev.json'

    use_elmo = True
    add_query_node = False
    evaluation_mode = False
    max_nodes = 500
    max_query_size = 25
    max_candidates = 80
    max_candidates_len = 10

    logger = preprocess.config_logger('main')
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenize = TweetTokenizer().tokenize
    if not os.path.isfile('{}.preprocessed.pickle'.format(in_file)) \
            or not os.path.isfile('{}.preprocessed.pickle'.format(dev_file)):
        logger.error("train file or dev file is not exists.")
        exit()
    if not evaluation_mode:
        logger.info('Load preprocessed train and dev file.')
        dataset = MyDataset(in_file, use_elmo, max_nodes, max_query_size, max_candidates, max_candidates_len)
        dev_dataset = MyDataset(dev_file, use_elmo, max_nodes, max_query_size, max_candidates, max_candidates_len)
        trainer = Trainer(dataset, dev_dataset)
        trainer.train()
    else:
        logger.info('Load preprocessed evaluation data file.')
        dataset = MyDataset(dev_file, use_elmo, max_nodes, max_query_size, max_candidates, max_candidates_len)
        trainer = Trainer(dataset, dataset)
        trainer.eval()