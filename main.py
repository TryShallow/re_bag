import preprocess
import os

from dataset_glove_elmo import MyDataset
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
    # if not os.path.isfile('{}.preprocessed.pickle'.format(in_file)) \
    #         or not os.path.isfile('{}.preprocessed.pickle'.format(dev_file)):
    #     logger.error("train file or dev file is not exists.")
    #     exit()
    if not evaluation_mode:
        logger.info('Load preprocessed train and dev file.')
        dataset = MyDataset(in_file, range(10), use_elmo, max_nodes, max_query_size, max_candidates,
                            max_candidates_len)
        dev_dataset = MyDataset(dev_file, range(25, 30), use_elmo, max_nodes, max_query_size, max_candidates,
                                max_candidates_len)
        logger.info("Data has prepared, train: %d, dev: %d." % (len(dataset), len(dev_dataset)))
        trainer = Trainer(dataset, dev_dataset, logger)
        trainer.train()
    else:
        logger.info('Load preprocessed evaluation data file.')
        dataset = MyDataset(dev_file, use_elmo, max_nodes, max_query_size, max_candidates,
                            max_candidates_len)
        trainer = Trainer(dataset, dataset)
        trainer.eval()