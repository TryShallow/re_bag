import json
import time
import os
import nltk
import pickle
import logging
import numpy as np
import argparse

from progressbar import ProgressBar, Percentage, Bar, Timer, ETA
from nltk.tokenize import TweetTokenizer
# from allennlp.commands.elmo import ElmoEmbedder

GPU_INDEX = 0

class Preprocessor(object):
    def __init__(self, file_name, options_file, weight_file, glove_file, logger, is_masked=False):
        self.file_name = file_name
        self.options_file = options_file
        self.weight_file = weight_file
        self.glove_file = glove_file
        self.logger = logger

        self.tokenizer = TweetTokenizer()
        self.max_support_length = 512
        self.is_masked = is_masked
        self.use_elmo = True
        self.elmo_split_interval = 8
        self.tag_dict = {'<PAD>': 0, '<UNK>': 1, '<POS>': 2, '<EOS>': 3}
        self.elmo_slice_len = 10000

    def preprocess(self):
        preprocess_graph_file_name = self.file_name
        graph_pickle_file = '{}.preprocessed.pickle'.format(preprocess_graph_file_name)
        preprocess_elmo_file_name = self.file_name + ".elmo"
        elmo_pickle_file = '{}.preprocessed.pickle'.format(preprocess_elmo_file_name)
        # preprocess_glove_file_name = self.file_name + ".glove"
        # preprocess_extra_file_name = self.file_name + ".extra"

        supports = self.do_preprocess4graph(graph_pickle_file)
        with open(graph_pickle_file, 'rb') as f:
            data_graph = [d for d in pickle.load(f)]
            self.logger.info(data_graph[0].keys())

        text_data = []
        for index, graph_d in enumerate(data_graph):
            tmp = {}
            tmp['query'] = graph_d['query']
            tmp['query_full_token'] = graph_d['query_full_token']
            tmp['nodes_mask'] = graph_d['nodes_mask']
            tmp['candidates'] = graph_d['candidates']
            tmp['nodes_candidates_id'] = graph_d['nodes_candidates_id']
            tmp['supports'] = graph_d['supports']
            text_data.append(tmp)
        if self.use_elmo:
            data_slices = len(text_data) // self.elmo_slice_len
            print(data_slices, len(text_data), self.elmo_slice_len)
            for x in range(data_slices):
                self.do_preprocess4elmo(text_data[x * self.elmo_slice_len: (x + 1) * self.elmo_slice_len],
                                        elmo_pickle_file + "." + str(x))
            if len(text_data) % self.elmo_slice_len != 0:
                self.do_preprocess4elmo(text_data[data_slices * self.elmo_slice_len:], elmo_pickle_file + "." + str(data_slices))

    def do_preprocess4graph(self, pickle_file):
        with open(self.file_name, 'r') as f:
            data = json.load(f)
            # data = data[:10]
            supports = self.do_preprocess(data, mode='supports')
            self.logger.info(supports)

        if not os.path.isfile(pickle_file):
            self.logger.info('Preprocessing Json data for Graph...')
            data_preprocessed = self.do_preprocess(data, mode='graph', supports=supports)
            self.logger.info('Preprocessing Graph data finished.')
            with open(pickle_file, 'wb') as f:
                pickle.dump(data_preprocessed, f)
                self.logger.info('Successfully save preprocessed Graph data file %s', pickle_file)
        else:
            self.logger.info('Preprocessed Graph data is already existed, no preprocessing will be executed.')
        return supports

    def do_preprocess4elmo(self, text_data, pickle_file):
        if not os.path.isfile(pickle_file):
            elmo_embedder = ElmoEmbedder(self.options_file, self.weight_file, GPU_INDEX)
            self.logger.info("Preprocessing Json data for Elmo...")
            data = self.do_preprocess(text_data, mode='elmo', ee=elmo_embedder)
            self.logger.info("Preprocessing Elmo data finished.")
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f)
                self.logger.info("Successfully save preprocessed Elmo data file " + pickle_file)
        else:
            self.logger.info('Preprocessed Elmo data is already existed, no preprocessing will be executed.')

    def do_preprocess(self, data_mb, mode, supports=None, ee=None, gloveEmbMap=None, vocab2index=None, bert_model=None):
        data_gen = []
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=len(data_mb)).start()
        print("Total data count", len(data_mb))
        data_count = 0
        for index, data in enumerate(data_mb):
            if mode == 'supports':
                tmp = {}
                tmp['supports'] = [self.tokenizer.tokenize(support) for support in data['supports']]
                for i in range(len(tmp['supports'])):
                    if len(tmp['supports'][i]) > self.max_support_length:
                        tmp['supports'][i] = tmp['supports'][i][: self.max_support_length]
                data_gen.append(tmp)
            elif mode == 'graph':
                preprocess_graph_data = self.preprocess4graph(data, supports[index]['supports'])
                data_gen.append(preprocess_graph_data)
            elif mode == 'elmo':
                preprocess_elmo_data = self.preprocess4elmo(data, ee)
                data_gen.append(preprocess_elmo_data)

            data_count += 1
            pbar.update(data_count)
        pbar.finish()
        return data_gen

    def preprocess4graph(self, data, supports):
        if data.__contains__('annotations'):
            self.logger.info(data['query'])
            self.logger.info(data['candidates'])
            self.logger.info(data['answer'])
            self.logger.info(data['annotations'])
            data.pop('annotations')

        # print data to handle
        self.logger.info(data['query'])
        self.logger.info(data['candidates'])
        self.logger.info(data['answer'])

        first_blank_pos = data['query'].find(' ')
        self.logger.info('first blank pos in query: %d' % first_blank_pos)
        if first_blank_pos > 0:
            first_token_in_query = data['query'][: first_blank_pos]
        else:
            first_token_in_query = data['query']
        query = data['query'].replace('_', ' ')
        self.logger.info("query: " + query)
        data['query'] = self.tokenizer.tokenize(query)
        data['query_full_token'] = query
        self.logger.info(data['query'])
        candidates_orig = list(data['candidates'])
        data['candidates'] = [self.tokenizer.tokenize(candidate) for candidate in data['candidates']]
        self.logger.info(data['candidates'])

        marked_candidate = {}
        if self.is_masked:
            mask = [[self.ind(sindex, windex, cindex, candidate, marked_candidate) for windex, word_support in
                     enumerate(support) for cindex, candidate in enumerate(data['candidates']) if
                     self.check_masked(support, windex, candidate)] for sindex, support in enumerate(supports)]
        else:
            mask = [[self.ind(sindex, windex, cindex, candidate, marked_candidate)
                     for windex, word_support in enumerate(support) for cindex, candidate in
                     enumerate(data['candidates']) if self.check(support, windex, candidate)]
                    for sindex, support in enumerate(supports)]
            self.logger.info(mask)
            self.logger.info(len(mask))
            self.logger.info(len(supports))
            tok_unmarked_candidates = []
            unmarked_candidates_index_map = {}
            for candidate_index in range(len(data['candidates'])):
                if not marked_candidate.__contains__(candidate_index):
                    tok_unmarked_candidates.append(data['candidates'][candidate_index])
                    unmarked_candidates_index_map[len(tok_unmarked_candidates) - 1] = candidate_index
            if len(tok_unmarked_candidates) != 0:
                unmarked_mask = [[self.ind(sindex, windex, unmarked_candidates_index_map[cindex], candidate, marked_candidate)
                                  for windex, word_support in enumerate(support) for cindex, candidate in enumerate(tok_unmarked_candidates)
                                  if self.check(support, windex, candidate, for_unmarked=True)] for sindex, support in enumerate(supports)]
                mask = self.merge_two_masks(mask, unmarked_mask)
        self.logger.info(mask)
        nodes_id_name = []
        count = 0
        self.logger.info([[[x[-1] for x in c][0] for c in s] for s in mask])
        for e in [[[x[-1] for x in c][0] for c in s] for s in mask]:
            u = []
            for f in e:
                u.append((count, f))
                count += 1
            nodes_id_name.append(u)
        data['nodes_candidates_id'] = [[node_triple[-1] for node_triple in node][0] for nodes_in_a_support
                                       in mask for node in nodes_in_a_support]
        self.logger.info(nodes_id_name)
        self.logger.info(data['nodes_candidates_id'])

        edges_in, edges_out = [], []
        for e0 in nodes_id_name:
            for f0, w0 in e0:
                for f1, w1 in e0:
                    if f0 != f1:
                        edges_in.append((f0, f1))
                for e1 in nodes_id_name:
                    for f1, w1 in e1:
                        if e0 != e1 and w0 == w1:
                            edges_out.append((f0, f1))
        self.logger.info(edges_in)
        self.logger.info(edges_out)
        data['edges_in'] = edges_in
        data['edges_out'] = edges_out
        data['nodes_mask'] = mask

        data['relation_index'] = len(first_token_in_query)
        for index, answer in enumerate(candidates_orig):
            if answer == data['answer']:
                data['answer_candidate_id'] = index
                break
        return data

    def preprocess4elmo(self, text_data, ee):
        data_elmo = {}

        mask_ = [[x[:-1] for x in f] for e in text_data['nodes_mask'] for f in e]
        self.logger.info(mask_)
        self.logger.info(len(mask_))
        supports, query, query_full_tokens = text_data['supports'], text_data['query'], text_data['query_full_token']
        first_token_in_query = query[0].split('_')
        self.logger.info(query)
        self.logger.info(query_full_tokens)

        split_interval = self.elmo_split_interval
        if len(supports) <= split_interval:
            candidates, _ = ee.batch_to_embeddings(supports)
            candidates = candidates.data.cpu().numpy()
        else:
            candidates = None
            count = 0
            while count < len(supports):
                current_candidates, _ = ee.batch_to_embeddings(supports[count: min(count + split_interval, len(supports))])
                current_candidates = current_candidates.data.cpu().numpy()
                if candidates is None:
                    candidates = current_candidates
                else:
                    if candidates.shape[2] > current_candidates.shape[2]:
                        current_candidates = np.pad(current_candidates, (
                            (0, 0), (0, 0), (0, candidates.shape[2] - current_candidates.shape[2]), (0, 0)
                        ), 'constant')
                    elif current_candidates.shape[2] > candidates.shape[2]:
                        candidates = np.pad(candidates, (
                            (0, 0), (0, 0), (0, current_candidates.shape[2] - candidates.shape[2]), (0, 0)
                        ), 'constant')
                    candidates = np.concatenate((candidates, current_candidates))
                count += split_interval
            data_elmo['nodes_elmo'] = [(candidates.transpose((0, 2, 1, 3))[np.array(m).T.tolist()]).astype(np.float16)
                                       for m in mask_]

            # (batch_size, 3, num_timesteps, 1024)
            query, _ = ee.batch_to_embeddings([query])
            query = query.data.cpu().numpy()
            data_elmo['query_elmo'] = (query.transpose((0, 2, 1, 3))).astype(np.float16)[0]
            if len(first_token_in_query) == 1:
                data_elmo['query_full_token_elmo'] = data_elmo['query_elmo']
            else:
                query_full_tokens, _ = ee.batch_to_embeddings([first_token_in_query])
                query_full_tokens = query_full_tokens.cpu().numpy()
                data_elmo['query_full_token_elmo'] = np.concatenate(
                    (query_full_tokens.transpose((0, 2, 1, 3)).astype(np.float16)[0], data_elmo['query_elmo'][1:, :, :], 0)
                )
            return data_elmo

        return data_elmo

    def check(self, support, word_index, candidate, for_unmarked=False):
        if for_unmarked:
            return sum([self.is_contain_special_symbol(c_, support[word_index + j].lower())
                        for j, c_ in enumerate(candidate) if word_index + j < len(support)]) == len(candidate)
        else:
            return sum([support[word_index + j].lower() == c_ for j, c_ in enumerate(candidate)
                        if word_index + j < len(support)]) == len(candidate)

    def is_contain_special_symbol(self, candidate_tok, support_tok):
        if candidate_tok.isdigit():
            return support_tok.find(candidate_tok) >= 0
        else:
            return (support_tok == candidate_tok) or (candidate_tok + 's' == support_tok) or\
                   (support_tok.find(candidate_tok)) >= 0 and (support_tok.find('-') > 0 or\
                    support_tok.find('\'s') > 0 or support_tok.find(',') > 0)

    def check_masked(self, support, word_index, candidate):
        return sum([support[word_index + j] == c_ for j, c_ in enumerate(candidate) if
                    word_index + j < len(support)]) == len(candidate)

    def ind(self, support_index, word_index, candidate_index, candidate, marked_candidate):
        marked_candidate[candidate_index] = True
        return [[support_index, word_index + i, candidate_index] for i in range(len(candidate))]

    def merge_two_masks(self, mask, unmarked_mask):
        for i in range(len(mask)):
            if len(unmarked_mask[i]) != 0:
                if len(mask[i]) == 0:
                    mask[i] = unmarked_mask[i]
                else:
                    for unmarked_index in range(len(unmarked_mask[i])):
                        mask[i].append(unmarked_mask[i][unmarked_index])
                    mask[i].sort(key=lambda x: x[0][1])
        return mask


def config_logger(log_prefix):
    logger_prepared = logging.getLogger()
    # logger_prepared.setLevel(logging.INFO)
    logger_prepared.setLevel(logging.ERROR)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(pathname)s[line:%(lineno)d]: %(message)s')
    # write to terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_format)
    logger_prepared.addHandler(stream_handler)
    # write to file
    rq = time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))
    log_path = os.getcwd() + '/logs/' + log_prefix + "/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_filename = log_path + rq + '.log'
    file_handler = logging.FileHandler(log_filename, 'a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger_prepared.addHandler(file_handler)
    return logger_prepared


if __name__ == '__main__':
    # file_name = 'train.json'
    # file_name = 'dev.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)

    args = parser.parse_args()
    file_name = args.file_name
    if file_name == "" or not file_name :
        print('file_name is None')
        exit()
    options_file = 'data/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = 'data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
    glove_file = 'data/glove.840B.300d.zip'
    logger = config_logger("Preprocess")
    preprocessor = Preprocessor(file_name, options_file, weight_file, glove_file, logger)
    preprocessor.preprocess()