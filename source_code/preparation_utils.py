from concurrent.futures import process
from data_utils import PropertyInfoCache, WikidataPropertyRepresentationLoader
from data_utils import WikidataPropertyRepresentationProcessor, WikidataEntityRepresentationProcessor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import time
import pytorch_utils
from evaluation_metrics import acc_and_f1_multiclass as compute_metrics
from data_utils import GAT_Factual_Reasoning_Processor, DataProcessorForLM, Entity_Compatibility_Processor, \
    Wikidata_DataLoading_Processor
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer)
from wikidata_utils import WikidataDumpUtils
import evaluation_metrics
from tqdm import tqdm, trange
import numpy as np
import math
import random
import torch
import glob
import json
import shutil
import os
import pickle
import sys
import yaml
import csv
from pymongo import MongoClient
from data_utils import Wikidata_Database_Processor
from multiprocessing import Manager, Pool, Lock
import multiprocessing
import uuid


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

#control which GPU to use
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def argument_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--triple_feature_creation_type",
                        default="novel", type=str, help="create normal or novel type")

    parser.add_argument("--no_cuda", default=False, type=bool, help="")

    parser.add_argument(
        "--triple_file_path", default="./dataset/ALL_Relations_unique_triples/P6_triple.txt", type=str, help="")

    # parser.add_argument(
    #     "--triple_file_path", default="123", type=str, help="")

    parser.add_argument("--entity_id_file_path", default="1234", type=str,
                        help="")

    # Note: dataset information
    parser.add_argument("--model_type", default="relational_gat", type=str,
                        help="The model type of interactions of subject/object entities.")
    # 1. sup_contrastive_exp
    # 2. unsup_simple_transform
    # 3. relational_gat
    # 4. RGAT_MaxMargin

    parser.add_argument("--data_folder", default="./data/data_examples_and_features", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # "./data/output_contrastive_entity_pair"

    parser.add_argument("--dropout", default=0.0, type=float,
                        help="")

    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--output_tag", type=str, default="test",
                        help="show this tag in the output dir, to show the purpose of this run. "
                             "For instance, batch_32_2021Jan11_13-28-35_lambda-quad means "
                             "it is for show result for batch size 32")

    parser.add_argument("--max_des", type=int, default=50,
                        help="the max length of description split by space.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")

    # feature creation method
    parser.add_argument("--additional_feature", default="nothing", type=str,
                        help="use hypernym feature for entities")
    # additional_feature choices:
    # ["hypernym_embed", "nothing"]

    parser.add_argument("--feature_type", default="bert",
                        type=str, help="The method to create features")
    # feature creation choice:
    # ["glove",
    #  "bert"
    #  ]

    # Note: ================ glove params ======================
    parser.add_argument("--glove_embedding_type", default="glove_6B_300d",
                        help="glove_6B_50d | "
                             "glove_6B_100d | "
                             "glove_6B_200d | "
                             "glove_6B_300d | "
                             "glove_42B |"
                             "glove_840B")
    parser.add_argument("--glove_embed_size", default=300,
                        help="should consistent with above")
    # path_dict = {"glove_6B_50d": os.path.join(self.glove_data_folder, "glove.6B.50d.txt"),
    #              "glove_6B_100d": os.path.join(self.glove_data_folder, "glove.6B.100d.txt"),
    #              "glove_6B_200d": os.path.join(self.glove_data_folder, "glove.6B.200d.txt"),
    #              "glove_6B_300d": os.path.join(self.glove_data_folder, "glove.6B.300d.txt"),
    #              "glove_42B": os.path.join(self.glove_data_folder, "glove.42B.300d.txt"),
    #              "glove_840B": os.path.join(self.glove_data_folder, "glove.840B.300d.txt")}

    parser.add_argument("--glove_embedding_folder",
                        default="../../embeddings/glove_data/", help="glove data folder")

    # Note: parser information
    parser.add_argument("--CORENLP_HOME", type=str, default="../../pretrained_models/stanford-corenlp-4.2.0",
                        help="stanford corenlp parser. If use the same one for two instances, there will be conflict.")
    parser.add_argument("--corenlp_ports", type=int, default=9001)

    # Note: ====================== bert params ==================
    parser.add_argument("--pretrained_transformer_model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--pretrained_transformer_model_name_or_path", default="bert-base-cased", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--pretrained_bert_model_folder_for_feature_creation",
                        default=None,
                        type=str,
                        help="the pretrained model to create bert embedding for following task.")

    # ======= GAT model related parameter ===========
    # --------------- model configuration ------------
    parser.add_argument("--input_size", default=768,
                        type=int, help="initial input size")
    parser.add_argument("--hidden_size", default=300,
                        type=int, help="the hidden size for GAT layer")

    parser.add_argument("--heads", default=6, type=int,
                        help="number of heads in the GAT layer")
    parser.add_argument("--att_dropout", default=0, type=float, help="")
    parser.add_argument("--stack_layer_num", default=8,
                        type=int, help="the number of layers to stack")

    parser.add_argument("--num_classes", default=-1,
                        type=int, help="the number of class")
    parser.add_argument("--embed_dropout", default=0.3,
                        type=float, help="dropout for input word embedding")

    # Other parameters
    parser.add_argument("--max_seq_length", default=100, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded. Acutally as long as the max"
                             "lenght is lower than BERT max length, it is OK. The max lenght is 86 for bert tokenizer. "
                             "so it is OK")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    # ------- training details ----------
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--learning_rate", default=0.0005, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight deay if we apply some.")
    # -------------------------------------
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port', type=str,
                        default='', help="For distant debugging.")

    # ---------------- evaluation -------------
    # run = RunCompGAT(best_dev_eval_file="best_dev_info.json", considered_metrics="f1_weighted")
    parser.add_argument("--best_dev_eval_file", type=str,
                        default="best_dev_info.json")
    parser.add_argument("--considered_metrics", type=str, default="auc_score")
    parser.add_argument("--save_model_file_name",
                        type=str, default="best_model.pt")
    parser.add_argument("--output_mode", type=str, default="classification")

    # ---------------- do test -----------
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', default=True,
                        help="Rul evaluation during training at each logging step.")

    parser.add_argument("--do_test", action="store_true", default=True,
                        help="load the trained model and check testing result")
    parser.add_argument("--trained_result_folder", type=str,
                        default="./result/2020-04-21__09-49__543108")

    args = parser.parse_args()

    return args


class WikidataEntityExample:
    """
    June 14, 2022

    Current, we limit the wikidata entity connection to one hop

    entity:
       property_1 : value (string only)
       property_2 : value (string only)
       ...
    """

    def __init__(self, wikidata_id, label, description, property_to_value_list):
        self.wikidata_id = wikidata_id
        self.label = label
        self.description = description
        self.property_to_value_list = property_to_value_list  # The property id is sorted, value is a text_list

    # enddef

    def to_dict(self):
        return self.__dict__

    def to_str(self, property_cache):
        string = f"{self.wikidata_id} || {self.label} || {self.description}\n"

        for property_id, value_text_list in self.property_to_value_list.items():

            property_label, property_des = property_cache.get_property_label_and_des(
                property_id)
            string += f"{str(value_text_list)} || {property_id} || {property_label} || {property_des}\n"
        # endfor

        return string


class WikidataEntity:
    """
    Current, we limit the wikidata entity connection to one hop

    entity:
       property_1 : value (string only)
       property_2 : value (string only)
       ...
    """

    def __init__(self, wikidata_id, label, description, property_to_value_list_dict):
        self.wikidata_id = wikidata_id
        self.label = label
        self.description = description
        self.property_to_value_list_dict = property_to_value_list_dict

    # enddef

    def to_str(self, property_cache):
        string = f"{self.wikidata_id} || {self.label} || {self.description}\n"
        for property_id, value_text_list in self.property_to_value_list_dict.items():
            property_label, property_des = property_cache.get_property_label_and_des(
                property_id)
            string += f"{value_text_list} || {property_id} || {property_label} || {property_des}\n"
        # endfor
        return string


class WikidataEntityPairRelationExample_E1E2:
    """
    e1 always appears before e2
    No head and tail concept
    head and tail order are contained in rc_label_text (original or reversed order)
    """
    
    def __init__(self, guid, text_info, e1_wikidata_entity_obj, e2_wikidata_entity_obj, rc_label_text, nd_label_text):
        """
        e1_wikidata_entity_obj : type -> WikidataEntity
        e2_wikidata_entity_obj : type -> WikidataEntity
        """
        self.guid = guid
        self.text_info = text_info  # tokens information, head tokens, tail tokens
        self.e1_wikidata_entity_obj = e1_wikidata_entity_obj
        self.e2_wikidata_entity_obj = e2_wikidata_entity_obj
        self.rc_label_text = rc_label_text
        self.nd_label_text = nd_label_text

    def to_str(self, property_cache):
        string = ""
        string += f"{self.guid}\n"
        string += "=" * 30 + "\n\n"
        string += f"{self.e1_wikidata_entity_obj.to_str(property_cache)}\n"
        string += "-" * 30 + "\n"
        string += f"{self.e2_wikidata_entity_obj.to_str(property_cache)}\n"
        string += "-" * 30 + "\n"
        string += f"{self.rc_label_text}\n"
        string += "=" * 30 + "\n\n"
        string += f"{self.nd_label_text}\n"
        string += "=" * 30 + "\n\n"
        return string


class WikidataEntityPairRelationExample:
    def __init__(self, guid, text_info, head_wikidata_entity_obj, tail_wikidata_entity_obj, rc_label_text, nd_label_text):
        """
        subj_wikidata_entity_obj : type -> WikidataEntity
        obj_wikidata_entity_obj : type -> WikidataEntity
        """
        self.guid = guid
        self.text_info = text_info  # tokens information, head tokens, tail tokens
        self.head_wikidata_entity_obj = head_wikidata_entity_obj
        self.tail_wikidata_entity_obj = tail_wikidata_entity_obj
        self.rc_label_text = rc_label_text
        self.nd_label_text = nd_label_text

    def to_str(self, property_cache):
        string = ""
        string += f"{self.guid}\n"
        string += "=" * 30 + "\n\n"
        string += f"{self.head_wikidata_entity_obj.to_str(property_cache)}\n"
        string += "-" * 30 + "\n"
        string += f"{self.tail_wikidata_entity_obj.to_str(property_cache)}\n"
        string += "-" * 30 + "\n"
        string += f"{self.rc_label_text}\n"
        string += "=" * 30 + "\n\n"
        string += f"{self.nd_label_text}\n"
        string += "=" * 30 + "\n\n"
        return string


class WikidataEntityPairFeature_E1E2:
    def __init__(self,
                 guid,
                 e1_wikidata_id,
                 e2_wikidata_id,
                 e1_property_str_list,
                 e2_property_str_list,
                 e1_value_embed_list,
                 e2_value_embed_list,
                 rc_label_text,
                 nd_label_text):

        self.guid = guid
        self.e1_wikidata_id = e1_wikidata_id
        self.e2_wikidata_id = e2_wikidata_id
        self.e1_property_str_list = e1_property_str_list
        self.e2_property_str_list = e2_property_str_list
        self.e1_value_embed_list = e1_value_embed_list
        self.e2_value_embed_list = e2_value_embed_list
        self.rc_label_text = rc_label_text
        self.nd_label_text = nd_label_text
    # enddef


class WikidataEntityPairFeature:
    def __init__(self,
                 guid,
                 head_wikidata_id,
                 tail_wikidata_id,
                 head_property_str_list,
                 tail_property_str_list,
                 head_value_embed_list,
                 tail_value_embed_list,
                 rc_label_text,
                 nd_label_text):

        self.guid = guid
        self.head_wikidata_id = head_wikidata_id
        self.tail_wikidata_id = tail_wikidata_id
        self.head_property_str_list = head_property_str_list
        self.tail_property_str_list = tail_property_str_list
        self.head_value_embed_list = head_value_embed_list
        self.tail_value_embed_list = tail_value_embed_list
        self.rc_label_text = rc_label_text
        self.nd_label_text = nd_label_text
    # enddef


class PropertyProcessor:
    def __init__(self, args, all_filtered_property_id_to_label_description_dict_file=None, output_dir=None):
        self.all_filtered_property_id_to_label_description_dict_file = all_filtered_property_id_to_label_description_dict_file
        self.output_dir = output_dir

        self.args = args
        self.bert_model = None
        self.bert_tokenizer = None
        pass

    def create_all_candiate_property_str_to_info_dict(self):
        """
        It is called candidate property id, since not all property are available in database,
        some property jsons cannot be retrieved online because they are deprecated.

        Note: P-2 (description) and P-1 (label) are also added to the property id set

        original size: 2427
        There are 2429 candidate property ids, including P-2 and P-1
        """
        # ######## load all_filtered_property_id_to_label_description_dict_file #########
        with open(self.all_filtered_property_id_to_label_description_dict_file, mode="r") as fin:
            all_filtered_property_id_to_label_description_dict = json.load(fin)
        # endwith
        print(
            f"original size: {len(all_filtered_property_id_to_label_description_dict)}")

        ### add label and description ###
        all_filtered_property_id_to_label_description_dict["P-2"] = {
            "label": "description", "description": ""}
        all_filtered_property_id_to_label_description_dict["P-1"] = {
            "label": "label", "description": ""}
        print(
            f"There are {len(all_filtered_property_id_to_label_description_dict)} candidate property ids, including P-2 and P-1")

        # ############## output #############
        property_output_file = os.path.join(
            self.output_dir, "property_str_to_info_dict.json")
        with open(property_output_file, mode="w") as fout:
            json.dump(all_filtered_property_id_to_label_description_dict, fout)
        # endwith
        pass

    def load_all_candidate_property_str_to_info_dict(self):
        property_output_file = os.path.join(
            self.output_dir, "property_str_to_info_dict.json")

        with open(property_output_file, mode="r") as fin:
            all_candidate_property_str_to_info_dict = json.load(fin)
        # endwith

        return all_candidate_property_str_to_info_dict

    def _bert_convert_text_to_feature_vector_simple(self, text):
        """
        text: "I love chicago"
        inputs: <class 'transformers.tokenization_utils_base.BatchEncoding'>
        --------
        input_ids: tensor([[ 101, 1045, 2293, 3190,  102]])
        token_type_ids: tensor([[0, 0, 0, 0, 0]])
        attention_mask tensor([[1, 1, 1, 1, 1]])

        It returns the same as _bert_convert_text_to_feature_vector_cls()
        """

        inputs = self.bert_tokenizer(text, return_tensors="pt")
        inputs.to(self.args.device)

        with torch.no_grad():
            outputs, _ = self.bert_model(**inputs)
        # endwith

        outputs = outputs.squeeze(dim=0)
        text_vector = outputs[0].to('cpu').numpy()
        return text_vector

    def _bert_convert_text_to_feature_vector_simple(self, text):
        """
        text: "I love chicago"
        inputs: <class 'transformers.tokenization_utils_base.BatchEncoding'>
        --------
        input_ids: tensor([[ 101, 1045, 2293, 3190,  102]])
        token_type_ids: tensor([[0, 0, 0, 0, 0]])
        attention_mask tensor([[1, 1, 1, 1, 1]])

        It returns the same as _bert_convert_text_to_feature_vector_cls()
        """

        inputs = self.bert_tokenizer(text, return_tensors="pt")
        inputs.to(self.args.device)

        with torch.no_grad():
            outputs, _ = self.bert_model(**inputs)
        # endwith

        outputs = outputs.squeeze(dim=0)
        text_vector = outputs[0].to('cpu').numpy()
        return text_vector

    def _bert_convert_text_to_feature_vector_cls(self, text):
        token_list = text.strip().split()

        word_pieces_list = []
        word_boundaries_list = []

        for w in token_list:
            word_pieces = self.bert_tokenizer.tokenize(w)
            word_boundaries_list.append([len(word_pieces_list), len(word_pieces_list) + len(
                word_pieces)])  # Note: original word token corresponding to word piece boundary
            word_pieces_list += word_pieces
        # endfor
        assert len(word_boundaries_list) == len(token_list)

        total_input_tokens = ['[CLS]'] + word_pieces_list + ['[SEP]']
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(
            total_input_tokens)
        segment_ids = [0] * len(total_input_tokens)

        input_ids = torch.tensor(
            [input_ids], dtype=torch.long).to(self.args.device)
        segment_ids = torch.tensor(
            [segment_ids], dtype=torch.long).to(self.args.device)

        with torch.no_grad():
            sequence_output, _ = self.bert_model(input_ids=input_ids,
                                                 token_type_ids=segment_ids)
        # endwith

        sequence_output = sequence_output.squeeze(dim=0)
        # the first one corresponds to [CLS]
        text_vector = sequence_output[0].to('cpu').numpy()
        return text_vector

    def get_bert_embed(self, text_list):
        """
        Input the original text into bert model, without mask two objects.
        :return:
        """
        if self.bert_model is None and self.bert_tokenizer is None:
            from transformers import BertModel, BertTokenizer
            print("loading bert model: {} to create feature for sent tokens".format(
                self.args.pretrained_transformer_model_name_or_path))
            self.bert_model = BertModel.from_pretrained(
                self.args.pretrained_transformer_model_name_or_path)
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                self.args.pretrained_transformer_model_name_or_path)
            # load the pretrained bert model. The loaded tokenizer will follow the original pretrained config
            # For "do_lower_case" will decide whether or not make the sent_text lower case
            # It should consistent with the programs argument config
            # bert_pretrained_model, bert_pretrained_tokenizer = self._load_pretrained_bert_model(args, model_class,
            # tokenizer_class)
            self.bert_model.eval()
            self.bert_model.to(self.args.device)
            print(
                f"{self.args.pretrained_transformer_model_name_or_path} is successfully loaded.")
        # endif

        text_vector_list = []
        # for text in tqdm(text_list, desc="create bert embedding for text"):
        for text in tqdm(text_list, desc="all text"):
            text_vector = self._bert_convert_text_to_feature_vector_simple(
                text)
            # text_vector = self._bert_convert_text_to_feature_vector_cls(text) # this one is more complicated, discard
            text_vector_list.append(text_vector)
        # endfor

        return text_vector_list

    def handler_create_all_property_features(self):
        """
        create property embedding by encoding the label or/and description of the properties by BERT
        """
        # (1) ################ load all property ids ##################
        all_candidate_property_str_to_info_dict = self.load_all_candidate_property_str_to_info_dict()
        candidate_property_id_list = list(
            all_candidate_property_str_to_info_dict.keys())

        # ######### get sorted property id list #############
        candidate_property_id_list = list(
            all_candidate_property_str_to_info_dict.keys())
        sorted_candidate_property_id_list = sorted(
            candidate_property_id_list, key=lambda x: int(x.replace("P", "")))

        # index_to_str
        property_index_to_str_list = sorted_candidate_property_id_list

        # str_to_index
        property_str_to_index_dict = {}
        for index, property_str in enumerate(property_index_to_str_list):
            property_str_to_index_dict[property_str] = index
        # endfor

        # ####### output index_to_str, str_to_index #######
        property_index_to_str_file_path = os.path.join(
            self.output_dir, "property_index_to_str_list.json")
        with open(property_index_to_str_file_path, mode="w") as fout:
            json.dump(property_index_to_str_list, fout)
        # endwith

        property_str_to_index_dict_file_path = os.path.join(
            self.output_dir, "property_str_to_index_dict.json")
        with open(property_str_to_index_dict_file_path, mode="w") as fout:
            json.dump(property_str_to_index_dict, fout)
        # endwith

        # ######################## create property embedding #################
        for encode_content_option in ["label", "label_and_description"]:

            text_list = []
            for property_str in property_index_to_str_list:
                if encode_content_option == "label":
                    tmp_label = all_candidate_property_str_to_info_dict[property_str]["label"]
                    assert tmp_label is not None
                    assert len(tmp_label) > 0
                    text_list.append(tmp_label)
                # endif

                if encode_content_option == "label_and_description":
                    tmp_label = all_candidate_property_str_to_info_dict[property_str]["label"]
                    assert tmp_label is not None
                    assert len(tmp_label) > 0
                    tmp_text = tmp_label

                    # add description
                    tmp_des = all_candidate_property_str_to_info_dict[property_str]["description"]
                    if len(tmp_des) > 0:
                        tmp_text += ", " + tmp_des
                    # endif

                    text_list.append(tmp_text)
                # endif
            # endfor
            print(
                f"creating text embedding for option -- {encode_content_option}")
            text_embedding_list = self.get_bert_embed(text_list)

            # ================== output text embedding =============
            property_embedding_output_file = os.path.join(self.output_dir,
                                                          f"property_embed_{encode_content_option}")
            text_embedding_arr = np.array(text_embedding_list)
            np.save(property_embedding_output_file, text_embedding_arr)
        # endfor

    def load_property_info(self):
        property_output_folder = self.output_dir

        property_index_to_str_file_path = os.path.join(
            property_output_folder, "property_index_to_str_list.json")
        property_str_to_index_dict_file_path = os.path.join(
            property_output_folder, "property_str_to_index_dict.json")

        # (1)
        with open(property_index_to_str_file_path, mode="r") as fin:
            property_index_to_str_list = json.load(fin)
        # endwith

        # (2)
        with open(property_str_to_index_dict_file_path, mode="r") as fin:
            property_str_to_index_dict = json.load(fin)
        # endwith

        # (3)
        property_str_to_info_dict_path = os.path.join(
            property_output_folder, "property_str_to_info_dict.json")
        with open(property_str_to_info_dict_path, mode="r") as fin:
            property_str_to_info_dict = json.load(fin)
        # endwith

        property_embedding_label_file = os.path.join(
            property_output_folder, f"property_embed_label.npy")
        property_embedding_label_arr = np.load(property_embedding_label_file)

        property_embedding_label_and_des_file = os.path.join(property_output_folder,
                                                             f"property_embed_label_and_description.npy")
        property_embedding_label_and_description_arr = np.load(
            property_embedding_label_and_des_file)

        # property_str_to_index_dict -> 2429
        # property_embedding_label_and_description_arr -> (2429, 768)
        # property_embedding_label_arr -> (2429, 768)

        return property_index_to_str_list, property_str_to_index_dict, property_str_to_info_dict, property_embedding_label_arr, property_embedding_label_and_description_arr


def main_create_property_embedding():
    args = argument_parser()
    #
    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # #### The important directory to load property information ####
    output_folder = "./dataset/FINAL_property_id_to_info_data/output"
    # #############

    os.makedirs(output_folder, exist_ok=True)
    final_filtered_wikidata_property_id_to_label_description_dict_file = "./dataset/FINAL_property_id_to_info_data/input/final_filtered_property_to_label_description_dict.json"

    processor = PropertyProcessor(
        args=args,
        all_filtered_property_id_to_label_description_dict_file=final_filtered_wikidata_property_id_to_label_description_dict_file,
        output_dir=output_folder)

    # ######### create features for all properties ###########
    # processor.create_all_candiate_property_str_to_info_dict()

    processor.handler_create_all_property_features()

    property_index_to_str_list, \
        property_str_to_index_dict, \
        property_str_to_info_dict, \
        property_embedding_label_arr, \
        property_embedding_label_and_description_arr = processor.load_property_info()

    pass


class WikidataProcessor:
    """
    (I) process wikidata properties

    (II) process wikidata entities
    """
    def __init__(self,
                 args,
                 train_data_folder,
                 valid_normal_folder,
                 valid_novel_folder,
                 test_normal_folder,
                 test_novel_folder,
                 high_quality_human_id_file_path,
                 high_quality_human_id_set_current_in_database, 
                 high_quality_property_str_to_num_file_path,
                 train_rc_relation_list_file,
                 evaluation_rc_relation_list_file
                 ):
        """
        When loading Bert model:
        "Cannot re-initialize CUDA in forked subprocess. To use CUDA with
        "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing,
        you must use the 'spawn' start method.
        """
        
        self.args = args
        self.train_data_folder = train_data_folder
        self.valid_normal_folder = valid_normal_folder
        self.valid_novel_folder = valid_novel_folder
        self.test_normal_folder = test_normal_folder
        self.test_novel_folder = test_novel_folder
        self.high_quality_human_id_file_path = high_quality_human_id_file_path
        self.high_quality_human_id_set_current_in_database = high_quality_human_id_set_current_in_database
        self.high_quality_property_str_to_num_file_path = high_quality_property_str_to_num_file_path

        # ########## RC relation list file ############
        self.train_rc_relation_list_file = train_rc_relation_list_file
        self.evaluation_rc_relation_list_file = evaluation_rc_relation_list_file

        # these rc_str_list is SORTED
        self.train_rc_str_list = self.get_rc_relation_str_list_from_file_path(
            train_rc_relation_list_file)
        self.evaluation_rc_str_list = self.get_rc_relation_str_list_from_file_path(
            evaluation_rc_relation_list_file)

        #
        train_rc_str_to_num_dict = {}
        train_num_to_rc_str_dict = {}

        for index, rc_str in enumerate(self.train_rc_str_list):
            train_rc_str_to_num_dict[rc_str] = index
            train_num_to_rc_str_dict[index] = rc_str
        # endfor
        
        # ###### output trained model label mapping #########
        TRAINED_MODEL_relation_str_to_index_dict_file_path = os.path.join(args.output_dir, "TRAINED_MODEL_relation_str_to_index_dict.json")
        TRAINED_MODEL_index_to_relation_str_dict_file_path = os.path.join(args.output_dir, "TRAINED_MODEL_index_to_relation_str_dict.json")
        
        with open(TRAINED_MODEL_relation_str_to_index_dict_file_path, mode="w") as fout:
            json.dump(train_rc_str_to_num_dict, fout)
        
        with open(TRAINED_MODEL_index_to_relation_str_dict_file_path, mode="w") as fout:
            json.dump(train_num_to_rc_str_dict, fout)
        #endwith

        assert len(train_rc_str_to_num_dict) > 0
        assert len(train_num_to_rc_str_dict) > 0
        assert len(train_rc_str_to_num_dict) == len(train_num_to_rc_str_dict)

        self.train_rc_str_to_num_dict = train_rc_str_to_num_dict
        self.train_num_to_rc_str_dict = train_num_to_rc_str_dict

        # ########## nd_label dict ##########
        nd_label_list = self.get_nd_labels()

        nd_label_to_num_dict = {}
        num_to_nd_label_dict = {}

        for index, nd_label in enumerate(nd_label_list):
            nd_label_to_num_dict[nd_label] = index
            num_to_nd_label_dict[index] = nd_label
        # endfor

        self.nd_label_to_num_dict = nd_label_to_num_dict
        self.num_to_nd_label_dict = num_to_nd_label_dict

        self.bert_model = None
        self.bert_tokenizer = None

        # ####################### load property ####################
        property_info_folder = "./dataset/FINAL_property_id_to_info_data/output"
        property_processor = PropertyProcessor(
            args, output_dir=property_info_folder)
        property_index_to_str_list, property_str_to_index_dict, property_str_to_info_dict, property_embedding_label_arr, property_embedding_label_and_description_arr = \
            property_processor.load_property_info()

        self.property_str_to_index_dict = property_str_to_index_dict

        # #########################################################

        # if is_create_feature:
        #     print("loading bert model: {} to create feature for sent tokens".format(
        #         self.args.pretrained_transformer_model_name_or_path))
        #     self.bert_model = BertModel.from_pretrained(
        #         self.args.pretrained_transformer_model_name_or_path)
        #     self.bert_tokenizer = BertTokenizer.from_pretrained(
        #         self.args.pretrained_transformer_model_name_or_path)
        #     # load the pretrained bert model. The loaded tokenizer will follow the original pretrained config
        #     # For "do_lower_case" will decide whether or not make the sent_text lower case
        #     # It should consistent with the programs argument config
        #     # bert_pretrained_model, bert_pretrained_tokenizer = self._load_pretrained_bert_model(args, model_class,
        #     # tokenizer_class)
        #     self.bert_model.eval()
        #     self.bert_model.to(self.args.device)
        #     print(
        #         f"{self.args.pretrained_transformer_model_name_or_path} is successfully loaded.")
        # pass

    # ######################## BEGIN: utilities #########################
    def load_mongodb_config(self):
        stream = open("../mongodb_config.yaml", 'r')
        mongo_config_dict = yaml.load(stream, Loader=yaml.SafeLoader)
        return mongo_config_dict

    def connect_to_wikidata_dump_database(self):
        mongo_config_dict = self.load_mongodb_config()
        mongodb_uri = mongo_config_dict['MONGODB_URI']
        # load mongodb database
        # initialize mongodb
        # Creating a pymongo client
        client = MongoClient(mongodb_uri)

        # Getting the database instance
        database_name = "wikidata_dump"
        db = client[database_name]

        return db

    def connect_to_example_and_feature_database(self):
        mongo_config_dict = self.load_mongodb_config()
        mongodb_uri = mongo_config_dict['MONGODB_URI']
        # load mongodb database
        # initialize mongodb
        # Creating a pymongo client
        client = MongoClient(mongodb_uri)

        # Getting the database instance
        db = client['all_wikidata_features_new']

        return db

    def connect_to_wikidata_triple_feature_database(self):
        mongo_config_dict = self.load_mongodb_config()
        mongodb_uri = mongo_config_dict['MONGODB_URI']
        # load mongodb database
        # initialize mongodb
        # Creating a pymongo client
        client = MongoClient(mongodb_uri)

        # Getting the database instance
        db = client['wikidata_triple_features_with_str_property_new']

        return db

    def get_nd_labels(self):
        """
        novel: 0, normal: 1
        normal should be 1, since we want to make compatibility scoring model to score normal instance higher score
        novel instance lower score
        :return:
        """
        return ["novel", "normal"]

    def get_rc_relation_str_list_from_file_path(self, file_path):
        """
        Get the 24 relation str list
        relation_str like P61
        """
        relation_str_list = []
        # json obj file
        with open(file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                json_obj = json.loads(line)
                relation_str = json_obj["relation_id"]
                relation_str_list.append(relation_str)
            # endfor
        # endwith

        sorted_relation_str_list = sorted(
            relation_str_list, key=lambda x: int(x.replace("P", "")))
        return sorted_relation_str_list

    def load_cur_database_high_quality_human_id_set(self):
        with open(self.high_quality_human_id_set_current_in_database, mode="r") as fin:
            cur_database_high_quality_human_id_list = json.load(fin)
        # endwith
        cur_database_high_quality_human_id_list = list(
            set(cur_database_high_quality_human_id_list))
        print(
            f"There are totally {len(cur_database_high_quality_human_id_list)} human id set")

        return cur_database_high_quality_human_id_list

    def load_each_relation_human_position_info_dict(self):
        """
        There are totally 59 candidate relations (related to human) that can be used to swap to create novel dataset

        dict -> relation_id: human_position(head/tail)
        """
        candidate_relation_human_info_dict = {}
        annotated_good_candidate_relation_id_file = "./dataset/FINAL_property_id_to_info_data/input/Dec_19_2021_human_related_relation_info_with_good_candidate.csv"

        with open(annotated_good_candidate_relation_id_file, mode="r") as fin:
            csv_reader = csv.DictReader(fin)
            for row in csv_reader:
                if row["good_candidate"] == "+" or row["good_candidate"] == "only_ds":
                    # human position
                    subj_human_id_num = int(row["subj_human_id_num"])
                    obj_human_id_num = int(row["obj_human_id_num"])

                    if subj_human_id_num > obj_human_id_num:
                        human_position = "head"
                    if subj_human_id_num < obj_human_id_num:
                        human_position = "tail"

                    candidate_relation_human_info_dict[row["relation_id"]
                                                       ] = human_position
                # endif
            # endfor
        # endwith
        print(
            f"There are totally {len(candidate_relation_human_info_dict)} good candidate relation ids for create novel data."
        )
        return candidate_relation_human_info_dict

    def output_each_relation_human_position_info(self):

        rc_relation_str_list = self.get_rc_relation_str_list()
        total_relation_human_info_dict = self.load_each_relation_human_position_info_dict()

        # ######## output ########
        output_file = "./dataset/relation_information/FILTERD_ALL_relation_human_pos.txt"
        with open(output_file, mode="w") as fout:
            for relation_id in rc_relation_str_list:
                pos = total_relation_human_info_dict[relation_id]
                fout.write(f"{relation_id:<10}\t{pos:<10}\n")
            # endfor
        # endwith
        pass

    def randomly_sample_one_from_set(self, a_set):
        a_list = list(a_set)
        random_num = random.randint(0, len(a_list) - 1)
        item = a_list[random_num]
        return item

    def randomly_sample_n_item_from_set(self, a_set, num):
        random_set = set()
        a_list = list(a_set)
        while len(random_set) < num:
            random_num = random.randint(0, len(a_list) - 1)
            item = a_list[random_num]
            random_set.add(item)
        # endwhile
        return random_set

    # ######################## END: uitlities ##################

    def single_file_get_all_wikidata_entities_from_FEWREL_format(self, file_path_list, shared_entity_id_list):
        """
        fewrel format
        ==============
        every line is fewrel json obj

        Args:
            file_path_list (_type_): _description_
            shared_entity_id_list (_type_): _description_
        """
        for file_path in file_path_list:
            with open(file_path, mode="r") as fin:
                for line in fin:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    # endif
                    json_obj = json.loads(line)
                    h_id = json_obj["h"][1]
                    t_id = json_obj["t"][1]
                    shared_entity_id_list.append(h_id)
                    shared_entity_id_list.append(t_id)
                # endfor
            # endwith
        # endfor
        pass

    def single_file_get_all_wikidata_entities_from_ThreeLine_format(self, file_path_list, shared_entity_id_list):
        """
        Three Line format
        ==================
        fewrel json obj
        head entity id
        tail entity id
        \n
        """
        for file_path in file_path_list:
            # >>>>>>> process one file >>>>>>>
            block_line = []

            with open(file_path, mode="r") as fin:
                for line in fin:
                    line = line.strip()

                    if len(line) == 0:

                        json_obj = json.loads(block_line[0])
                        h_id = json_obj["h"][1]
                        t_id = json_obj["t"][1]
                        shared_entity_id_list.append(h_id)
                        shared_entity_id_list.append(t_id)

                        # reset block_line
                        block_line = []
                    else:
                        block_line.append(line)
                    # endif

                # endfor
            # endwith
        # endfor
        pass

    def parallel_get_all_entities_in_ThreeLine_format_dataset_file(self, folder_list, output_file_path):
        """
        Three Line format
        ==================
        fewrel json obj
        head entity id
        tail entity id
        \n

        Process validation data

        >>> Stats
        There are totally 48 file path to check.
        shared_entity_id_list size: 4800
        There are totally 4039 unique entity id in the dataset.
        """

        all_file_path_list = []

        for folder in folder_list:
            for root, subdir, file_list in os.walk(folder):
                for file_name in file_list:
                    file_path = os.path.join(root, file_name)
                    all_file_path_list.append(file_path)
                # endfor
            # endfor
        # endfor
        print(
            f"There are totally {len(all_file_path_list)} file path to check.")

        manager = Manager()
        shared_entity_id_list = manager.list()

        num_of_worker = os.cpu_count() - 2
        pool = Pool(processes=num_of_worker)
        block_size = math.ceil(len(all_file_path_list) / num_of_worker)

        job_list = []
        for w in range(num_of_worker):
            block_list = all_file_path_list[w *
                                            block_size: (w + 1) * block_size]
            single_worker = pool.apply_async(func=self.single_file_get_all_wikidata_entities_from_ThreeLine_format,
                                             args=(block_list,
                                                   shared_entity_id_list))

            job_list.append(single_worker)
        # endfor

        for job in job_list:
            job.get()
        # endfor

        print(f"shared_entity_id_list size: {len(shared_entity_id_list)}")
        all_entity_id_set = set(shared_entity_id_list)
        all_entity_id_list = list(all_entity_id_set)
        print(
            f"There are totally {len(all_entity_id_list)} unique entity id in the dataset.")

        # ####### output all entity id ##########
        with open(output_file_path, mode="w") as fout:
            json.dump(all_entity_id_list, fout)
        # endwith
        pass

    def parallel_get_all_entities_in_FEWREL_format_dataset_file(self, folder_list, output_file_path):
        """
        Process training data

        # Stats
        >>> train_100
        There are totally 24 file path to check.
        shared_entity_id_list size: 4800 (100 * 2 * 24 = 4800)
        There are totally 3789 unique entity id in the dataset.

        >>> train_100, 1000, 2000, 10_000
        There are totally 96 file path to check.
        shared_entity_id_list size: 628800
        There are totally 126_877 unique entity id in the datasets
        """

        all_file_path_list = []

        for folder in folder_list:
            for root, subdir, file_list in os.walk(folder):
                for file_name in file_list:
                    file_path = os.path.join(root, file_name)
                    all_file_path_list.append(file_path)
                # endfor
            # endfor
        # endfor
        print(
            f"There are totally {len(all_file_path_list)} file path to check.")

        manager = Manager()
        shared_entity_id_list = manager.list()

        num_of_worker = os.cpu_count() - 2
        pool = Pool(processes=num_of_worker)
        block_size = math.ceil(len(all_file_path_list) / num_of_worker)

        job_list = []
        for w in range(num_of_worker):
            block_list = all_file_path_list[w *
                                            block_size: (w + 1) * block_size]
            single_worker = pool.apply_async(func=self.single_file_get_all_wikidata_entities_from_FEWREL_format,
                                             args=(block_list,
                                                   shared_entity_id_list))

            job_list.append(single_worker)
        # endfor

        for job in job_list:
            job.get()
        # endfor

        print(f"shared_entity_id_list size: {len(shared_entity_id_list)}")
        all_entity_id_set = set(shared_entity_id_list)
        all_entity_id_list = list(all_entity_id_set)
        print(
            f"There are totally {len(all_entity_id_list)} unique entity id in the dataset.")

        # ####### output all entity id ##########
        with open(output_file_path, mode="w") as fout:
            json.dump(all_entity_id_list, fout)
        # endwith
        pass

    # ######## validate all entity id in database #########

    def single_worker_check_entity_in_database(self, entity_id_list):
        db = self.connect_to_wikidata_dump_database()

        in_database_set = set()
        not_in_database_set = set()

        for entity_id in tqdm(entity_id_list, desc="worker"):
            result_list = []
            for item in db.entity_obj.find({"id": entity_id}):
                result_list.append(item)
            # endfor

            if len(result_list) == 0:
                not_in_database_set.add(entity_id)
            else:
                in_database_set.add(entity_id)
            # endif
        # endfor

        return in_database_set, not_in_database_set

    def parallel_check_entity_id_LIST_in_wikidata_dump_database(self, all_entity_id_list):
        # ########## check if in the database ##########
        num_of_worker = os.cpu_count() - 2
        block_size = math.ceil(len(all_entity_id_list) / num_of_worker)
        pool = Pool(processes=num_of_worker)

        job_list = []
        for w in range(num_of_worker):
            block_list = all_entity_id_list[w *
                                            block_size: (w + 1) * block_size]
            single_worker = pool.apply_async(func=self.single_worker_check_entity_in_database,
                                             args=(block_list,))
            job_list.append(single_worker)
        # endfor

        all_in_database_set = set()
        all_not_in_database_set = set()
        for job in tqdm(job_list, desc="all jobs"):
            in_database_set, not_in_database_set = job.get()
            all_in_database_set.update(in_database_set)
            all_not_in_database_set.update(not_in_database_set)
        # endfor

        print(f"all in database set: {len(all_in_database_set)}")
        print(f"all NOT in database set: {len(all_not_in_database_set)}")

        pass

    def parallel_check_entity_id_in_wikidata_dump_database(self, entity_id_file_path_list):
        """
        split entity id into
        (1) in database
        (2) not in database

        """
        all_entity_id_list = []
        for file_path in entity_id_file_path_list:
            with open(file_path, mode="r") as fin:
                tmp_entity_id_list = json.load(fin)
                all_entity_id_list.extend(tmp_entity_id_list)
            # endwith
        # endfor
        print(f"There are {len(all_entity_id_list)} entity ids in the list.")
        all_entity_id_list = list(set(all_entity_id_list))
        print(f"unique entity id size : {len(all_entity_id_list)}")

        # (1)
        # when consider train_100 and valid
        # There are 7828 entity ids in the list.
        # unique entity id size : 7575

        # (2)
        # when consider train until 10000 and valid
        # There are 130916 entity ids in the list.
        # unique entity id size : 129663

        # ########## check if in the database ##########
        num_of_worker = os.cpu_count() - 2
        block_size = math.ceil(len(all_entity_id_list) / num_of_worker)
        pool = Pool(processes=num_of_worker)

        job_list = []
        for w in range(num_of_worker):
            block_list = all_entity_id_list[w *
                                            block_size: (w + 1) * block_size]
            single_worker = pool.apply_async(func=self.single_worker_check_entity_in_database,
                                             args=(block_list,))
            job_list.append(single_worker)
        # endfor

        all_in_database_set = set()
        all_not_in_database_set = set()
        for job in tqdm(job_list, desc="all jobs"):
            in_database_set, not_in_database_set = job.get()
            all_in_database_set.update(in_database_set)
            all_not_in_database_set.update(not_in_database_set)
        # endfor

        print(f"all in database set: {len(all_in_database_set)}")
        print(f"all NOT in database set: {len(all_not_in_database_set)}")

        pass

    def _single_worker_check_entity_in_all_wikidata_features_database(self,
                                                                      entity_id_list,
                                                                      worker_id):
        """
        Database: all_wikidata_features
        check the entity ids that has both (1) examples (2) features
        """
        entity_id_set_IN_database = set()
        entity_id_set_NOT_in_database = set()

        # ############ create database to store entity feature ##############
        example_and_feature_db = self.connect_to_example_and_feature_database()

        for wikidata_id in tqdm(entity_id_list, desc=f"worker-{worker_id}"):
            # >>>> check feature <<<<
            feature_flag = False
            feature_result_list = []
            # here use "wikidata_id"
            for item in example_and_feature_db.entity_feature.find({"wikidata_id": wikidata_id}):
                feature_result_list.append(item)
            # endfor
            if len(feature_result_list) > 0:
                #print(f"FEATURE: {wikidata_id} already in database")
                feature_flag = True
            # endif

            # >>>>> check example <<<<<
            example_flag = False
            example_result_list = []
            for item in example_and_feature_db.entity_example.find({"wikidata_id": wikidata_id}):
                example_result_list.append(item)
            # endfor
            if len(example_result_list) > 0:
                #print(f"EXAMPLE : {wikidata_id} already in database.")
                example_flag = True
            # endif

            # skip if both in database
            if feature_flag and example_flag:
                entity_id_set_IN_database.add(wikidata_id)
            else:
                entity_id_set_NOT_in_database.add(wikidata_id)
            # endif
        # endfor
        return entity_id_set_IN_database, entity_id_set_NOT_in_database

    def parallel_check_entity_id_LIST_in_all_wikidata_feature_database(self, all_entity_id_list):
        """
        Check entity is in all_feature_dataset
        """
        # ########## check if in the database ##########
        num_of_worker = os.cpu_count() - 2
        block_size = math.ceil(len(all_entity_id_list) / num_of_worker)
        pool = Pool(processes=num_of_worker)

        job_list = []
        for w in range(num_of_worker):
            block_list = all_entity_id_list[w *
                                            block_size: (w + 1) * block_size]
            single_worker = pool.apply_async(func=self._single_worker_check_entity_in_all_wikidata_features_database,
                                             args=(block_list, w))
            job_list.append(single_worker)
        # endfor

        all_in_database_set = set()
        all_not_in_database_set = set()
        for job in tqdm(job_list, desc="all jobs"):
            in_database_set, not_in_database_set = job.get()
            all_in_database_set.update(in_database_set)
            all_not_in_database_set.update(not_in_database_set)
        # endfor

        print(f"all in database set: {len(all_in_database_set)}")
        print(f"all NOT in database set: {len(all_not_in_database_set)}")

        # ######## output #########
        all_entity_id_IN_database_file = "./dataset/high_quality_human_id_folder/all_entity_id_IN_database.json"
        all_entity_id_NOT_in_database_file = "./dataset/high_quality_human_id_folder/all_entity_id_NOT_in_database.json"

        with open(all_entity_id_IN_database_file, mode="w") as fout:
            json.dump(list(all_in_database_set), fout)
        # endwith

        with open(all_entity_id_NOT_in_database_file, mode="w") as fout:
            json.dump(list(all_not_in_database_set), fout)
        # endwith

        pass

    def _single_worker_get_one_hop_entity_id(self,
                                             input_wikidata_id_list,
                                             shared_entity_id_list_in_database,
                                             shared_entity_id_list_NOT_in_database):

        # connect to database
        db = self.connect_to_wikidata_dump_database()

        for wikidata_entity_id in tqdm(input_wikidata_id_list, desc=f"worker"):
            # ------ get json_obj ------
            result_list = []
            for item in db.entity_obj.find({"id": wikidata_entity_id}):
                result_list.append(item)
            # endfor

            # if this entity is not in database, skip
            if len(result_list) > 0:
                shared_entity_id_list_in_database.append(wikidata_entity_id)
            else:
                shared_entity_id_list_NOT_in_database.append(
                    wikidata_entity_id)
                print(f"skip: {wikidata_entity_id}")
                continue
            # endif

            json_obj = result_list[0]
            # ----------- get all the entity id set for this json_obj -----------------
            for property_id, value_info_list in json_obj["claims"].items():

                for value_info in value_info_list:
                    rank = value_info["rank"]
                    # the rank can be {"normal", "preferred", "deprecated"}, skip deprecated.
                    if rank == "deprecated":
                        continue

                    if value_info["mainsnak"]["snaktype"] != "value":
                        continue
                    # endif

                    data_value = value_info["mainsnak"]["datavalue"]
                    data_type = data_value["type"]
                    value_obj = data_value["value"]

                    if data_type == "wikibase-entityid":
                        tmp_wikidata_entity_id = value_obj["id"]
                        result_list = []
                        for item in db.entity_obj.find({"id": tmp_wikidata_entity_id}):
                            result_list.append(item)
                        # endfor

                        if len(result_list) > 0:
                            shared_entity_id_list_in_database.append(
                                tmp_wikidata_entity_id)
                        else:
                            shared_entity_id_list_NOT_in_database.append(
                                tmp_wikidata_entity_id)
                            print(
                                f"skip: cannot find {tmp_wikidata_entity_id}")
                        # endif
                    # endif
                # endfor
            # endfor
        # endfor

    def parallel_get_all_one_hop_entity_from_entity_id_list(self,
                                                            entity_id_list,
                                                            output_entity_id_file_path_in_database,
                                                            output_entity_id_file_path_NOT_in_database):
        """
        >>>>>>>>  train_100 and valid
        There are 7828 item in all_entity_id_list.
        Unique entity ids num is 7575.

        -- one hop entities --
        There are 85206 entity ids in database.
        There are 1 entity ids NOT in database.

        >>>>>>>> train_until_10000 and valid
        There are 130916 item in all_entity_id_list.
        Unique entity ids num is 129663.

        -- one hop entities --
        There are 587380 entity ids in database.
        There are 7 entity ids NOT in database.

        """
        all_entity_id_list = entity_id_list

        # #### check if one-hot entity id are in database ######
        manager = Manager()
        shared_in_database_entity_id_list = manager.list()
        shared_NOT_in_database_entity_id_list = manager.list()

        # ######### single process ########
        # WikidataProcessor._single_worker_get_one_hop_entity_id(input_entity_id_list,
        #                                                        shared_in_database_entity_id_list,
        #                                                        shared_NOT_in_database_entity_id_list)

        # ############ parallel ##############
        num_of_worker = os.cpu_count() - 2
        pool = Pool(processes=num_of_worker)
        block_size = math.ceil(len(all_entity_id_list) / num_of_worker)

        job_list = []
        for w in range(num_of_worker):
            block_list = all_entity_id_list[w *
                                            block_size: (w + 1) * block_size]
            job = pool.apply_async(func=self._single_worker_get_one_hop_entity_id,
                                   args=(block_list,
                                         shared_in_database_entity_id_list,
                                         shared_NOT_in_database_entity_id_list))
            job_list.append(job)
        # endfor
        pool.close()

        for job in job_list:
            job.get()
        # endfor

        # ############### output file (already included input_entity_id_list) ###########
        # in database
        all_one_hop_entity_id_list_in_database = list(
            set(list(shared_in_database_entity_id_list)))
        print(
            f"There are {len(all_one_hop_entity_id_list_in_database)} entity ids in database.")
        with open(output_entity_id_file_path_in_database, mode="w") as fout:
            json.dump(all_one_hop_entity_id_list_in_database, fout)
        # endwith

        # NOT in database
        all_one_hop_entity_id_NOT_in_database = list(
            set(list(shared_NOT_in_database_entity_id_list)))
        print(
            f"There are {len(all_one_hop_entity_id_NOT_in_database)} entity ids NOT in database.")
        with open(output_entity_id_file_path_NOT_in_database, mode="w") as fout:
            json.dump(all_one_hop_entity_id_NOT_in_database, fout)
        # endwith

    def parallel_get_all_one_hop_entity_id_list(self,
                                                entity_id_file_path_list,
                                                output_entity_id_file_path_in_database,
                                                output_entity_id_file_path_NOT_in_database):
        """
        >>>>>>>>  train_100 and valid
        There are 7828 item in all_entity_id_list.
        Unique entity ids num is 7575.

        -- one hop entities --
        There are 85206 entity ids in database.
        There are 1 entity ids NOT in database.

        >>>>>>>> train_until_10000 and valid
        There are 130916 item in all_entity_id_list.
        Unique entity ids num is 129663.

        -- one hop entities --
        There are 587380 entity ids in database.
        There are 7 entity ids NOT in database.

        """
        all_entity_id_list = []
        for file_path in entity_id_file_path_list:
            with open(file_path, mode="r") as fin:
                tmp_entity_id_list = json.load(fin)
                all_entity_id_list.extend(tmp_entity_id_list)
            # endwith
        # endfor
        print(
            f"There are {len(all_entity_id_list)} item in all_entity_id_list.")
        all_entity_id_list = list(set(all_entity_id_list))
        print(f"Unique entity ids num is {len(all_entity_id_list)}.")

        # #### check if one-hot entity id are in database ######
        manager = Manager()
        shared_in_database_entity_id_list = manager.list()
        shared_NOT_in_database_entity_id_list = manager.list()

        # ######### single process ########
        # WikidataProcessor._single_worker_get_one_hop_entity_id(input_entity_id_list,
        #                                                        shared_in_database_entity_id_list,
        #                                                        shared_NOT_in_database_entity_id_list)

        # ############ parallel ##############
        num_of_worker = os.cpu_count() - 2
        pool = Pool(processes=num_of_worker)
        block_size = math.ceil(len(all_entity_id_list) / num_of_worker)

        job_list = []
        for w in range(num_of_worker):
            block_list = all_entity_id_list[w *
                                            block_size: (w + 1) * block_size]
            job = pool.apply_async(func=self._single_worker_get_one_hop_entity_id,
                                   args=(block_list,
                                         shared_in_database_entity_id_list,
                                         shared_NOT_in_database_entity_id_list))
            job_list.append(job)
        # endfor
        pool.close()

        for job in job_list:
            job.get()
        # endfor

        # ############### output file (already included input_entity_id_list) ###########
        # in database
        all_one_hop_entity_id_list_in_database = list(
            set(list(shared_in_database_entity_id_list)))
        print(
            f"There are {len(all_one_hop_entity_id_list_in_database)} entity ids in database.")
        with open(output_entity_id_file_path_in_database, mode="w") as fout:
            json.dump(all_one_hop_entity_id_list_in_database, fout)
        # endwith

        # NOT in database
        all_one_hop_entity_id_NOT_in_database = list(
            set(list(shared_NOT_in_database_entity_id_list)))
        print(
            f"There are {len(all_one_hop_entity_id_NOT_in_database)} entity ids NOT in database.")
        with open(output_entity_id_file_path_NOT_in_database, mode="w") as fout:
            json.dump(all_one_hop_entity_id_NOT_in_database, fout)
        # endwith

    # ################## check high quality human ids ##################

    def load_high_quality_human_id_set(self):
        """
        There are 8_695_657 high quality human id set.

        Returns:
            _type_: _description_
        """

        high_quality_human_id_set = set()

        with open(self.high_quality_human_id_file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                high_quality_human_id_set.add(line)
            # endfor
        # endwith

        print(
            f"There are {len(high_quality_human_id_set)} high quality human id set.")
        return high_quality_human_id_set

    def load_high_quality_property_id_set(self):
        """
        Only these candidate property ids are considered, others discarded.
        """
        with open(self.high_quality_property_str_to_num_file_path, mode="r") as fin:
            high_quality_property_str_to_num_dict = json.load(fin)
        # endwith

        high_quality_property_str_set = set(
            high_quality_property_str_to_num_dict.keys())

        return high_quality_property_str_set

    def load_cur_high_quality_human_id_list_in_all_wikidata_features_database(self):
        """
        Not all high quality human ids are created with (a) examples (b) features
        load the cur high quality humand ids in all_wikidata_featuers database
        """
        cur_human_id_IN_database_file = "./dataset/high_quality_human_id_folder/all_entity_id_IN_database.json"
        with open(cur_human_id_IN_database_file, mode="r") as fin:
            cur_human_id_IN_database_list = json.load(fin)
        # endwith

        return cur_human_id_IN_database_list

    def get_entity_label_and_description(self, json_obj):
        """
        Get the label and description
        """

        label = ""
        description = ""

        try:
            label = json_obj["labels"]["en"]["value"]
        except:
            print(f"ERROR label")
        # endtry

        try:
            description = json_obj["descriptions"]["en"]["value"]
        except:
            print(f"ERROR description")
        # endtry

        return label, description

    def _build_property_id_to_value_pair_list_from_wikidata_json_obj(self, db, json_obj, candidate_property_id_set):
        """
        FILTER Criteria
        ================
        1. any property_id of data type ExternalID are discarded (based on property id data type).
        2. For property_id of data type "string", some of them are also just identifiers, ids, etc. are discarded (based on property id label pattern, ending with code, identifier etc.)

        Output
        ======
        property_id_to_value_pair_list : the key is sorted
        property_id_to_wikidata_id_list_dict
        wikidata_id_to_property_id_list_dict

        Knowledge
        ==========
        (1)
        value_type_set:
        {'time', 'wikibase-entityid', 'globecoordinate', 'monolingualtext', 'string', 'quantity'}

        (2)
        discard_property_id: because we want to find the underlying knowledge for relation discard_property_id
        We need to remove this property in the subject, object one-hop property information. Otherwise, the
        training will be overfit on this property.

        For instance,
        Internet, contains relation -> "discoverer or inventor": Vint Cerf and Bob Kahn
        """

        # ####### connect to database #########
        # db is the wikidata_dump database
        # json_obj is the original wikidata json obj
        # candidate_property_id_set is the filtered property id set

        property_id_to_value_string_dict = {}

        # In the property relation of an entity, some properties are connected to a wikidata entity
        # This information is also very important to be recorded in the example and feature
        property_id_to_wikidata_id_list_dict = {}

        wikidata_id = json_obj["id"]

        # if wikidata_id == "Q1400":
        #     print("hello")

        for property_id, value_info_list in json_obj["claims"].items():

            if property_id == "P150":
                print("hello")
            # endif

            # >>> "DISCARD property": wikidata current list does not contain this property. Discard.
            if property_id not in candidate_property_id_set:
                print(
                    f"{property_id} id discarded by wikidata, skip when building property_value_list")
                continue
            # endif

            # one property might have multiple value
            property_id_to_value_string_dict[property_id] = []

            for value_info in value_info_list:
                rank = value_info["rank"]
                if rank == "deprecated":
                    continue

                if value_info["mainsnak"]["snaktype"] != "value":
                    continue
                # endif

                data_value = value_info["mainsnak"]["datavalue"]
                data_type = data_value["type"]
                value_obj = data_value["value"]

                if data_type == "string":
                    if len(value_obj) > 0:
                        property_id_to_value_string_dict[property_id].append(
                            value_obj)
                    # endif
                # endif

                if data_type == "quantity":
                    value_output = ""
                    for k, v in value_obj.items():
                        value_output += f"{k} {v} "
                    # endfor
                    value_output = value_output.strip()

                    if len(value_output) > 0:
                        property_id_to_value_string_dict[property_id].append(
                            value_output.strip())
                    # endif
                # endif

                if data_type == "monolingualtext":
                    mono_value_output = value_obj["text"].strip()
                    if len(mono_value_output) > 0:
                        property_id_to_value_string_dict[property_id].append(
                            mono_value_output)
                    # endif
                # endif

                if data_type == "globecoordinate":
                    coo_data_output = ""
                    for k, v in value_obj.items():
                        if k != "globe":
                            coo_data_output += f"{k} {v} "
                    # endfor
                    coo_data_output = coo_data_output.strip()
                    if len(coo_data_output) > 0:
                        property_id_to_value_string_dict[property_id].append(
                            coo_data_output)
                    # endif
                # endif

                if data_type == "wikibase-entityid":  # the tail is a wikidata entity
                    wikidata_entity_id = value_obj["id"]

                    result_list = []
                    for item in db.entity_obj.find({"id": wikidata_entity_id}):
                        result_list.append(item)
                    # endfor
                    if len(result_list) > 0:
                        wikidata_obj = result_list[0]
                    else:
                        with open("tmp_related_wikidata_entity_id_list.txt", mode="a") as fout:
                            fout.write(f"{wikidata_entity_id}\n")
                        # endwith
                        print(f"ERROR: cannot find {wikidata_entity_id}")
                        continue
                    # endif

                    label = ""
                    des = ""
                    try:
                        label = wikidata_obj["labels"]["en"]["value"]
                    except:
                        print(f"No label: {wikidata_entity_id}")
                    # endtry

                    try:
                        des = wikidata_obj["descriptions"]["en"]["value"]
                    except:
                        print(f"No description: {wikidata_entity_id}")
                    # endtry
                    if len(label) == 0 and len(des) == 0:
                        continue

                    entity_value_output = f"{label}, {des}"
                    entity_value_output = entity_value_output.strip()

                    if len(entity_value_output) > 0:
                        property_id_to_value_string_dict[property_id].append(
                            entity_value_output)
                    # endif

                    # ######## add to dictionary -> property_id_to_wikidata_id_list_dict ############
                    if property_id not in property_id_to_wikidata_id_list_dict:
                        property_id_to_wikidata_id_list_dict[property_id] = []
                    # endif
                    property_id_to_wikidata_id_list_dict[property_id].append(
                        wikidata_entity_id)
                # endif

                if data_type == "time":
                    time_value_output = ""
                    for k, v in value_obj.items():
                        time_value_output += f"{k} {v} "
                    # endfor
                    time_value_output = time_value_output.strip()
                    if len(time_value_output) > 0:
                        property_id_to_value_string_dict[property_id].append(
                            time_value_output)
                    # endif
                # endif
            # endfor
        # endfor

        # ############# I. ################
        wikidata_id_to_property_id_list_dict = {}
        for property_id, wikidata_id_list in property_id_to_wikidata_id_list_dict.items():
            for tmp_id in wikidata_id_list:
                if tmp_id not in wikidata_id_to_property_id_list_dict:
                    wikidata_id_to_property_id_list_dict[tmp_id] = []
                # endif
                wikidata_id_to_property_id_list_dict[tmp_id].append(
                    property_id)
            # endfor
        # endfor

        # ############# II. ###############
        # sort and create property_id_to_value_pair_list
        property_id_to_value_pair_list = []
        for tmp_property_id, value_text_list in sorted(property_id_to_value_string_dict.items(), key=lambda x: int(x[0].replace("P", ""))):
            
            assert isinstance(value_text_list, list)
            property_id_to_value_pair_list.append(
                [tmp_property_id, value_text_list])
            # endif
        # endfor

        return property_id_to_value_pair_list, property_id_to_wikidata_id_list_dict, wikidata_id_to_property_id_list_dict

    def _bert_convert_text_to_feature_vector_simple(self, text):
        """
        text: "I love chicago"
        inputs: <class 'transformers.tokenization_utils_base.BatchEncoding'>
        --------
        input_ids: tensor([[ 101, 1045, 2293, 3190,  102]])
        token_type_ids: tensor([[0, 0, 0, 0, 0]])
        attention_mask tensor([[1, 1, 1, 1, 1]])

        It returns the same as _bert_convert_text_to_feature_vector_cls()
        """

        inputs = self.bert_tokenizer(text, return_tensors="pt")
        inputs.to(self.args.device)

        # self.bert_model.eval()

        with torch.no_grad():
            outputs, _ = self.bert_model(**inputs)
        # endwith

        outputs = outputs.squeeze(dim=0)
        text_vector = outputs[0].to('cpu').numpy()
        return text_vector

    def _bert_convert_text_to_feature_vector_cls(self, text):
        token_list = text.strip().split()

        word_pieces_list = []
        word_boundaries_list = []

        for w in token_list:
            word_pieces = self.bert_tokenizer.tokenize(w)
            word_boundaries_list.append([len(word_pieces_list), len(word_pieces_list) + len(
                word_pieces)])  # Note: original word token corresponding to word piece boundary
            word_pieces_list += word_pieces
        # endfor
        assert len(word_boundaries_list) == len(token_list)

        total_input_tokens = ['[CLS]'] + word_pieces_list + ['[SEP]']
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(
            total_input_tokens)
        segment_ids = [0] * len(total_input_tokens)

        input_ids = torch.tensor(
            [input_ids], dtype=torch.long).to(self.args.device)
        segment_ids = torch.tensor(
            [segment_ids], dtype=torch.long).to(self.args.device)

        with torch.no_grad():
            sequence_output, _ = self.bert_model(input_ids=input_ids,
                                                 token_type_ids=segment_ids)
        # endwith

        sequence_output = sequence_output.squeeze(dim=0)
        # the first one corresponds to [CLS]
        text_vector = sequence_output[0].to('cpu').numpy()
        return text_vector

    def get_bert_embed(self, text_list):
        """
        Input the original text into bert model, without mask two objects.
        :return:
        """
        assert isinstance(text_list, list)

        if self.bert_model is None and self.bert_tokenizer is None:
            from transformers import BertModel, BertTokenizer
            print("loading bert model: {} to create feature for sent tokens".format(
                self.args.pretrained_transformer_model_name_or_path))
            self.bert_model = BertModel.from_pretrained(
                self.args.pretrained_transformer_model_name_or_path)
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                self.args.pretrained_transformer_model_name_or_path)
            # load the pretrained bert model. The loaded tokenizer will follow the original pretrained config
            # For "do_lower_case" will decide whether or not make the sent_text lower case
            # It should consistent with the programs argument config
            # bert_pretrained_model, bert_pretrained_tokenizer = self._load_pretrained_bert_model(args, model_class,
            # tokenizer_class)
            self.bert_model.eval()
            self.bert_model.to(self.args.device)
            print(
                f"{self.args.pretrained_transformer_model_name_or_path} is successfully loaded.")
        # endif

        text_vector_list = []
        # for text in tqdm(text_list, desc="create bert embedding for text"):
        for text in text_list:
            text_vector = self._bert_convert_text_to_feature_vector_simple(
                text)
            # text_vector = self._bert_convert_text_to_feature_vector_cls(text) # this one is more complicated, discard
            text_vector_list.append(text_vector)
        # endfor

        return text_vector_list

    def _single_worker_create_wikidata_entity_example_and_feature_object_and_store_to_database(self,
                                                                                               wikidata_id_list,
                                                                                               candidate_property_id_set,
                                                                                               worder_index):
        """
        Note!!
        (1) create example for each entity and store into database

        (2) Create feature for each property (including target property that need to be discarded.) # TODO: maybe do not nee this
        Discard it when create negative example.
        """
        wikidata_id_set_not_in_database = set()

        # ############ create database to store entity feature ##############
        example_and_feature_db = self.connect_to_example_and_feature_database()
        wikidata_dump_db = self.connect_to_wikidata_dump_database()

        # (1) ############# create wikidata example #############
        for wikidata_id in tqdm(wikidata_id_list, desc=f"worker-{worder_index}"):

            result_list = []
            # in wikidata_dump check "id"
            for item in wikidata_dump_db.entity_obj.find({"id": wikidata_id}):
                result_list.append(item)
            # endfor

            wikidata_obj = None
            if len(result_list) > 0:
                wikidata_obj = result_list[0]
            # endif

            # not include download here
            # if len(result_list) == 0:
            #     wikidata_obj = self.download_single_entity_id_and_store_to_database(
            #         wikidata_id)
            # # endif

            if wikidata_obj is None:
                print(f"{wikidata_id} not in database.")
                wikidata_id_set_not_in_database.add(wikidata_id)
                continue

            # ############### BEGIN: if this id feature has already created and stored in database, skip #########

            # >>>> check feature <<<<
            feature_flag = False
            feature_result_list = []
            # here use "wikidata_id"
            for item in example_and_feature_db.entity_feature.find({"wikidata_id": wikidata_id}):
                feature_result_list.append(item)
            # endfor
            if len(feature_result_list) > 0:
                print(f"FEATURE: {wikidata_id} already in database")
                feature_flag = True
            # endif

            # >>>>> check example <<<<<
            example_flag = False
            example_result_list = []
            for item in example_and_feature_db.entity_example.find({"wikidata_id": wikidata_id}):
                example_result_list.append(item)
            # endfor
            if len(example_result_list) > 0:
                print(f"EXAMPLE : {wikidata_id} already in database.")
                example_flag = True
            # endif

            # skip if both in database
            if feature_flag and example_flag:
                print(
                    f"SKIP: {wikidata_id}, both feature and example in database.")
                continue
            # endif
            # ############## END: if either feature and example is not in database, update the database for both ##########

            # ########### if not create entity feature #################
            label, des = self.get_entity_label_and_description(wikidata_obj)
            if len(label) == 0 and len(des) == 0:
                continue
            # endif

            if wikidata_obj is not None:

                # old one
                # property_id_to_value_pair_dict, property_id_to_wikidata_id_list_dict, wikidata_id_to_property_id_list_dict = \
                #     self._build_property_id_to_value_pair_dict_from_wikidata_json_obj(wikidata_dump_db,
                #                                                                       wikidata_obj,
                #                                                                       candidate_property_id_set)

                # first item is a list with (property_id, value_text_list)
                # # already filtered property id based on "candiate_property_id_set" in this method
                property_id_to_value_pair_list, property_id_to_wikidata_id_list_dict, wikidata_id_to_property_id_list_dict = \
                    self._build_property_id_to_value_pair_list_from_wikidata_json_obj(wikidata_dump_db,
                                                                                      wikidata_obj,
                                                                                      candidate_property_id_set)

                # To make it consistent with data in database, convert list to dict
                property_id_to_value_pair_dict = {}
                for tmp_property_id, tmp_value_text_list in property_id_to_value_pair_list:
                    property_id_to_value_pair_dict[tmp_property_id] = tmp_value_text_list
                # endfor

                # add label and description into property list
                # if len(des) > 0:
                #     description_to_value_pair = ["P-2", [des]]
                # # endif

                # if len(label) > 0:
                #     label_to_value_pair = ["P-1", [label]]
                # # endif

                if len(des) > 0 and len(label) > 0:
                    property_id_to_value_pair_dict.update(
                        {"P-2": [des], "P-1": [label]})

                    # property_id_to_value_pair_list = [description_to_value_pair, label_to_value_pair] + \
                    #     property_id_to_value_pair_list
                if len(des) > 0 and len(label) == 0:
                    property_id_to_value_pair_dict.update({"P-2": [des]})

                    # property_id_to_value_pair_list = [
                    #     description_to_value_pair] + property_id_to_value_pair_list

                if len(des) == 0 and len(label) > 0:
                    property_id_to_value_pair_dict.update({"P-1": [label]})

                    # property_id_to_value_pair_list = [
                    #     label_to_value_pair] + property_id_to_value_pair_list

                sorted_property_id_to_value_pair_list = []
                for k, v in sorted(property_id_to_value_pair_dict.items(), key=lambda x: int(x[0].replace("P", ""))):
                    assert isinstance(v, list)
                    sorted_property_id_to_value_pair_list.append([k, v])
                    # sorted_property_id_to_value_pair_dict[k] = v
                # endfor

                # ########################## (1) create example and store into database #################

                example_entry = {"wikidata_id": wikidata_id,
                                 "label": label,
                                 "description": des,
                                 "property_id_to_value_text_list_tuple": sorted_property_id_to_value_pair_list,
                                 "property_id_to_wikidata_id_list_dict": property_id_to_wikidata_id_list_dict,
                                 "wikidata_id_to_property_id_list_dict": wikidata_id_to_property_id_list_dict
                                 }

                example_and_feature_db.entity_example.insert_one(example_entry)

                # ########################## (2) create feature and store into database #################
                # property_id_to_feature_pair_dict = {}
                property_id_to_feature_embed_list = []
                

                for property_id, value_text_list in sorted_property_id_to_value_pair_list:
                    
                    assert isinstance(value_text_list, list)
                    
                    text_vector_list = self.get_bert_embed(value_text_list)
                    if len(text_vector_list) == 0:
                        continue
                    # get mean of the vector, when len(text_vector_list) == 1, it is OK
                    arr = np.array(text_vector_list)
                    mean_vector = np.mean(arr, axis=0)

                    # property_id_to_feature_pair_list.append(
                    #     [property_id, mean_vector.tolist()])
                    # property_id_to_feature_pair_dict[property_id] = mean_vector.tolist()
                    property_id_to_feature_embed_list.append([property_id, mean_vector.tolist()])

                # endfor

                feature_entry = {"wikidata_id": wikidata_id,
                                 "label": label,
                                 "description": des,
                                 "property_id_to_feature_embed_list": property_id_to_feature_embed_list,
                                 "property_id_to_wikidata_id_list_dict": property_id_to_wikidata_id_list_dict,
                                 "wikidata_id_to_property_id_list_dict": wikidata_id_to_property_id_list_dict
                                 }

                example_and_feature_db.entity_feature.insert_one(feature_entry)
            # endif
        # endfor

    def main_parallel_create_feature_for_entity_id_list(self, all_entity_ids_list):
        """
        When run in parallel, it has error.

        RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, 
        you must use the 'spawn' start method



        old method name:
        main_parallel_create_feature_for_all_entities_exclude_human_entity_pool

        run the split entity ids files
        """
        # 1. load candidate property list
        candidate_property_id_set = self.load_high_quality_property_id_set()

        # ############## single #################

        self._single_worker_create_wikidata_entity_example_and_feature_object_and_store_to_database(all_entity_ids_list,
                                                                                                    candidate_property_id_set,
                                                                                                    0)

        # ############## parallel ###############
        # num_of_worker = os.cpu_count() - 2
        # block_size = math.ceil(len(all_entity_ids_list) / num_of_worker)
        # pool = Pool(processes=num_of_worker)

        # job_list = []
        # for w in range(num_of_worker):
        #     block_list = all_entity_ids_list[w *
        #                                      block_size: (w + 1) * block_size]
        #     job = pool.apply_async(func=self._single_worker_create_wikidata_entity_example_and_feature_object_and_store_to_database,
        #                            args=(block_list,
        #                                  candidate_property_id_set,
        #                                  w))
        #     job_list.append(job)
        # # endfor
        # pool.close()

        # for job in job_list:
        #     job.get()
        # # endfor

        pass

    # ############ create triple features ################

    def extract_features_with_property_str(self, wikidata_feature_obj, discard_entity_id_set, tmp_discard_property_id_set):
        """
        discard potential knowledge leak
        """
        wikidata_id = wikidata_feature_obj["wikidata_id"]
        property_id_to_feature_embed_list = wikidata_feature_obj["property_id_to_feature_embed_list"]

        # find out discard property id set
        discard_property_id_set = set()
        discard_property_id_set.update(tmp_discard_property_id_set)
        wikidata_id_to_property_id_list_dict = wikidata_feature_obj[
            "wikidata_id_to_property_id_list_dict"]
        for tmp_wikidata_entity_id, tmp_property_id_list in wikidata_id_to_property_id_list_dict.items():
            if tmp_wikidata_entity_id in discard_entity_id_set:
                discard_property_id_set.update(tmp_property_id_list)
            # endif
        # endfor

        # if len(discard_property_id_set) > 1:
        #     print("hello")
        # #endif

        property_str_list = []
        property_value_embed_list = []
        for tmp_property_id, tmp_feature_value in sorted(property_id_to_feature_embed_list):

            if tmp_property_id in discard_property_id_set:
                continue
            # endif

            # property_str -> value_embedding are aligned one to one mapping
            property_str_list.append(tmp_property_id)
            property_value_embed_list.append(
                np.array(tmp_feature_value).astype(np.float32))
        # endfor

        return wikidata_id, property_str_list, property_value_embed_list

    def NORMAL_get_feature_from_entity_pair_ids_and_label_list(self, entity_pair_ids_and_label_list, mode, job_index):
        """
        Jan 21, 2021

        Discard potential knowledge leak, the same treatment as create example object
        """
        # ############# connect to feature database ##########
        wikidata_triple_feature_db = self.connect_to_wikidata_triple_feature_database()
        entity_example_and_feature_feature_db = self.connect_to_example_and_feature_database()

        # ######################### utils methods #######################
        def find_feature_json_obj_in_db(db, wikidata_id):
            result_list = []
            for item in db.entity_feature.find({"wikidata_id": wikidata_id}):
                result_list.append(item)
            # endfor
            if len(result_list) > 0:
                return result_list[0]
            else:
                return None

        # ################################# START building examples ########################################

        for guid, head_wikidata_id, tail_wikidata_id, rc_label_text, nd_label_text in tqdm(
                entity_pair_ids_and_label_list, desc=f"job-{job_index} fetch feature from db"):

            # Do not do it yet! Do it during training, so that it can adapt to new index
            # rc_label_index = self.rc_label_str_to_index_dict[rc_label_text]
            # nd_label_index = self.nd_label_str_to_index_dict[nd_label_text]

            skip_head = False
            skip_tail = False

            # head discard entity
            head_discard_entity_id_set = set()
            head_discard_entity_id_set.add(
                tail_wikidata_id)  # to avoid information leak
            head_feature_obj = find_feature_json_obj_in_db(
                entity_example_and_feature_feature_db, head_wikidata_id)

            # head discard property
            head_discard_property_str_set = set()
            head_discard_property_str_set.add(rc_label_text)
            if rc_label_text in {"P61", "P170"}:
                head_discard_property_str_set.add("P800")
            # endif

            if head_feature_obj is not None:
                # head_wikidata_id, head_property_index_list, head_property_value_embed_list = self.extract_features(
                #     head_feature_obj, self.property_str_to_index_dict, head_discard_entity_id_set, rc_label_text)

                head_wikidata_id, head_property_str_list, head_property_value_embed_list = self.extract_features_with_property_str(
                    head_feature_obj, head_discard_entity_id_set, head_discard_property_str_set)

            else:
                skip_head = True
                # lock.acquire()
                # tmp_output_file = os.path.join(self.args.data_folder,
                #                                f"{self.args.data_tag}_{mode}_data_entity_id_not_in_FEATURE_database.txt")
                #
                # with open(tmp_output_file, mode="a") as fout:
                #     fout.write(f"{mode}\t{head_wikidata_id}\n")
                # # endwith
                # lock.release()
            # endif

            # tail discard entity
            tail_discard_entity_id_set = set()
            tail_discard_entity_id_set.add(head_wikidata_id)
            tail_feature_obj = find_feature_json_obj_in_db(
                entity_example_and_feature_feature_db, tail_wikidata_id)

            # trail discard property
            tail_discard_property_str_set = set()
            tail_discard_property_str_set.add(rc_label_text)
            if rc_label_text in {"P61", "P170"}:
                tail_discard_property_str_set.add("P800")
            # endif

            if tail_feature_obj is not None:
                tail_wikidata_id, tail_property_str_list, tail_property_value_embed_list = self.extract_features_with_property_str(
                    tail_feature_obj, tail_discard_entity_id_set, tail_discard_property_str_set)
            else:
                skip_tail = True
                # lock.acquire()
                # tmp_output_file = os.path.join(self.args.data_folder,
                #                                f"{self.args.data_tag}_{mode}_data_entity_id_not_in_FEATURE_database.txt")
                #
                # with open(tmp_output_file, mode="a") as fout:
                #     fout.write(f"{mode}\t{tail_wikidata_id}\n")
                # # endwith
                # lock.release()
            # endif

            # ####### SKIP ##########
            if skip_head or skip_tail:
                continue
            # endif

            # feature_entry = WikidataEntityPairFeature(guid=guid,
            #                                           head_wikidata_id=head_wikidata_id,
            #                                           tail_wikidata_id=tail_wikidata_id,
            #                                           head_property_id_list=head_property_index_list,
            #                                           tail_property_id_list=tail_property_index_list,
            #                                           head_property_value_embed_list=head_property_value_embed_list,
            #                                           tail_property_value_embed_list=tail_property_value_embed_list,
            #                                           rc_label_index=rc_label_index,
            #                                           nd_label_index=nd_label_index)
            assert len(head_property_str_list) == len(
                head_property_value_embed_list)
            assert len(tail_property_str_list) == len(
                tail_property_value_embed_list)

            head_property_value_embed_list = [
                item.tolist() for item in head_property_value_embed_list]
            tail_property_value_embed_list = [
                item.tolist() for item in tail_property_value_embed_list]

            # make property id as str, so that we can flexibly adapt it to other property mapping
            wikidata_entry = {
                "guid": guid,
                "head_wikidata_id": head_wikidata_id,
                "tail_wikidata_id": tail_wikidata_id,
                "head_property_id_list": head_property_str_list,
                "tail_property_id_list": tail_property_str_list,
                "head_property_value_embed_list": head_property_value_embed_list,
                "tail_property_value_embed_list": tail_property_value_embed_list,
                "rc_label_index": rc_label_text,
                "nd_label_index": nd_label_text
            }

            tmp_result = []
            for item in wikidata_triple_feature_db[f"{rc_label_text}_normal"].find({"guid": guid}):
                tmp_result.append(item)
            # endfor

            if len(tmp_result) == 0:
                wikidata_triple_feature_db[f"{rc_label_text}_normal"].insert_one(
                    wikidata_entry)
            # endif

            # feature_list.append(feature_entry)
        # endfor
        # return feature_list

    def NOVEL_get_feature_from_entity_pair_ids_and_label_list(self, entity_pair_ids_and_label_list, mode, job_index):
        """
        Jan 21, 2021

        Discard potential knowledge leak, the same treatment as create example object
        """
        # ############# connect to feature database ##########
        wikidata_triple_feature_db = self.connect_to_wikidata_triple_feature_database()
        entity_example_and_feature_feature_db = self.connect_to_example_and_feature_database()

        # ######################### utils methods #######################
        def find_feature_json_obj_in_db(db, wikidata_id):
            result_list = []
            for item in db.entity_feature.find({"wikidata_id": wikidata_id}):
                result_list.append(item)
            # endfor
            if len(result_list) > 0:
                return result_list[0]
            else:
                return None

        # ################################# START building examples ########################################

        for guid, head_wikidata_id, tail_wikidata_id, rc_label_text, nd_label_text in tqdm(
                entity_pair_ids_and_label_list, desc=f"job-{job_index} fetch feature from db"):

            # ### if duplicated skip !
            entity_pair_set = set()
            for item in wikidata_triple_feature_db[f"{rc_label_text}_novel"].find({"guid": guid}):
                entity_pair_set.add(
                    (item["head_wikidata_id"], item["tail_wikidata_id"]))
            # endfor
            cur_entity_pair = (head_wikidata_id, tail_wikidata_id)
            if cur_entity_pair in entity_pair_set:
                continue
            # endif

            # Do not do it yet! Do it during training, so that it can adapt to new index
            # rc_label_index = self.rc_label_str_to_index_dict[rc_label_text]
            # nd_label_index = self.nd_label_str_to_index_dict[nd_label_text]

            skip_head = False
            skip_tail = False

            # head discard entity
            head_discard_entity_id_set = set()
            head_discard_entity_id_set.add(
                tail_wikidata_id)  # to avoid information leak
            head_feature_obj = find_feature_json_obj_in_db(
                entity_example_and_feature_feature_db, head_wikidata_id)

            # head discard property
            head_discard_property_str_set = set()
            head_discard_property_str_set.add(rc_label_text)
            if rc_label_text in {"P61", "P170"}:
                head_discard_property_str_set.add("P800")
            # endif

            if head_feature_obj is not None:
                # head_wikidata_id, head_property_index_list, head_property_value_embed_list = self.extract_features(
                #     head_feature_obj, self.property_str_to_index_dict, head_discard_entity_id_set, rc_label_text)

                head_wikidata_id, head_property_str_list, head_property_value_embed_list = self.extract_features_with_property_str(
                    head_feature_obj, head_discard_entity_id_set, head_discard_property_str_set)

            else:
                skip_head = True
                # lock.acquire()
                # tmp_output_file = os.path.join(self.args.data_folder,
                #                                f"{self.args.data_tag}_{mode}_data_entity_id_not_in_FEATURE_database.txt")
                #
                # with open(tmp_output_file, mode="a") as fout:
                #     fout.write(f"{mode}\t{head_wikidata_id}\n")
                # # endwith
                # lock.release()
            # endif

            # tail discard entity
            tail_discard_entity_id_set = set()
            tail_discard_entity_id_set.add(head_wikidata_id)
            tail_feature_obj = find_feature_json_obj_in_db(
                entity_example_and_feature_feature_db, tail_wikidata_id)

            # trail discard property
            tail_discard_property_str_set = set()
            tail_discard_property_str_set.add(rc_label_text)
            if rc_label_text in {"P61", "P170"}:
                tail_discard_property_str_set.add("P800")
            # endif

            if tail_feature_obj is not None:
                tail_wikidata_id, tail_property_str_list, tail_property_value_embed_list = self.extract_features_with_property_str(
                    tail_feature_obj, tail_discard_entity_id_set, tail_discard_property_str_set)
            else:
                skip_tail = True
                # lock.acquire()
                # tmp_output_file = os.path.join(self.args.data_folder,
                #                                f"{self.args.data_tag}_{mode}_data_entity_id_not_in_FEATURE_database.txt")
                #
                # with open(tmp_output_file, mode="a") as fout:
                #     fout.write(f"{mode}\t{tail_wikidata_id}\n")
                # # endwith
                # lock.release()
            # endif

            # ####### SKIP ##########
            if skip_head or skip_tail:
                continue
            # endif

            # feature_entry = WikidataEntityPairFeature(guid=guid,
            #                                           head_wikidata_id=head_wikidata_id,
            #                                           tail_wikidata_id=tail_wikidata_id,
            #                                           head_property_id_list=head_property_index_list,
            #                                           tail_property_id_list=tail_property_index_list,
            #                                           head_property_value_embed_list=head_property_value_embed_list,
            #                                           tail_property_value_embed_list=tail_property_value_embed_list,
            #                                           rc_label_index=rc_label_index,
            #                                           nd_label_index=nd_label_index)
            assert len(head_property_str_list) == len(
                head_property_value_embed_list)
            assert len(tail_property_str_list) == len(
                tail_property_value_embed_list)

            head_property_value_embed_list = [
                item.tolist() for item in head_property_value_embed_list]
            tail_property_value_embed_list = [
                item.tolist() for item in tail_property_value_embed_list]

            # make property id as str, so that we can flexibly adapt it to other property mapping
            wikidata_entry = {
                "guid": guid,
                "head_wikidata_id": head_wikidata_id,
                "tail_wikidata_id": tail_wikidata_id,
                "head_property_id_list": head_property_str_list,
                "tail_property_id_list": tail_property_str_list,
                "head_property_value_embed_list": head_property_value_embed_list,
                "tail_property_value_embed_list": tail_property_value_embed_list,
                "rc_label_index": rc_label_text,
                "nd_label_index": nd_label_text
            }

            wikidata_triple_feature_db[f"{rc_label_text}_novel"].insert_one(
                wikidata_entry)
            # feature_list.append(feature_entry)
        # endfor
        # return feature_list

    def create_normal_feature_for_wikidata_triple(self, triple_size_cap):
        """
        entity_pair_ids_label_list:
        guid, head_wikidata_id, tail_wikidata_id, rc_label_text, nd_label_text

        'P135-4249--56c493d0-c434-4038-86cc-c39b0f7a9964', 'Q296047', 'Q3347105', 'P135', 'normal'

        Args:
            candidate_property_id_set:
            wikidata_triple_list:

        Returns:

        """
        print(f"PATH -- {self.args.triple_file_path}")
        # sys.exit(0)

        # ############## load all wikidata triple ids ############
        triple_list = []
        with open(self.args.triple_file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                triple_list.append(line)
            # endfor
        # endwith
        print(f"There are {len(triple_list)} wikidata triples in this file.")
        # #############

        # (2) load candidate property list
        candidate_property_id_set = self.load_high_quality_property_id_set()

        for triple_item in tqdm(triple_list[:triple_size_cap], desc="all triples"):
            guid, head_id, relation_id, tail_id = triple_item.split(",")

            # create example and feature first
            self._single_worker_create_wikidata_entity_example_and_feature_object_and_store_to_database(
                [head_id, tail_id],
                candidate_property_id_set,
                0)

            entity_pair_ids_label_list = []
            entity_pair_ids_label_list.append(
                [guid, head_id, tail_id, relation_id, "normal"])

            self.NORMAL_get_feature_from_entity_pair_ids_and_label_list(
                entity_pair_ids_label_list, "train", 1)
            # create normal feature for entity pairs
        # endfor

    def create_novel_feature_for_wikidata_triple(self, triple_size_cap):

        # ############## load all wikidata triple ids ############
        triple_list = []
        with open(self.args.triple_file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                triple_list.append(line)
            # endfor
        # endwith
        print(
            f"There are {len(triple_list)} wikidata triples in this file for novel feature creation")
        # #############

        # (2) load candidate property list
        candidate_property_id_set = self.load_high_quality_property_id_set()

        # (3)
        relation_id_to_human_pos_dict = self.load_each_relation_human_position_info_dict()

        # 1. load human position information
        # self.relation_id_to_human_pos_dict

        # 2. get high quality human id set
        cur_database_human_id_set = set(
            self.load_cur_database_high_quality_human_id_set())

        for triple_item in tqdm(triple_list[:triple_size_cap], desc="all triples"):
            guid, head_id, relation_id, tail_id = triple_item.split(",")

            # create novel examples
            candidate_human_id_set = self.randomly_sample_n_item_from_set(
                cur_database_human_id_set, 5)

            entity_pair_ids_label_list = []

            for human_id in candidate_human_id_set:
                if relation_id_to_human_pos_dict[relation_id] == "head":
                    entity_pair_ids_label_list.append(
                        [guid, human_id, tail_id, relation_id, "novel"])  # novel label

                if relation_id_to_human_pos_dict[relation_id] == "tail":
                    entity_pair_ids_label_list.append(
                        [guid, head_id, human_id, relation_id, "novel"])  # novel label
            # endif

            all_entity_id_set = set()
            all_entity_id_set.update(candidate_human_id_set)
            all_entity_id_set.add(head_id)
            all_entity_id_set.add(tail_id)
            # create example and feature first
            self._single_worker_create_wikidata_entity_example_and_feature_object_and_store_to_database(
                list(all_entity_id_set),
                candidate_property_id_set,
                0)

            # create novel feature for entity pairs
            self.NOVEL_get_feature_from_entity_pair_ids_and_label_list(
                entity_pair_ids_label_list, "train", 1)
        # endfor

    def create_triple_feature_creation_scripts(self):
        import copy

        # #### output folder
        output_folder = "./dataset/triple_feature_creation_script"

        # #### template input
        template_input_file = "./dataset/create_triple_feature_script_template.sh"
        template_str = None
        with open(template_input_file, mode="r") as fin:
            template_str = fin.read()
        # endwith

        # #### relation triple folder
        relation_triple_folder = "./dataset/ALL_Relations_unique_triples"

        rc_relation_id_list = self.get_rc_relation_str_list()

        for mode in ["normal", "novel"]:
            for index, relation_str in enumerate(rc_relation_id_list):
                index += 1

                # relation triple path
                relation_triple_file_path = os.path.join(
                    relation_triple_folder, f"{relation_str}_triple.txt")

                cur_template_str = copy.deepcopy(template_str)
                cur_template_str = cur_template_str.replace(
                    "[feature_type]", mode)
                cur_template_str = cur_template_str.replace(
                    "[relation_triple_path]", relation_triple_file_path)

                # output file_path
                output_file_path = os.path.join(
                    output_folder, f"{mode}_relation_{index}.sh")
                with open(output_file_path, mode="w") as fout:
                    fout.write(f"{cur_template_str}\n")
                # endwith
            # endfor
        # endfor

        pass

    def count_stats_in_wikidata_triple_features(self):

        wikidata_triple_db = self.connect_to_wikidata_triple_feature_database()
        all_collections_name_list = wikidata_triple_db.list_collection_names()

        relation_to_normal_novel_num_dict = {}

        for collection_name in all_collections_name_list:
            print(f"checking collecion {collection_name}")

            relation_id = collection_name.split("_")[0]
            mode = collection_name.split("_")[1]
            if relation_id not in relation_to_normal_novel_num_dict:
                relation_to_normal_novel_num_dict[relation_id] = {
                    "normal": 0, "novel": 0}
            # endif

            num = wikidata_triple_db[f"{collection_name}"].count_documents({})
            relation_to_normal_novel_num_dict[relation_id][mode] = num

        # endfor

        ###### output ####
        output_file = "./dataset/cur_wikidata_triple_features_stats.txt"
        with open(output_file, mode="w") as fout:
            fout.write("{:<10}{:<10}{:<10}\n".format(
                'relation', 'normal', 'novel'))
            for relation_id, stats_info_dict in sorted(relation_to_normal_novel_num_dict.items(), key=lambda x: int(x[0].replace("P", ""))):
                fout.write(
                    f"{relation_id:<10} {stats_info_dict['normal']:<10} {stats_info_dict['novel']:<10}\n")
            # endfor
        # endwith

        pass

    def pymongo_create_index_for_collections(self):
        """
        automatically create index, instead of manually create using interface in mongodb compass
        """
        wikidata_triple_db = self.connect_to_wikidata_triple_feature_database()
        all_collections_name_list = wikidata_triple_db.list_collection_names()

        for collection_name in all_collections_name_list:
           

            # if collection_name != "P6_normal":
            #     continue

            index_info = wikidata_triple_db[f"{collection_name}"].index_information(
            )
            
            skip = False
            for k, v in index_info.items():
                if v["key"][0][0] == "guid":
                    skip = True
                    break
                #endfor
            #endfor
            
            if skip:
                print(f"skip: {collection_name}")
                continue
            
            # check first if guid index is already created
            if "guid" not in index_info:
                print(f"creating guid index for collection - {collection_name} ...")
                wikidata_triple_db[f"{collection_name}"].create_index('guid')
            # endif
        # endfor
        pass

    # TODO:
    # ############### create example and feature from database ###########
    
    def load_and_cache_examples_with_e1_e2(self, args, mode=None):
        """
        Create example for "e1, e2 version" of validation and test dataset
        
        Args:
            args (_type_): This args should come from main.py file, be consistent with main.py file's args
            mode (_type_, optional): valid or test. Defaults to None. Train data is loaded dynamically from database.
        """
        # train example is dynamically generated
        assert mode in ["valid", "test"]
        os.makedirs(args.valid_test_example_and_feature_folder, exist_ok=True)

        # ######## create examples #########
        examples = None
        cached_example_file_path = None
        if mode == "train":
            # cached_example_file_path = os.path.join(
            #     args.data_folder, f"{mode}_cached_example_each_relation_size_{args.data_tag}")
            print(">>>>>> should not load train data here.")
            sys.exit(0)
            pass
        else:
            cached_example_file_path = os.path.join(args.valid_test_example_and_feature_folder, f"{mode}_cached_example")
        # endif
        assert cached_example_file_path is not None

        # if cased file exists, load file
        if os.path.exists(cached_example_file_path):
            print(
                f"Loading examples from cached file: {cached_example_file_path}")
            with open(cached_example_file_path, mode="rb") as fin:
                examples = pickle.load(fin)
            # endwith
            print(f"{mode} dataset is loaded.")
            return examples
        else:
            if mode == "valid":
                validation_folder_nd_label_pair_list = [self.valid_normal_folder,
                                                        self.valid_novel_folder]
                examples = self.load_valid_OR_test_with_e12_examples_from_folder(validation_folder_nd_label_pair_list)
            # endif
            if mode == "test":
                test_folder_nd_label_pair_list = [self.test_normal_folder,
                                                  self.test_novel_folder]
                examples = self.load_valid_OR_test_with_e12_examples_from_folder(test_folder_nd_label_pair_list)
            # endif
            assert examples is not None

            # ---------------------------------------------------------------------------
            # ############### sort entry ids and reorder ################
            example_dict = {}
            for exp in examples:
                guid = exp.guid
                example_dict[guid] = exp
            # endfor

            # ############## reorder ###############
            sorted_example_list = []
            for guid, exp in sorted(example_dict.items(), key=lambda x: x[0]):
                sorted_example_list.append(exp)
            # endfor
            # ---------------------------------------------------------------------------

            print("Saving examples into cached file {}".format(
                cached_example_file_path))
            with open(cached_example_file_path, mode="wb") as fout:
                pickle.dump(sorted_example_list, fout)
            # endwith
            return sorted_example_list
        # endif

    def load_and_cache_features_with_e1_e2(self, args, mode=None):
        """
        for dev, test dataset
        h, t version
        
        not from database, from text
        """
        assert mode in [
            "valid", "test"]  # train example is dynamically generated
        feature_type = args.feature_type
        os.makedirs(
            args.valid_test_example_and_feature_folder, exist_ok=True)

        # if feature_type == "glove":
        #     assert model_class == None
        #     assert tokenizer_class == None
        # elif "bert" in feature_type:
        #     assert model_class is not None
        #     assert tokenizer_class is not None
        # # endif

        cached_features_file = None
        if args.feature_type == "bert":
            if mode == "train":
                # cached_features_file = os.path.join(args.data_folder,
                #                                     f"{mode}_cached_feature_{args.feature_type}_{args.data_tag}")
                print(">>>>>> should not load train data here.")
                sys.exit(0)
            else:
                cached_features_file = os.path.join(args.valid_test_example_and_feature_folder,
                                                    f"{mode}_cached_feature_{args.feature_type}")
            # endif
        # endif

        if args.feature_type == "glove":
            cached_features_file = os.path.join(args.valid_test_example_and_feature_folder,
                                                f"{mode}_cached_feature_{args.max_seq_length}_{args.feature_type}")

        assert cached_features_file is not None, "{} is None".format(
            cached_features_file)

        # if cased file exists, load file
        if os.path.exists(cached_features_file):
            print(f"Loading features from cached file {cached_features_file}")
            features = torch.load(cached_features_file,
                                  map_location=args.device.type)
            print(f"dataset is loaded to device: {args.device.type}")
            return features
        else:
            features = None
            if mode == "valid":
                examples = self.load_and_cache_examples_with_e1_e2(
                    args, mode="valid")
                features = self.load_valid_OR_test_features_with_e1_e2(examples)

            # endif
            if mode == "test":
                examples = self.load_and_cache_examples_with_e1_e2(
                    args, mode="test")
                features = self.load_valid_OR_test_features_with_e1_e2(examples)

            # endif
            assert features is not None

            # ---------------------------------------------------------------------------
            # ############### sort entry ids and reorder ################
            feature_dict = {}
            for fea in features:
                guid = fea.guid
                feature_dict[guid] = fea
            # endfor

            # ############## reorder ###############
            sorted_feature_list = []
            for guid, fea in sorted(feature_dict.items(), key=lambda x: x[0]):
                sorted_feature_list.append(fea)
            # endfor
            # ---------------------------------------------------------------------------

            print("Saving features into cached file {}".format(
                cached_features_file))
            torch.save(sorted_feature_list, cached_features_file)
            return sorted_feature_list

        # endif

    def load_and_cache_examples_from_database(self, args, mode=None):
        """
        h, t version
        not from database, from text
        
        Args:
            args (_type_): This args should come from main.py file, be consistent with main.py file's args
            mode (_type_, optional): valid or test. Defaults to None. Train data is loaded dynamically from database.

        Returns:
            _type_: _description_
        """
        assert mode in [
            "valid", "test"]  # train example is dynamically generated
        os.makedirs(args.valid_test_example_and_feature_folder, exist_ok=True)

        # ######## create examples #########
        examples = None
        cached_example_file_path = None
        if mode == "train":
            # cached_example_file_path = os.path.join(
            #     args.data_folder, f"{mode}_cached_example_each_relation_size_{args.data_tag}")
            print(">>>>>> should not load train data here.")
            sys.exit(0)
            pass
        else:
            cached_example_file_path = os.path.join(args.valid_test_example_and_feature_folder, f"{mode}_cached_example")
        # endif
        assert cached_example_file_path is not None

        # if cased file exists, load file
        if os.path.exists(cached_example_file_path):
            print(
                f"Loading examples from cached file: {cached_example_file_path}")
            with open(cached_example_file_path, mode="rb") as fin:
                examples = pickle.load(fin)
            # endwith
            print(f"{mode} dataset is loaded.")
            return examples
        else:
            if mode == "valid":
                validation_folder_nd_label_pair_list = [(self.valid_normal_folder, "normal"),
                                                        (self.valid_novel_folder, "novel")]
                examples = self.load_valid_OR_text_examples(
                    validation_folder_nd_label_pair_list)
            # endif
            if mode == "test":
                test_folder_nd_label_pair_list = [(self.test_normal_folder, "normal"),
                                                  (self.test_novel_folder, "novel")]
                examples = self.load_valid_OR_text_examples(
                    test_folder_nd_label_pair_list)
            # endif
            assert examples is not None

            # ---------------------------------------------------------------------------
            # ############### sort entry ids and reorder ################
            example_dict = {}
            for exp in examples:
                guid = exp.guid
                example_dict[guid] = exp
            # endfor

            # ############## reorder ###############
            sorted_example_list = []
            for guid, exp in sorted(example_dict.items(), key=lambda x: x[0]):
                sorted_example_list.append(exp)
            # endfor
            # ---------------------------------------------------------------------------

            print("Saving examples into cached file {}".format(
                cached_example_file_path))
            with open(cached_example_file_path, mode="wb") as fout:
                pickle.dump(sorted_example_list, fout)
            # endwith
            return sorted_example_list
        # endif

    def load_and_cache_features_from_database(self, args, mode=None):
        """
        for dev, test dataset
        h, t version
        
        not from database, from text
        """
        assert mode in [
            "valid", "test"]  # train example is dynamically generated
        feature_type = args.feature_type
        os.makedirs(
            args.valid_test_example_and_feature_folder, exist_ok=True)

        # if feature_type == "glove":
        #     assert model_class == None
        #     assert tokenizer_class == None
        # elif "bert" in feature_type:
        #     assert model_class is not None
        #     assert tokenizer_class is not None
        # # endif

        cached_features_file = None
        if args.feature_type == "bert":
            if mode == "train":
                # cached_features_file = os.path.join(args.data_folder,
                #                                     f"{mode}_cached_feature_{args.feature_type}_{args.data_tag}")
                print(">>>>>> should not load train data here.")
                sys.exit(0)
            else:
                cached_features_file = os.path.join(args.valid_test_example_and_feature_folder,
                                                    f"{mode}_cached_feature_{args.feature_type}")
            # endif
        # endif

        if args.feature_type == "glove":
            cached_features_file = os.path.join(args.valid_test_example_and_feature_folder,
                                                f"{mode}_cached_feature_{args.max_seq_length}_{args.feature_type}")

        assert cached_features_file is not None, "{} is None".format(
            cached_features_file)

        # if cased file exists, load file
        if os.path.exists(cached_features_file):
            print(f"Loading features from cached file {cached_features_file}")
            features = torch.load(cached_features_file,
                                  map_location=args.device.type)
            print(f"dataset is loaded to device: {args.device.type}")
            return features
        else:
            features = None
            if mode == "valid":
                examples = self.load_and_cache_examples_from_database(
                    args, mode="valid")
                features = self.load_valid_OR_test_features(examples)

            # endif
            if mode == "test":
                examples = self.load_and_cache_examples_from_database(
                    args, mode="test")
                features = self.load_valid_OR_test_features(examples)

            # endif
            assert features is not None

            # ---------------------------------------------------------------------------
            # ############### sort entry ids and reorder ################
            feature_dict = {}
            for fea in features:
                guid = fea.guid
                feature_dict[guid] = fea
            # endfor

            # ############## reorder ###############
            sorted_feature_list = []
            for guid, fea in sorted(feature_dict.items(), key=lambda x: x[0]):
                sorted_feature_list.append(fea)
            # endfor
            # ---------------------------------------------------------------------------

            print("Saving features into cached file {}".format(
                cached_features_file))
            torch.save(sorted_feature_list, cached_features_file)
            return sorted_feature_list

        # endif
    
    
    def check_whether_two_example_are_equal(self, exp_1, exp_2):
        try:
            assert exp_1.guid == exp_2.guid
            assert exp_1.nd_label_text == exp_2.nd_label_text
            assert exp_1.rc_label_text == exp_2.rc_label_text
            
            # #### check head entity obj ####
            exp_1.head_wikidata_entity_obj.wikidata_id == exp_2.head_wikidata_entity_obj.wikidata_id
            exp_1.head_wikidata_entity_obj.description == exp_2.head_wikidata_entity_obj.description
            exp_1.head_wikidata_entity_obj.label == exp_2.head_wikidata_entity_obj.label

            for i in range(len(exp_1.head_wikidata_entity_obj.property_to_value_list)):
                tmp_item_1 = exp_1.head_wikidata_entity_obj.property_to_value_list[i]
                tmp_item_2 = exp_2.head_wikidata_entity_obj.property_to_value_list[i]
                
                assert tmp_item_1[0] == tmp_item_2[0]
                assert " ".join(tmp_item_1[1]) == " ".join(tmp_item_2[1])
            #endfor
            
            # #### check tail entity obj ####
            exp_1.tail_wikidata_entity_obj.wikidata_id == exp_2.tail_wikidata_entity_obj.wikidata_id
            exp_1.tail_wikidata_entity_obj.label == exp_2.tail_wikidata_entity_obj.label
            exp_1.tail_wikidata_entity_obj.description == exp_2.tail_wikidata_entity_obj.description
            
            for j in range(len(exp_1.tail_wikidata_entity_obj.property_to_value_list)):
                tmp_item_1 = exp_1.tail_wikidata_entity_obj.property_to_value_list[j]
                tmp_item_2 = exp_2.tail_wikidata_entity_obj.property_to_value_list[j]
                
                assert tmp_item_1[0] == tmp_item_2[0]
                assert " ".join(tmp_item_1[1]) == " ".join(tmp_item_2[1])
            #endfor
            
            # #### check text_info ####
            exp_1_text_info = exp_1.text_info
            exp_2_text_info = exp_2.text_info
            
            assert exp_1_text_info["id"] == exp_2_text_info["id"]
            assert " ".join(exp_1_text_info["tokens"]) == " ".join(exp_2_text_info["tokens"])
            assert exp_1_text_info["h"][0] == exp_2_text_info["h"][0]
            assert exp_1_text_info["t"][1] == exp_2_text_info["t"][1]
            
            # >>> h
            for k in range(len(exp_1_text_info["h"][2])):
                for m in range(len(exp_1_text_info["h"][2][k])):
                    assert exp_1_text_info["h"][2][k][m] == exp_2_text_info["h"][2][k][m]
                #endfor
            #endfor
            
            # >>> t
            for k in range(len(exp_1_text_info["t"][2])):
                for m in range(len(exp_1_text_info["t"][2][k])):
                    assert exp_1_text_info["t"][2][k][m] == exp_2_text_info["t"][2][k][m]
                #endfor
            #endfor
                
            return True
        except:
            return False
        #end
    

    def get_instance_id_that_needs_create_features(self, current_example_list, reference_example_list):
        # convert list to dict
        current_example_dict = {}
        for cur_tmp_exp in current_example_list:
            current_example_dict[cur_tmp_exp.guid] = cur_tmp_exp
        #endfor
        
        reference_example_dict = {}
        for ref_tmp_exp in reference_example_list:
            reference_example_dict[ref_tmp_exp.guid] = ref_tmp_exp
        #endfor
        
        total_guid_set_needs_creating_feature = set()
        
        # (1) newly added instances
        newly_added_instance_guid_set = set(current_example_dict.keys()) - set(reference_example_dict.keys())
        print(f">>>>>>>>>> There are {len(newly_added_instance_guid_set)} newly added instances.")
        
        # (2) check modified instances
        modified_instance_guid_set = set()
        for k, v in tqdm(current_example_dict.items(), desc="comparing"):
            if k in newly_added_instance_guid_set:
                continue
            #endif
            
            cur_exp = v
            refer_exp = reference_example_dict[k]
            
            is_equal = self.check_whether_two_example_are_equal(cur_exp, refer_exp)
            if not is_equal:
                modified_instance_guid_set.add(k)
            #endif
        #endfor
        print(f">>>>>>>>>> There are {len(modified_instance_guid_set)} modified instances.")
        
        ## add all
        total_guid_set_needs_creating_feature.update(newly_added_instance_guid_set)
        total_guid_set_needs_creating_feature.update(modified_instance_guid_set)
        print(f">>>>>>>>>> There are {len(total_guid_set_needs_creating_feature)} guid in total need to create feature.")
        
        total_guid_list = list(total_guid_set_needs_creating_feature)
        total_exp_list = []
        for guid in total_guid_list:
            total_exp_list.append(current_example_dict[guid])
        #endfor
        
        return total_guid_list, total_exp_list
    
        
    def load_and_cache_features_from_database_based_on_existing_dump(self,
                                                                     args,
                                                                     current_example_list,
                                                                     mode=None
                                                                     ):
        """
        DO not create features for every instances. 
        Only create features, when:
        
        (1) The example is modified
        (2) There is new instance added to example
        """
        assert mode in {"valid", "test"}
        
        cached_features_file = os.path.join(args.valid_test_example_and_feature_folder,
                                                    f"{mode}_cached_feature_{args.feature_type}")
        
        if not os.path.exists(cached_features_file):
        
            # load example reference
            reference_example_file_path = os.path.join(args.valid_test_example_and_feature_reference_folder, f"{mode}_cached_example")
            assert os.path.exists(reference_example_file_path)
            with open(reference_example_file_path, mode="rb") as fin:
                reference_example_list = pickle.load(fin)
            # endwith
            print(f"{mode} reference example is loaded.")
            
            # compare reference and current_example see which example instance are (1) modified (2) newly added
            new_guid_list, new_exp_list = self.get_instance_id_that_needs_create_features(current_example_list, reference_example_list)
            new_feature_list = self.load_valid_OR_test_features(new_exp_list)
            new_guid_to_feature_dict = dict(zip(new_guid_list, new_feature_list))
                
            # load feature reference
            reference_feature_file_path = os.path.join(args.valid_test_example_and_feature_reference_folder, f"{mode}_cached_feature_{args.feature_type}")
            assert os.path.exists(reference_feature_file_path)
            print(f"loading {mode} reference feature list...")
            reference_features_list = torch.load(reference_feature_file_path, map_location=args.device.type)
            print(f"{mode} reference feature is loaded to device: {args.device.type}")
            print("DONE.")
            
            # get total feature dictionary
            total_feature_dict = {}
            total_feature_dict.update(new_guid_to_feature_dict)
            for tmp_fea in reference_features_list:
                total_feature_dict[tmp_fea.guid] = tmp_fea
            #endfor
            
            print(f"total feature (new + reference) num: {len(total_feature_dict)}, cur example num: {len(current_example_list)}")
            assert len(total_feature_dict) >= len(current_example_list)
            
            # get current_example_list guid list
            current_feature_list = []
            for tmp_exp in current_example_list:
                cur_guid = tmp_exp.guid
                current_feature_list.append(total_feature_dict[cur_guid])
            #endfor
            assert len(current_feature_list) == len(current_example_list)
        
            #### dump feature #####
            print("Saving features into cached file {}".format(cached_features_file))
            torch.save(current_feature_list, cached_features_file)
            return current_feature_list
        else:
            print(f"Loading features from cached file {cached_features_file}")
            features = torch.load(cached_features_file,
                                  map_location=args.device.type)
            print(f"dataset is loaded to device: {args.device.type}")
            return features
        
        

    def _add_label_description_to_property_to_value_list(self, wikidata_example_json_obj):
        """
        # 1. sorted
        # 2. Note: to make it consistent, if value_text_list is not a list, convert it to list
        # 3. filter out property id that not in candidate property id list
        # 4. either it is str or list, if it is 0, skip

        Args:
            wikidata_example_json_obj (_type_): _description_

        Returns:
            _type_: _description_
        """
        label = wikidata_example_json_obj["label"]
        description = wikidata_example_json_obj["description"]
        # this is not sorted
        property_to_value_list = wikidata_example_json_obj["property_to_value_list"]

        # add label and description
        if len(label) > 0:
            property_to_value_list.append(["P-1", [label]])
        # endif

        if len(description) > 0:
            property_to_value_list.append(["P-2", [description]])
        # endif

        # 1. sorted
        # 2. Note: to make it consistent, if value_text_list is not a list, convert it to list
        # 3. filter out property id that not in candidate property id list
        SORTED_property_to_value_list = []
        for property_id, value_text_list in sorted(property_to_value_list, key=lambda x: int(x[0].replace("P", ""))):

            if len(value_text_list) == 0:  # either it is str or list, if it is 0, skip
                continue
            # endif

            if not isinstance(value_text_list, list):
                value_text_list = [value_text_list]
            # endif

            if property_id not in self.property_str_to_index_dict:
                continue
            # endif

            SORTED_property_to_value_list.append(
                [property_id, value_text_list])
        # endfor

        return SORTED_property_to_value_list

    def _single_worker_load_example_from_file_path_list(self, file_path_and_nd_label_and_rc_label_list):

        total_example_list = []

        for file_path, nd_label, rc_label in file_path_and_nd_label_and_rc_label_list:

            # >>>>>>> process single file >>>>>
            block_list = []
            with open(file_path, mode="r") as fin:

                for line in fin:
                    line = line.strip()

                    if len(line) == 0:

                        fewrel_json_obj = json.loads(block_list[0])
                        head_json_obj = json.loads(block_list[1])
                        tail_json_obj = json.loads(block_list[2])

                        # add label and description to the property_to_value_list
                        head_property_to_value_list = self._add_label_description_to_property_to_value_list(
                            head_json_obj)
                        tail_property_to_value_list = self._add_label_description_to_property_to_value_list(
                            tail_json_obj)

                        # guid is already globally unique
                        guid = fewrel_json_obj['id']

                        # create example obj
                        # wikidata_id, label, description, property_to_value_list
                        head_wikidata_entity = WikidataEntityExample(wikidata_id=head_json_obj["id"],
                                                                     label=head_json_obj["label"],
                                                                     description=head_json_obj["description"],
                                                                     property_to_value_list=head_property_to_value_list)

                        tail_wikidata_entity = WikidataEntityExample(wikidata_id=tail_json_obj["id"],
                                                                     label=tail_json_obj["label"],
                                                                     description=tail_json_obj["description"],
                                                                     property_to_value_list=tail_property_to_value_list)

                        wikidata_entity_pair = WikidataEntityPairRelationExample(guid=guid,
                                                                                 text_info=fewrel_json_obj,
                                                                                 head_wikidata_entity_obj=head_wikidata_entity,
                                                                                 tail_wikidata_entity_obj=tail_wikidata_entity,
                                                                                 rc_label_text=rc_label,
                                                                                 nd_label_text=nd_label)

                        total_example_list.append(wikidata_entity_pair)

                        # reset block
                        block_list = []
                    else:
                        block_list.append(line)
                    # endif
                # endfor
            # endwith
        # endfor

        return total_example_list
    
    
    def load_valid_OR_test_with_e12_examples_from_folder(self, folder_list):
        
        total_example_list = []

        for folder in folder_list:
            
            for root, subdir, file_name_list in os.walk(folder):
                
                for file_name in file_name_list:
                    file_path = os.path.join(root, file_name)
                    
                    # >>>>>>> process single file >>>>>
                    block_list = []
                    with open(file_path, mode="r") as fin:

                        for line in fin:
                            line = line.strip()

                            if len(line) == 0:

                                fewrel_json_obj = json.loads(block_list[0])
                                e1_json_obj = json.loads(block_list[1])
                                e2_json_obj = json.loads(block_list[2])
                                
                                rc_label = fewrel_json_obj["rc_label"]
                                nd_label = fewrel_json_obj["nd_label"]

                                # add label and description to the property_to_value_list
                                e1_property_to_value_list = self._add_label_description_to_property_to_value_list(e1_json_obj)
                                e2_property_to_value_list = self._add_label_description_to_property_to_value_list(e2_json_obj)

                                # guid is already globally unique
                                guid = fewrel_json_obj['id']

                                # create example obj
                                # wikidata_id, label, description, property_to_value_list
                                e1_wikidata_entity = WikidataEntityExample(wikidata_id=e1_json_obj["id"],
                                                                            label=e1_json_obj["label"],
                                                                            description=e1_json_obj["description"],
                                                                            property_to_value_list=e1_property_to_value_list)

                                e2_wikidata_entity = WikidataEntityExample(wikidata_id=e2_json_obj["id"],
                                                                            label=e2_json_obj["label"],
                                                                            description=e2_json_obj["description"],
                                                                            property_to_value_list=e2_property_to_value_list)

                                wikidata_entity_pair = WikidataEntityPairRelationExample_E1E2(guid=guid,
                                                                                              text_info=fewrel_json_obj,
                                                                                              e1_wikidata_entity_obj=e1_wikidata_entity,
                                                                                              e2_wikidata_entity_obj=e2_wikidata_entity,
                                                                                              rc_label_text=rc_label,
                                                                                              nd_label_text=nd_label)

                                total_example_list.append(wikidata_entity_pair)

                                # reset block
                                block_list = []
                            else:
                                block_list.append(line)
                            # endif
                        # endfor
                    # endwith
                # endfor

        return total_example_list
    

    def load_valid_OR_text_examples(self, folder_nd_label_pair_list):
        """
        folder_list = [(self.valid_normal_folder, "normal"),
                       (self.valid_novel_folder, "novel")]

        Load validation examples for all relations.
        Avoid loading knowledge leaked attribute information.

        such as "inventor" and "notable work" are inverse relation.
        If the training data contains such information, then the model might overfit this dataset,
        rather than learn more useful knowledge from the data
        Returns:

        """
        # ######### The validation data and test data are in ThreeLine format ========
        # The data are already processed into wikidata example format, which can be loaded directly

        total_file_path_and_nb_label_and_rc_label_list = []

        for folder, nd_label in folder_nd_label_pair_list:
            print(f"{nd_label}")
            for root, subdir, file_name_list in os.walk(folder):
                for file_name in file_name_list:
                    relation_id = file_name.replace(".txt", "")
                    rc_label = relation_id

                    if relation_id not in self.evaluation_rc_str_list:
                        continue
                    # endif

                    file_path = os.path.join(root, file_name)

                    entry = (file_path, nd_label, rc_label)
                    total_file_path_and_nb_label_and_rc_label_list.append(
                        entry)
                # endfor
            # endfor
        # endfor

        # ######### process ThreeLine format file and load examples ###########

        # ######## single process #######
        all_example_list = self._single_worker_load_example_from_file_path_list(
            total_file_path_and_nb_label_and_rc_label_list)
        return all_example_list

    def _convert_property_to_value_text_list_to_feature(self, property_to_value_list):

        sorted_property_id_list = []
        value_embed_list = []

        for property_id, value_text_list in sorted(property_to_value_list, key=lambda x: int(x[0].replace("P", ""))):

            if len(value_text_list) == 0:
                value_text_list = [""]

            assert isinstance(value_text_list, list)

            # --- value embed ---
            text_vector_list = self.get_bert_embed(value_text_list)

            # get mean of the vector, when len(text_vector_list) == 1, it is OK
            arr = np.array(text_vector_list)
            mean_vector = np.mean(arr, axis=0)

            sorted_property_id_list.append(property_id)
            value_embed_list.append(mean_vector.tolist())
        # endfor

        return sorted_property_id_list, value_embed_list

    def _single_worker_create_features_from_example_list(self, examples):
        """
        head_wikidata_entity = WikidataEntityExample(wikidata_id=head_json_obj["id"],
                                                    label=head_json_obj["label"],
                                                    description=head_json_obj["description"],
                                                    property_to_value_list=head_json_obj["property_to_value_list"])

        tail_wikidata_entity = WikidataEntityExample(wikidata_id=tail_json_obj["id"],
                                                        label=tail_json_obj["label"],
                                                        description=tail_json_obj["description"],
                                                        property_to_value_list=tail_json_obj["property_to_value_list"])


        wikidata_entity_pair = WikidataEntityPairRelationExample(guid=guid,
                                                                head_wikidata_entity_obj=head_wikidata_entity,
                                                                tail_wikidata_entity_obj=tail_wikidata_entity,
                                                                rc_label_text=rc_label,
                                                                nd_label_text=nd_label)

        ---->>>


        class WikidataEntityPairFeature:
            def __init__(self,
                    guid,
                    head_wikidata_id, 
                    tail_wikidata_id,
                    head_property_id_list, 
                    tail_property_id_list,
                    head_property_value_embed_list, 
                    tail_property_value_embed_list,
                    rc_label_index,
                    nd_label_index):
            self.guid = guid
            self.head_wikidata_id = head_wikidata_id
            self.tail_wikidata_id = tail_wikidata_id
            self.head_property_id_list = head_property_id_list
            self.tail_property_id_list = tail_property_id_list
            self.head_property_value_embed_list = head_property_value_embed_list
            self.tail_property_value_embed_list = tail_property_value_embed_list
            self.rc_label_index = rc_label_index
            self.nd_label_index = nd_label_index
        """

        total_feature_list = []

        for index, exp in enumerate(tqdm(examples, desc="all examples")):

            # if index == 50:
            #     print("hello")

            guid = exp.guid
            rc_label_text = exp.rc_label_text
            nd_label_text = exp.nd_label_text

            # head
            head_wikidata_entity_obj = exp.head_wikidata_entity_obj
            head_wikidata_id = head_wikidata_entity_obj.wikidata_id
            head_property_to_value_list = head_wikidata_entity_obj.property_to_value_list
            head_property_id_list, head_value_embed_list = self._convert_property_to_value_text_list_to_feature(
                head_property_to_value_list)
            assert len(head_property_id_list) == len(head_value_embed_list)

            # tail
            tail_wikidata_entity_obj = exp.tail_wikidata_entity_obj
            tail_wikidata_id = tail_wikidata_entity_obj.wikidata_id
            tail_property_to_value_list = tail_wikidata_entity_obj.property_to_value_list
            tail_property_id_list, tail_value_embed_list = self._convert_property_to_value_text_list_to_feature(
                tail_property_to_value_list)
            assert len(tail_property_id_list) == len(tail_value_embed_list)

            # create feature obj
            triple_feature = WikidataEntityPairFeature(guid=guid,
                                                       head_wikidata_id=head_wikidata_id,
                                                       tail_wikidata_id=tail_wikidata_id,
                                                       head_property_str_list=head_property_id_list,
                                                       tail_property_str_list=tail_property_id_list,
                                                       head_value_embed_list=head_value_embed_list,
                                                       tail_value_embed_list=tail_value_embed_list,
                                                       rc_label_text=rc_label_text,
                                                       nd_label_text=nd_label_text)

            total_feature_list.append(triple_feature)
        # endfor

        return total_feature_list

    def load_valid_OR_test_features(self, examples):
        """
        Args:
            folder_nd_label_pair_list (_type_): _description_
        """

        # ######## single process #######
        all_feature_list = self._single_worker_create_features_from_example_list(
            examples)

        return all_feature_list
    
    
    def load_valid_OR_test_features_with_e1_e2(self, examples):
        
        total_feature_list = []

        for index, exp in enumerate(tqdm(examples, desc="all examples")):

            # if index == 50:
            #     print("hello")

            guid = exp.guid
            rc_label_text = exp.rc_label_text
            nd_label_text = exp.nd_label_text

            # e1
            e1_wikidata_entity_obj = exp.e1_wikidata_entity_obj
            e1_wikidata_id = e1_wikidata_entity_obj.wikidata_id
            e1_property_to_value_list = e1_wikidata_entity_obj.property_to_value_list
            e1_property_id_list, e1_value_embed_list = self._convert_property_to_value_text_list_to_feature(
                e1_property_to_value_list)
            assert len(e1_property_id_list) == len(e1_value_embed_list)

            # tail
            e2_wikidata_entity_obj = exp.e2_wikidata_entity_obj
            e2_wikidata_id = e2_wikidata_entity_obj.wikidata_id
            e2_property_to_value_list = e2_wikidata_entity_obj.property_to_value_list
            e2_property_id_list, e2_value_embed_list = self._convert_property_to_value_text_list_to_feature(
                e2_property_to_value_list)
            assert len(e2_property_id_list) == len(e2_value_embed_list)

            # create feature obj
            triple_feature = WikidataEntityPairFeature_E1E2(guid=guid,
                                                            e1_wikidata_id=e1_wikidata_id,
                                                            e2_wikidata_id=e2_wikidata_id,
                                                            e1_property_str_list=e1_property_id_list,
                                                            e2_property_str_list=e2_property_id_list,
                                                            e1_value_embed_list=e1_value_embed_list,
                                                            e2_value_embed_list=e2_value_embed_list,
                                                            rc_label_text=rc_label_text,
                                                            nd_label_text=nd_label_text)

            total_feature_list.append(triple_feature)
        # endfor

        return total_feature_list
        
    
    def validate_check_guid_list(self, example_list, feature_list, mode=None):
        
        assert mode in {"valid", "test"}
        
        example_guid_list = []
        feature_guid_list = []
        
        for i in range(len(example_list)):
            example_guid_list.append(example_list[i].guid)
        #endfor
        
        for i in range(len(feature_list)):
            feature_guid_list.append(feature_list[i].guid)
        #endfor
        
        assert example_guid_list == feature_guid_list
        
        print(f"{mode} DONE.")
        pass
    

    def validate_consistency_of_cached_examples_and_features(self, example_list, feature_list):
        """
        June 15, 2022

        All train, valid, test dataset are loaded and processed using python multiprocessing.
        The order of the entries is not consistent.
        Before save object to cache files, we sorted the entries based on the guid.

        We need to validate:
        1. train / valid / test dataset the order of guid are consistent
        2. make sure that all property: value pair, the properties are consistent
        """
        error_tag = False

        assert example_list is not None and feature_list is not None
        assert len(example_list) == len(feature_list)

        property_not_cosistenty_with_value_embed_count = 0

        # ######## validate examples and features ###########
        for i in tqdm(range(len(example_list))):

            cur_exp = example_list[i]
            cur_fea = feature_list[i]

            # ------------- validate id ------------
            exp_id = cur_exp.guid
            fea_id = cur_fea.guid
            
            # if exp_id == 'P61-0-test-s-823f561b-2531-4b32-ab3e-a3491d95c83f':
            #     print("hello")

            assert exp_id == fea_id

            # ------------- validate label ------------
            exp_rc_label = cur_exp.rc_label_text
            exp_nd_label = cur_exp.nd_label_text

            fea_rc_label = cur_fea.rc_label_text
            fea_nd_label = cur_fea.nd_label_text

            assert exp_rc_label == fea_rc_label
            assert exp_nd_label == fea_nd_label

            # --------- validate relation:value dict ---------
            # example
            exp_head_keys = set([property_str for property_str,
                                value_list in cur_exp.head_wikidata_entity_obj.property_to_value_list])
            exp_tail_keys = set([property_str for property_str,
                                value_list in cur_exp.tail_wikidata_entity_obj.property_to_value_list])

            # features
            fea_head_keys_id = set(cur_fea.head_property_str_list)
            fea_tail_keys_id = set(cur_fea.tail_property_str_list)

            try:
                assert exp_head_keys == fea_head_keys_id
                assert exp_tail_keys == fea_tail_keys_id
            except:
                error_tag = True
                
                with open("DATASET_validation_error_log.txt", mode="a") as fout:
                    print(exp_id)
                    print(exp_head_keys)
                    print(fea_head_keys_id)
                    print()
                    print(exp_tail_keys)
                    print(fea_tail_keys_id)
                    
                    fout.write(f"exp_id: {exp_id}\n")
                    fout.write(f"exp_head_keys: {exp_head_keys}\n")
                    fout.write(f"fea_head_keys_ids: {fea_head_keys_id}\n")
                    fout.write("-----\n")
                    fout.write(f"exp_tail_keys: {exp_tail_keys}\n")
                    fout.write(f"fea_tail_keys_id: {fea_tail_keys_id}\n\n\n")
                #endwith
            # endtry

            # head
            try:
                assert len(cur_fea.head_property_str_list) == len(
                    cur_fea.head_value_embed_list)
                assert len(cur_fea.tail_property_str_list) == len(
                    cur_fea.tail_value_embed_list)
            except:
                error_tag = True
                
                property_not_cosistenty_with_value_embed_count += 1
                print(f"{exp_id} property str list size != value_embed_list size")
                
                with open("DATASET_validation_error_log.txt", mode="a") as fout:
                    fout.write(f"{exp_id} property str list size != value_embed_list size \n\n\n")
                #endwith
                
            # endtry

        # endfor

        print(
            f"property_not_cosistenty_with_value_embed_count = {property_not_cosistenty_with_value_embed_count}")
        print("DONE.")
        
        if error_tag:
            print("Please check file - [DATASET_validation_error_log.txt] for details.")
        else:
            print("passed validation test.")
        #endif
        
        pass

    @DeprecationWarning
    def FIX_ISSUE_validate_consistency_of_cached_examples_and_features(self, args, example_list, feature_list, mode=None):
        """
        June 15, 2022

        All train, valid, test dataset are loaded and processed using python multiprocessing.
        The order of the entries is not consistent.
        Before save object to cache files, we sorted the entries based on the guid.

        We need to validate:
        1. train / valid / test dataset the order of guid are consistent
        2. make sure that all property: value pair, the properties are consistent
        """
        assert mode in {"valid", "test"}

        assert example_list is not None and feature_list is not None
        assert len(example_list) == len(feature_list)

        property_not_cosistenty_with_value_embed_count = 0

        # ######## validate examples and features ###########
        for i in tqdm(range(len(example_list))):

            cur_exp = example_list[i]
            cur_fea = feature_list[i]

            # ------------- validate id ------------
            exp_id = cur_exp.guid
            fea_id = cur_fea.guid
            assert exp_id == fea_id

            # ------------- validate label ------------
            exp_rc_label = cur_exp.rc_label_text
            exp_nd_label = cur_exp.nd_label_text

            fea_rc_label = cur_fea.rc_label_text
            fea_nd_label = cur_fea.nd_label_text

            assert exp_rc_label == fea_rc_label
            assert exp_nd_label == fea_nd_label

            # --------- validate relation:value dict ---------
            # example
            exp_head_keys = set([property_str for property_str,
                                value_list in cur_exp.head_wikidata_entity_obj.property_to_value_list])
            exp_tail_keys = set([property_str for property_str,
                                value_list in cur_exp.tail_wikidata_entity_obj.property_to_value_list])

            # features
            fea_head_keys_id = set(cur_fea.head_property_str_list)
            fea_tail_keys_id = set(cur_fea.tail_property_str_list)

            try:
                assert exp_head_keys == fea_head_keys_id
                assert exp_tail_keys == fea_tail_keys_id
            except:
                print(exp_id)
                print(exp_head_keys)
                print(fea_head_keys_id)
                print()
                print(exp_tail_keys)
                print(fea_tail_keys_id)
            # endtry

            # head
            try:
                assert len(cur_fea.head_property_str_list) == len(
                    cur_fea.head_value_embed_list)
                assert len(cur_fea.tail_property_str_list) == len(
                    cur_fea.tail_value_embed_list)
            except:
                property_not_cosistenty_with_value_embed_count += 1
                print(f"{exp_id} property str list size != value_embed_list size")

                print("create new features")
                new_fea = self._single_worker_create_features_from_example_list([
                                                                                cur_exp])

                feature_list[i] = new_fea[0]  # it is a list
            # endtry

        # endfor

        fea_output_file_path = os.path.join(
            args.valid_test_example_and_feature_folder, f"{mode}_new_feature_cache")
        print("Saving features into cached file {}".format(
            fea_output_file_path))
        torch.save(feature_list, fea_output_file_path)

        print(
            f"property_not_cosistenty_with_value_embed_count = {property_not_cosistenty_with_value_embed_count}")
        print("DONE.")
        pass

    @DeprecationWarning
    def clean_property_id_in_examples_and_features(self, args, example_list, feature_list, property_str_to_num_dict, mode=None):
        """
        An candidate list of property id is maintained, discard other properties in examples and features
        """
        assert mode in {"valid", "test"}

        def clean_single_exp(exp, property_str_to_num_dict):
            """
            clean example obj

            Args:
                exp (_type_): _description_
                property_str_to_num_dict (_type_): _description_
            """

            def clean_wikidata_entity_obj(entity_obj, property_str_to_num_dict):
                property_to_value_list = entity_obj.property_to_value_list

                new_property_to_value_list = []
                for tmp_property, tmp_value_text_list in property_to_value_list:
                    if tmp_property not in property_str_to_num_dict:
                        continue
                    # endif
                    new_property_to_value_list.append(
                        [tmp_property, tmp_value_text_list])
                # endfor

                entity_obj.property_to_value_list = new_property_to_value_list

                return entity_obj
            # enddef

            # ### head ###
            head_wikidata_entity_obj = exp.head_wikidata_entity_obj
            CLEAN_head_wikidata_entity_obj = clean_wikidata_entity_obj(
                head_wikidata_entity_obj, property_str_to_num_dict)

            # ### tail ###
            tail_wikidata_entity_obj = exp.tail_wikidata_entity_obj
            CLEAN_tail_wikidata_entity_obj = clean_wikidata_entity_obj(
                tail_wikidata_entity_obj, property_str_to_num_dict)

            # ### assign to original exp obj
            exp.head_wikidata_entity_obj = CLEAN_head_wikidata_entity_obj
            exp.tail_wikidata_entity_obj = CLEAN_tail_wikidata_entity_obj

            return exp
        # enddef

        def clean_single_fea(fea, property_str_to_num_dict):
            """
            clean feature obj

            Args:
                fea (_type_): _description_
                property_str_to_num_dict (_type_): _description_
            """

            def clean_property_str_and_value_embed_together(property_str_list, value_embed_list, property_str_to_num_dict):
                """
                property_str_list and value_embed_list are aligned together
                Args:
                    property_str_list (_type_): _description_
                    value_embed_list (_type_): _description_
                """
                CLEAN_property_str_list = []
                CLEAN_value_embed_list = []

                assert len(property_str_list) == len(value_embed_list)
                for i in range(len(property_str_list)):
                    if property_str_list[i] not in property_str_to_num_dict:
                        continue
                    # endif

                    CLEAN_property_str_list.append(property_str_list[i])
                    CLEAN_value_embed_list.append(value_embed_list[i])
                # endfor

                return CLEAN_property_str_list, CLEAN_value_embed_list
            # enddef

            # #### head ####
            head_property_str_list, head_value_embed_list = fea.head_property_str_list, fea.head_value_embed_list
            CLEAN_head_property_str_list, CLEAN_head_value_embed_list = clean_property_str_and_value_embed_together(head_property_str_list,
                                                                                                                    head_value_embed_list,
                                                                                                                    property_str_to_num_dict
                                                                                                                    )
            fea.head_property_str_list = CLEAN_head_property_str_list
            fea.head_value_embed_list = CLEAN_head_value_embed_list

            # #### tail ####
            tail_property_str_list, tail_value_embed_list = fea.tail_property_str_list, fea.tail_value_embed_list
            CLEAN_tail_property_str_list, CLEAN_tail_value_embed_list = clean_property_str_and_value_embed_together(tail_property_str_list,
                                                                                                                    tail_value_embed_list,
                                                                                                                    property_str_to_num_dict)
            fea.tail_property_str_list = CLEAN_tail_property_str_list
            fea.tail_value_embed_list = CLEAN_tail_value_embed_list

            return fea

        # ######## clean exp list #########

        CLEAN_example_list = []
        for exp in tqdm(example_list, desc="all example"):
            clean_exp = clean_single_exp(exp, property_str_to_num_dict)
            CLEAN_example_list.append(clean_exp)
        # endfor

        # ####### clean fea list ########
        CLEAN_feature_list = []
        for fea in tqdm(feature_list, desc="all feature"):
            clean_fea = clean_single_fea(fea, property_str_to_num_dict)
            CLEAN_feature_list.append(clean_fea)
        # endfor

        # ########## output #########

        # ## output example list ##
        output_example_path = os.path.join(
            args.valid_test_example_and_feature_folder, f"new_{mode}_example_list")
        print("Saving features into cached file {}".format(output_example_path))
        with open(output_example_path, mode="wb") as fout:
            pickle.dump(CLEAN_example_list, fout)
        # endwith

        # ## output feature list ##
        output_feature_path = os.path.join(
            args.valid_test_example_and_feature_folder, f"new_{mode}_feature_list")
        print("Saving features into cached file {}".format(output_feature_path))
        torch.save(CLEAN_feature_list, output_feature_path)

        pass

    def _single_worker_load_relation_to_guid_list_dict(self, relation_id, label):
        db = self.connect_to_wikidata_triple_feature_database()

        guid_list = []
        for item in db[f"{relation_id}_{label}"].find({}):
            guid = item["guid"]
            guid_list.append(guid)
        # endfor
        print(f"{relation_id}_{label} training data guid list is loaded.")

        return guid_list, relation_id, label

    def parallel_load_relation_train_data_guid_dict(self, args):
        """

        Args:
            args (_type_): _description_

        Returns:
            _type_: _description_
        """

        FILEPATH = os.path.join(args.train_tmp_folder,
                                "all_relation_id_to_guid_list_dict_db.json")

        if os.path.exists(FILEPATH):
            with open(FILEPATH, mode="r") as fin:
                all_relation_id_to_guid_list_dict = json.load(fin)
            # endwith
            # training relation should be the same as evalation relation.
            discard_relation_id_set = set(
                all_relation_id_to_guid_list_dict.keys()) - set(self.train_rc_str_list)
            print(f"DISCARD relation id: {discard_relation_id_set}")
            # not necessarily remote the relation that are discarded. Train them all. And then decide later.
            for tmp_id in discard_relation_id_set:
                del all_relation_id_to_guid_list_dict[tmp_id]
            # endfor

            return all_relation_id_to_guid_list_dict
        # endif

        begin = time.time()

        all_relation_id_to_guid_list_dict = {}
        all_cases_list = []
        for relation_id in self.train_rc_str_list:
            all_relation_id_to_guid_list_dict[relation_id] = {
                "normal": [], "novel": []}
            all_cases_list.append([relation_id, "normal"])
            all_cases_list.append([relation_id, "novel"])
        # endfor

        # ######## paprallel #########
        num_of_worker = os.cpu_count() - 2
        pool = Pool(processes=num_of_worker)

        job_list = []
        for case in all_cases_list:
            single_job = pool.apply_async(func=self._single_worker_load_relation_to_guid_list_dict,
                                          args=(case[0], case[1]))
            job_list.append(single_job)
        # endfor

        for job in job_list:
            guid_list, relation_id, label = job.get()
            all_relation_id_to_guid_list_dict[relation_id][label] = list(
                set(guid_list))
        # endfor

        length = time.time() - begin
        print(f"TIME: {length / 60} mins")

        with open(FILEPATH, mode="w") as fout:
            json.dump(all_relation_id_to_guid_list_dict, fout)
        # endwith

        return all_relation_id_to_guid_list_dict

    def write_train_data_stats(self, all_relation_id_to_guid_list_dict, output_file_path):
        with open(output_file_path, mode="w") as fout:
            csv_writer = csv.DictWriter(
                fout, fieldnames=["relation_id", "normal", "novel"])
            csv_writer.writeheader()

            for k, v in all_relation_id_to_guid_list_dict.items():
                entry = {"relation_id": k, "normal": len(
                    v["normal"]), "novel": len(v["novel"])}
                csv_writer.writerow(entry)
            # endfor
        # endwith
        pass

    # ###################### load training feature in batch #####################
    def get_model_input_with_e1_e2(self, feature_obj_list):
        """
        Oct 18, 2022
        # TODO: to finish
        feature_obj_list -> model input
        
        # Note
        In the feature_obj_list, the feature object is in class WikidataEntityPairFeature_E1E2
        If it is non-reversed relation, head -> e1, tail -> e2
        If it is reversed relation, head -> e2, tail -> e1
        Since the trained model is based on head and tail. 
        So, in evaluation session, the order of e1 and e2 should be adjusted to match head and tail
        """
        
        head_property_id_list_of_list = []
        tail_property_id_list_of_list = []
        head_property_value_embed_matrix_list_of_list = []
        tail_property_value_embed_matrix_list_of_list = []

        rc_label_id_list = []  # based on rc_str_to_id_dict
        nd_label_id_list = []

        for feature_obj in feature_obj_list:
            
            # rc_label_text
            rc_label_text = feature_obj.rc_label_text
            is_reversed = None
            if "@rev" in rc_label_text:
                is_reversed = True
            else:
                is_reversed = False
            #endif
            original_rc_label_text = rc_label_text.replace("@rev", "")
            
            nd_label_index = self.nd_label_to_num_dict[feature_obj.nd_label_text]
            rc_label_index = self.train_rc_str_to_num_dict[original_rc_label_text]

            nd_label_id_list.append(nd_label_index)
            # this is with @rev removed, after head and tail is aligned well, it is OK
            rc_label_id_list.append(rc_label_index)  
            
            # >>>>>>> it could happend that some property str are not in self.property_str_to_index_dict >>>>>>>
            # >>>>>>> This kind of property str needs to be discarde >>>>> such as P443

            # Note
            # Based on is_reversed, align e1, e2 with head, tail
            assert is_reversed is not None
            
            HEAD_PROPERTY_STR_LIST = None
            TAIL_PROPERTY_STR_LIST = None
            
            HEAD_VALUE_EMBED_LIST = None
            TAIL_VALUE_EMBED_LIST = None
            if is_reversed:
                # reversed
                HEAD_PROPERTY_STR_LIST = feature_obj.e2_property_str_list
                TAIL_PROPERTY_STR_LIST = feature_obj.e1_property_str_list
                
                # reversed
                HEAD_VALUE_EMBED_LIST = feature_obj.e2_value_embed_list
                TAIL_VALUE_EMBED_LIST = feature_obj.e1_value_embed_list
            else:
                HEAD_PROPERTY_STR_LIST = feature_obj.e1_property_str_list
                TAIL_PROPERTY_STR_LIST = feature_obj.e2_property_str_list
                
                HEAD_VALUE_EMBED_LIST = feature_obj.e1_value_embed_list
                TAIL_VALUE_EMBED_LIST = feature_obj.e2_value_embed_list
            #endif
            
            assert HEAD_PROPERTY_STR_LIST is not None
            assert TAIL_PROPERTY_STR_LIST is not None
            assert HEAD_VALUE_EMBED_LIST is not None
            assert TAIL_VALUE_EMBED_LIST is not None
            
            # HEAD
            head_property_str_list = HEAD_PROPERTY_STR_LIST
            head_property_index_list = []
            head_property_value_embed_list = []
            for index, property_str in enumerate(head_property_str_list):
                if property_str not in self.property_str_to_index_dict:
                    continue
                # endif
                head_property_index_list.append(
                    self.property_str_to_index_dict[property_str])
                # head_property_value_embed_list.append(feature_obj.head_value_embed_list[index])
                head_property_value_embed_list.append(HEAD_VALUE_EMBED_LIST[index])
            # endfor
            assert len(head_property_index_list) == len(
                head_property_value_embed_list)

            # TAIL
            tail_property_str_list = TAIL_PROPERTY_STR_LIST
            tail_property_index_list = []
            tail_property_value_embed_list = []
            for index, property_str in enumerate(tail_property_str_list):
                if property_str not in self.property_str_to_index_dict:
                    continue
                # endif
                tail_property_index_list.append(
                    self.property_str_to_index_dict[property_str])
                # tail_property_value_embed_list.append(feature_obj.tail_value_embed_list[index])
                tail_property_value_embed_list.append(TAIL_VALUE_EMBED_LIST[index])
            # endfor
            
            assert len(tail_property_index_list) == len(
                tail_property_value_embed_list)

            
            head_property_id_list_of_list.append(head_property_index_list)
            tail_property_id_list_of_list.append(tail_property_index_list)

            head_property_value_embed_matrix_list_of_list.append(
                head_property_value_embed_list)
            tail_property_value_embed_matrix_list_of_list.append(
                tail_property_value_embed_list)
        # endfor

        return head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
            tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
            nd_label_id_list, rc_label_id_list
    
    

    def get_model_input(self, feature_obj_list):
        """
        feature_obj_list -> model input

        June 15, 2022
        """
        head_property_id_list_of_list = []
        tail_property_id_list_of_list = []
        head_property_value_embed_matrix_list_of_list = []
        tail_property_value_embed_matrix_list_of_list = []

        rc_label_id_list = []  # based on rc_str_to_id_dict
        nd_label_id_list = []

        for feature_obj in feature_obj_list:

            # >>>>>>> it could happend that some property str are not in self.property_str_to_index_dict >>>>>>>
            # >>>>>>> This kind of property str needs to be discarde >>>>> such as P443

            # HEAD
            head_property_str_list = feature_obj.head_property_str_list
            head_property_index_list = []
            head_property_value_embed_list = []
            for index, property_str in enumerate(head_property_str_list):
                if property_str not in self.property_str_to_index_dict:
                    continue
                # endif
                head_property_index_list.append(
                    self.property_str_to_index_dict[property_str])
                head_property_value_embed_list.append(
                    feature_obj.head_value_embed_list[index])
            # endfor
            assert len(head_property_index_list) == len(
                head_property_value_embed_list)

            # TAIL
            tail_property_str_list = feature_obj.tail_property_str_list
            tail_property_index_list = []
            tail_property_value_embed_list = []
            for index, property_str in enumerate(tail_property_str_list):
                if property_str not in self.property_str_to_index_dict:
                    continue
                # endif
                tail_property_index_list.append(
                    self.property_str_to_index_dict[property_str])
                tail_property_value_embed_list.append(
                    feature_obj.tail_value_embed_list[index])
            # endfor
            assert len(tail_property_index_list) == len(
                tail_property_value_embed_list)

            nd_label_index = self.nd_label_to_num_dict[feature_obj.nd_label_text]
            rc_label_index = self.train_rc_str_to_num_dict[feature_obj.rc_label_text]

            nd_label_id_list.append(nd_label_index)
            rc_label_id_list.append(rc_label_index)

            head_property_id_list_of_list.append(head_property_index_list)
            tail_property_id_list_of_list.append(tail_property_index_list)

            head_property_value_embed_matrix_list_of_list.append(
                head_property_value_embed_list)
            tail_property_value_embed_matrix_list_of_list.append(
                tail_property_value_embed_list)
        # endfor

        return head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
            tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
            nd_label_id_list, rc_label_id_list

    def _single_worker_load_features(self, batch_guid_list):
        """
        For each batch, it can contain multiple relation ids, it is OK
        """

        # db = self.connect_to_triple_feature_db()
        db = self.connect_to_wikidata_triple_feature_database()

        feature_obj_list = []

        for guid in tqdm(batch_guid_list, desc="batch get feature from db"):

            cur_relation_id = guid.split("-")[0]

            # ###### get one normal ######
            normal_result_list = []
            for item in db[f"{cur_relation_id}_normal"].find({"guid": guid}):
                normal_result_list.append(item)
            # endfor

            normal_feature_obj = None
            if len(normal_result_list) > 0:
                normal_feature_obj = normal_result_list[0]

            # ###### get one novel: there could be multiple novel instance with the same guid in database ########
            novel_result_list = []
            for item in db[f"{cur_relation_id}_novel"].find({"guid": guid}):
                novel_result_list.append(item)
            # endfor

            novel_feature_obj = None
            if len(novel_result_list) > 0:
                novel_feature_obj = self.randomly_sample_one_from_set(
                    novel_result_list)
            # endif

            if normal_feature_obj is not None and novel_feature_obj is not None:
                feature_obj_list.append(normal_feature_obj)
                feature_obj_list.append(novel_feature_obj)
            # endif
        # endfor

        # ###########
        wikidata_feature_list = []

        for json_obj in feature_obj_list:

            feature_entry = WikidataEntityPairFeature(guid=json_obj["guid"],
                                                      head_wikidata_id=json_obj["head_wikidata_id"],
                                                      tail_wikidata_id=json_obj["tail_wikidata_id"],
                                                      head_property_str_list=json_obj["head_property_id_list"],
                                                      tail_property_str_list=json_obj["tail_property_id_list"],
                                                      head_value_embed_list=json_obj["head_property_value_embed_list"],
                                                      tail_value_embed_list=json_obj["tail_property_value_embed_list"],
                                                      rc_label_text=json_obj["rc_label_index"],  # P1302
                                                      nd_label_text=json_obj["nd_label_index"])  # normal

            wikidata_feature_list.append(feature_entry)
        # endfor

        return wikidata_feature_list

    def parallel_load_normal_novel_features_per_batch(self, batch_guid_list):
        """
        minimized the loading time, by parallel
        """
        # debug
        # all_wikidata_feature_list = self._single_worker_load_features(batch_guid_list)
        # return self.get_model_input(all_wikidata_feature_list)

        num_of_worker = self.args.load_db_data_num_worker
        pool = Pool(processes=num_of_worker)
        block_size = math.ceil(len(batch_guid_list) / num_of_worker)

        job_list = []
        for w in range(num_of_worker):
            block_list = batch_guid_list[block_size * w: block_size * (w + 1)]
            single_job = pool.apply_async(func=self._single_worker_load_features,
                                          args=(block_list,))
            job_list.append(single_job)
        # endfor

        # !! important always close the pool
        # You're creating new processes inside a loop, and then forgetting to close them once you're done with them.
        # As a result, there comes a point where you have too many open processes. This is a bad idea.
        # otherwise error: OSError: [Errno 24] Too many open files
        pool.close()

        all_wikidata_feature_list = []

        for job in job_list:
            tmp_wikidata_feature_list = job.get()
            all_wikidata_feature_list.extend(tmp_wikidata_feature_list)
        # endfor

        return self.get_model_input(all_wikidata_feature_list)

    # ############# create entity examples and features ##############

    def collect_all_entity_id_and_split_to_10(self):

        def _single_get_entity_id_set_from_file_path_list(file_path_list):
            entity_id_set = set()

            for file_path in file_path_list:
                with open(file_path, mode="r") as fin:
                    for line in fin:
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        # endif
                        # P6-0,Q15980310,P6,Q3785077
                        instance_id, head_entity_id, relation_id, tail_entity_id = line.split(
                            ",")
                        entity_id_set.add(head_entity_id)
                        entity_id_set.add(tail_entity_id)
                    # endfor
                # endwith
            # endfor
            return entity_id_set

        # get all triple file path list
        all_triple_file_path_list = []

        triple_folder = "./dataset/ALL_Relations_unique_triples"
        for root, subdir, file_name_list in os.walk(triple_folder):
            for file_n in file_name_list:
                file_path = os.path.join(root, file_n)
                all_triple_file_path_list.append(file_path)
            # endfor
        # endfor

        # single process

        all_entity_id_set = _single_get_entity_id_set_from_file_path_list(
            all_triple_file_path_list)
        print(f"There are totally {len(all_entity_id_set)} entity ids.")

        # ######## split all entity id to 10 fold and output ########
        all_entity_id_list = list(all_entity_id_set)
        block_size = math.ceil(len(all_entity_id_list) / 10)

        output_folder = "./dataset/split_all_train_entity_id_list"

        for i in range(10):
            block_list = all_entity_id_list[i *
                                            block_size: (i + 1) * block_size]
            output_file_path = os.path.join(
                output_folder, f"entity_list_{i}.txt")
            with open(output_file_path, mode="w") as fout:
                for tmp_entity_id in block_list:
                    fout.write(f"{tmp_entity_id}\n")
                # endfor
            # endwith
        # endfor
        pass
    
    def create_entity_example_and_feature_creation_scripts(self):
        import copy

        # #### output folder
        output_folder = "./dataset/entity_example_and_feature_creation_script"

        # #### template input
        template_input_file = "./dataset/create_entity_example_and_feature_script_template.sh"
        template_str = None
        with open(template_input_file, mode="r") as fin:
            template_str = fin.read()
        # endwith

        for i in range(10):
        

            # relation triple path
            relation_triple_file_path = f"./dataset/split_all_train_entity_id_list/entity_list_{i}.txt"

            cur_template_str = copy.deepcopy(template_str)
            cur_template_str = cur_template_str.replace(
                "[entity_id_file_path]", relation_triple_file_path)

            # output file_path
            output_file_path = os.path.join(
                output_folder, f"create_entity_id_list_{i}.sh")
            with open(output_file_path, mode="w") as fout:
                fout.write(f"{cur_template_str}\n")
            # endwith
        # endfor
    # endfor
    
    def split_human_entity_id_into_10_fold(self):
        # all human id set
        all_human_id_set = set()
        
        # ###### load all high quality human entity id
        high_quality_human_id_file_path = "./dataset/high_quality_human_id_folder/high_quality_human_id_set_May_5_2022.txt"
        with open(high_quality_human_id_file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                #endif
                all_human_id_set.add(line)
            #endfor
        #endwith
        
        print(f"There are totally {len(all_human_id_set)} human ids")
                
        
        
        # ######## split all entity id to 10 fold and output ########
        all_entity_id_list = list(all_human_id_set)
        
        block_size = math.ceil(len(all_entity_id_list) / 10)

        output_folder = "./dataset/split_all_high_quality_human_entity_id_list"
        os.makedirs(output_folder, exist_ok=True)
        
        for i in range(10):
            block_list = all_entity_id_list[i *
                                            block_size: (i + 1) * block_size]
            output_file_path = os.path.join(
                output_folder, f"human_entity_list_{i}.txt")
            with open(output_file_path, mode="w") as fout:
                for tmp_entity_id in block_list:
                    fout.write(f"{tmp_entity_id}\n")
                # endfor
            # endwith
        # endfor
        
        pass

        pass


def main_get_all_entities_in_train_valid_test():
    args = argument_parser()
    #
    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    train_data_folder = "./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/training/train_data_100"
    valid_normal_folder = "./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/validation_normal"
    valid_novel_folder = "./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/validation_novel"
    test_normal_folder = "./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/test_normal"
    test_novel_folder = "./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/test_novel"

    high_quality_human_id_file_path = "./dataset/high_quality_human_id_folder/high_quality_human_id_set_May_5_2022.txt"
    high_quality_human_id_set_current_in_database = "./dataset/high_quality_human_id_folder/all_entity_id_IN_database.json"
    
    high_quality_property_str_to_num_file_path = "./dataset/FINAL_property_id_to_info_data/output/property_str_to_index_dict.json"

    total_rc_relation_list_file = "./dataset/relation_information/FILTERED_ALL_relation_label_des_only_info.json"

    train_rc_relation_list_file = "./dataset/model_training_config/train_FILTERED_ALL_relation_label_des_only_info.json"
    evaluation_rc_relation_list_file = "./dataset/model_training_config/evaluation_FILTERED_ALL_relation_label_des_only_info.json"

    # rc_relation_list_file = "./dataset/relation_information/FILTERED_ALL_relation_label_des_only_info.json"

    # #### dataset stats output ###
    #data_info_output_folder = "./data/info_output"
    #os.makedirs(data_info_output_folder, exist_ok=True)

    processor = WikidataProcessor(
        args=args,
        train_data_folder=train_data_folder,
        valid_normal_folder=valid_normal_folder,
        valid_novel_folder=valid_novel_folder,
        test_normal_folder=test_normal_folder,
        test_novel_folder=test_novel_folder,
        high_quality_human_id_file_path=high_quality_human_id_file_path,
        high_quality_human_id_set_current_in_database=high_quality_human_id_set_current_in_database,
        high_quality_property_str_to_num_file_path=high_quality_property_str_to_num_file_path,

        total_rc_relation_list_file=total_rc_relation_list_file,
        train_rc_relation_list_file=train_rc_relation_list_file,
        evaluation_rc_relation_list_file=evaluation_rc_relation_list_file
    )

    entity_info_folder = "./dataset/tmp_train_folder_100"

    # --
    # # 1. get all entity id in training data
    # # (1)
    # training_output_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_train_until_10000.txt")
    # train_folder_list = ["./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/training/train_data_100",
    #                      "./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/training/train_data_1000",
    #                      "./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/training/train_data_2000",
    #                      "./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/training/train_data_10000"]

    # # (2)
    # training_output_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_train_until_100.txt")
    # train_folder_list = ["./dataset/FINAL_RELEASE_factual_novelty_detection_dataset/training/train_data_100"]

    # processor.parallel_get_all_entities_in_FEWREL_format_dataset_file(train_folder_list, training_output_file_path)
    # --

    # --
    # # 2. get all entity id in validation (validation data are not anonymized, process it the same as training data)
    # valid_output_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_valid.txt")
    # valid_folder_list = [valid_normal_folder, valid_novel_folder]
    # processor.parallel_get_all_entities_in_ThreeLine_format_dataset_file(valid_folder_list, valid_output_file_path)
    # --

    # # 3. test data entity ids are anonymized, it needs to be processed differently. It need to be directly loaded into databae
    # Do it later

    # --
    # TODO: test data
    # # 4. verify that train and valid dataset entities are in database

    # # (1)
    # # For train_100 and valid
    # # all in database set: 7575
    # # all NOT in database set: 0

    # # (2)
    # # all in database set: 129663
    # # all NOT in database set: 0

    # # training_output_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_train_100.txt")
    # training_output_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_train_until_10000.txt")

    # valid_output_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_valid.txt")
    # file_path_list = [training_output_file_path,
    #                valid_output_file_path]
    # processor.parallel_check_entity_id_in_wikidata_dump_database(file_path_list)
    # --

    # --
    # # 5. get all one-hop entities
    # the output data contains (a). head / tail entity id  (b). one hop entity id

    # For test data, in its property_value list, if the value is a entity,
    # the entity has already been convered to a label and description
    # so we cannot get all one-hop entities for test data

    # valid_entity_id_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_valid.txt")
    # # train_entity_id_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_train_100.txt")
    # train_entity_id_file_path = os.path.join(entity_info_folder, "head_and_tail_entity_id_in_train_until_10000.txt")

    # input_file_path_list = [valid_entity_id_file_path,
    #                         train_entity_id_file_path]

    # # output_entity_id_file_path_in_database_path = os.path.join(entity_info_folder, "h_t_one_hop_train_100_valid_IN_database_entity_id_list.txt")
    # # output_entity_id_file_path_NOT_in_database_path = os.path.join(entity_info_folder, "h_t_one_hop_train_100_valid_NOT_in_database_entity_id_list.txt")

    # output_entity_id_file_path_in_database_path = os.path.join(entity_info_folder, "h_t_one_hop_train_until_10000_valid_IN_database_entity_id_list.txt")
    # output_entity_id_file_path_NOT_in_database_path = os.path.join(entity_info_folder, "h_t_one_hop_train_until_10000_valid_NOT_in_database_entity_id_list.txt")

    # processor.parallel_get_all_one_hop_entity_id_list(input_file_path_list,
    #                                                   output_entity_id_file_path_in_database_path,
    #                                                   output_entity_id_file_path_NOT_in_database_path)
    # --

    # --
    # # 6. ------------ process all human entities --------------
    # high_quality_human_id_set = processor.load_high_quality_human_id_set()

    # # all in database set: 8_695_657
    # # all NOT in database set: 0

    # # processor.parallel_check_entity_id_LIST_in_wikidata_dump_database(list(high_quality_human_id_set))

    # # 6.1 ------ get one hop entity id from all high quality human entities -----

    # Note:
    # Since train_100, train_until_10000 and valid only has 8 not in database
    # This run for high quality human takes TOO LONG time
    # We has already verfied that these high quality human ids are in database
    # So we discard this run. When faced with some entity not in database in human entity one-hop, just ignore them.

    # output_human_id_one_hop_entity_id_file_path_IN_database_path = os.path.join(entity_info_folder, "high_quality_human_id_IN_database_entity_id_list.txt")
    # output_human_id_one_hop_entity_id_file_path_NOT_in_database_path = os.path.join(entity_info_folder, "high_quality_human_id_NOT_in_database_entity_id_list.txt")
    # processor.parallel_get_all_one_hop_entity_from_entity_id_list(list(high_quality_human_id_set),
    #                                                   output_human_id_one_hop_entity_id_file_path_IN_database_path,
    #                                                   output_human_id_one_hop_entity_id_file_path_NOT_in_database_path)
    # --

    # --
    # # 6.2 ------ check the list of high quality human id that in all_wikidata_features database ------
    # # ----- check the dataset with both (a) examples (b) features ------
    # # all in database set: 1_259_278
    # # all NOT in database set: 7_436_379
    # high_quality_human_id_set = processor.load_high_quality_human_id_set()
    # high_quality_human_id_list = list(high_quality_human_id_set)
    # processor.parallel_check_entity_id_LIST_in_all_wikidata_feature_database(high_quality_human_id_list)
    
    # Jun 17, 2022
    # all in database set: 677_798
    # all NOT in database set: 8017859
    
    # --

    # --
    # # 6.3 ----- load high quality human id list in databaes ------
    # # Current high quality human id in database: 1_259_278
    # #
    # cur_IN_databaes_high_quality_human_id_list = \
    #     processor.load_cur_high_quality_human_id_list_in_all_wikidata_features_database()
    # print(f"Current high quality human id in database: {len(cur_IN_databaes_high_quality_human_id_list)}.")
    
    # 9000 * 24 * 5 = 1080_000 human
    # --

    # TODO: download later
    # download entities NOT in database
    # processor.handler_download_entities_not_in_database()

    # processor.main_parallel_get_data_types_for_all_entities_exclude_human_entity_pool()
    # processor.main_parallel_get_data_type_for_all_high_quality_human()

    # 7. load high quality property id set
    # processor.load_high_quality_property_id_set()

    # 8. create features for entities

    # --
    # # 8.a create features for train_100 and valid
    # # The entity that already in database use previous candidate property id
    # # This new run use new candidate property id
    # h_t_one_hop_train_100_and_valid_entity_list_file_path = "./dataset/tmp_train_folder_100/h_t_one_hop_train_100_valid_IN_database_entity_id_list.txt"
    # with open(h_t_one_hop_train_100_and_valid_entity_list_file_path, mode="r") as fin:
    #     h_t_one_hop_train_100_and_valid_entity_id_list = json.load(fin)
    # #endwith

    # print(f"There are {len(h_t_one_hop_train_100_and_valid_entity_id_list)} entity ids need to be processed to store in all_wikidata_features database.")
    # processor.main_parallel_create_feature_for_entity_id_list(h_t_one_hop_train_100_and_valid_entity_id_list)
    # --

    # 8.b create features for high_quality human ids
    # TODO

    # processor.split_train_valid_test_entity_id_into_different_files()
    # processor.split_human_entity_id_into_different_files()

    # do not have this method
    # processor.main_parallel_create_features_for_all_entities()

    # ################# create features for triples #############
    # below needs args parameter, need to make sure uncomment the code to run the program
    # --
    # # 9. create normal features
    # if args.triple_feature_creation_type == "normal":
    #     processor.create_normal_feature_for_wikidata_triple(triple_size_cap=9000)
    # #endif

    # if args.triple_feature_creation_type == "novel":
    #     processor.create_novel_feature_for_wikidata_triple(triple_size_cap=9000)
    # #endif
    # --

    # 10. create bash script to create normal and novel feature for wikidata triples
    # processor.create_triple_feature_creation_scripts()

    # 11. check the num of documents in wikidata_triple_features in the database
    # processor.count_stats_in_wikidata_triple_features()

    # 12. using pymongo to automatically create index
    processor.pymongo_create_index_for_collections()

    # 13. create examples for validation and test dataset

    # processor.collect_all_entity_id_and_split_to_10()

    # 1. load entity id list

    # 2. load candidate property list
    # parser.add_argument("--entity_id_file_path", default="", type=str, help="")
    
    # python preparation_utils.py --entity_id_file_path ./dataset/split_all_train_entity_id_list/entity_list_0.txt
    
    # print(f"loading entity id list from file path -> {args.entity_id_file_path}")
    # input("")
    # all_entity_id_list = []
    # with open(args.entity_id_file_path, mode="r") as fin:
    #     for line in fin:
    #         line = line.strip()
    #         if len(line) == 0:
    #             continue
    #         # endif
    #         all_entity_id_list.append(line)
    #     # endfor
    # # endwith

    # candidate_property_id_set = processor.load_high_quality_property_id_set()
    # processor._single_worker_create_wikidata_entity_example_and_feature_object_and_store_to_database(all_entity_id_list,
    #                                                                                                  candidate_property_id_set,
    #                                                                                                  0)
    
    # processor.create_entity_example_and_feature_creation_scripts()
    # python preparation_utils.py --entity_id_file_path ./dataset/split_all_high_quality_human_entity_id_list/human_entity_list_0.txt
    
    pass



if __name__ == '__main__':

    # (1) create universal property mapping and property embedding
    # main_create_property_embedding()

    # (2)
    # a. get all the entities in train / valid / test dataset
    # b. get all the high quality human ids
    # c. create wikidata example and feature objects, load into database
    main_get_all_entities_in_train_valid_test()

    # (3)
    # main_wikidata_triple_processor()
