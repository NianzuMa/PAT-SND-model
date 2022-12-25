"""
Using unsupervised learning method to automatically select the most important attribute of
of entities in both subject and object position.

For instance,

Tom Cruise plays a role in the Mission Impossible.

Tom Cruise:
    occupation: actor
    date of birth
    citizenship
    ...

Mission Impossible:
    instance_of: movie
    director
    actors
    production time
    genre
    ...

The goal is that, the unsupervised model can automatically select
-> "occupation" for subject
-> "instance_of" for object
match and learn the knowledge that "actor" acts "movie" is the most important

Then this knowledge can be used in the test case to detect novelty.


   EXPERIMENTS
==================
# 1. sup_contrastive_exp
This experiments use

# 2. unsup_simple_transform

# 3.

"""

from preparation_utils import WikidataProcessor, PropertyProcessor

#from data_utils import PropertyInfoCache, WikidataPropertyRepresentationLoader
#from data_utils import WikidataPropertyRepresentationProcessor, WikidataEntityRepresentationProcessor

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
from pymongo import MongoClient
from data_utils import Wikidata_Database_Processor

# control which GPU to use
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from main_visualize import VisualizeAttention

# --------------- add paths to PYTHON_PATH -----------
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# ------------------ import common package ----------

# ----------------- import model related package -----------

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


# from GAT_models import Net

# ------------ import customized utils -------------


# --------------- config logging -------------

# ------------ create embedding for wikidata property ----------


# python main_db.py \
#   --seed=1123 \
#   --mongodb_url="mongodb://bob:xyz123@bingliu-lamda.cs.uic.edu:27017/wikidata_triple_features" \
#   --resume_training \
#   --resume_checkpoint_dir="./output_tune_param/RGAT_MaxMargin_Stack_2022Feb08_22-08-57_bingliu-lamda_RGAT_MaxMargin_Stack_batch_256_lr_0.001_among_all_relation" \
#   --checkpoint_file_name="checkpoint.pt" \
#   --training_sampling_mode="among_all_relation" \
#   --output_tag="RGAT_MaxMargin_Stack_batch_256_lr_0.001_among_all_relation" \
#   --model_type="RGAT_MaxMargin_Stack" \
#   --logging_steps=40 \
#   --output_dir="./output_tune_param" \
#   --per_gpu_train_batch_size=256 \
#   --per_gpu_eval_batch_size=256 \
#   --load_db_data_num_worker=4 \
#   --learning_rate=0.001 # 0.0005 for batch_size=32


def argument_parser():
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('--mongodb_url', type=str, default="mongodb://bob:xyz123@localhost:27017/wikidata_triple_features", help="Log every X updates steps.")

    parser.add_argument('--checkpoint_file_name', type=str,
                        default="checkpoint.pt", help="Log every X updates steps.")

    parser.add_argument('--resume_training', action='store_true',
                        default=False, help="Log every X updates steps.")

    parser.add_argument('--resume_checkpoint_dir', type=str,
                        default="./output/RGAT_MaxMargin_Stack_2022Feb05_12-19-21_bingliu-lamda_RGAT_MaxMargin_Stack_batch_256_lr_0.001", help="Log every X updates steps.")

    parser.add_argument('--best_model_folder', type=str, default="best_normalized_auc_model",
                        help="Log every X updates steps.")

    parser.add_argument('--logging_steps', type=int,
                        default=1, help="Log every X updates steps.")

    parser.add_argument('--load_db_data_num_worker', type=int,
                        default=4, help="Log every X updates steps.")

    parser.add_argument('--training_sampling_mode', type=str, default="among_all_relation",
                        help="Log every X updates steps.")
    # 1. by_each_relation
    # 2. among_all_relation

    # ------------- dataset -------------
    # train_data_folder = "./data/guid_filtered_ds_train_data_100"
    # # important to set this
    # args.data_tag = 100
    # valid_normal_folder = "./data/guid_fewrel_valid_normal"
    # valid_novel_folder = "./data/guid_fewrel_valid_novel"
    # test_normal_folder = "./data/guid_fewrel_test_normal"
    # test_novel_folder = "./data/guid_fewrel_test_novel"
    # rc_relation_list_file = "./data/RC_relation_list_21.csv"

    # ############################################### dataset #####################################################################
    parser.add_argument("--data_tag", default=100,
                        type=int, help="dataset path")

    # ###### train data ######
    # >>> fewrel format
    parser.add_argument("--train_data_folder", default="./data/guid_filtered_ds_train_data_100", type=str,
                        help="dataset path")
    parser.add_argument("--train_tmp_folder", default="./dataset/tmp_train_folder_100", type=str,
                        help="tmp training folder, for different size of training data, it should be different, remember to change it.")

    # ##### # NOTE: validation and test data root folder ######
    # parser.add_argument("--valid_test_root_folder", default="../dataset/FINAL_RELEASE_v3/ht_FINAL_RELEASE_v3", type=str, help="dataset path")
    parser.add_argument("--valid_test_root_folder",
                        default="../dataset/FINAL_RELEASE_v3/e12_FINAL_RELEASE_v3", type=str, help="dataset path")

    # ##### # NOTE: validation and test example and feature folder #############
    parser.add_argument("--valid_test_example_and_feature_folder",
                        default="../dataset/nd_example_and_feature_e1_e2", type=str, help="dataset path")
    
    parser.add_argument("--predicted_rc_label_file", 
                        default="../dataset/rc_eval_output/2022Oct17_15-47-30_bingliu-lamda_TRAIN_SIZE_2000_with_rev_relation/best_prediction_result.json", type=str, help="")


    # high_quality_human_id_file_path = "./dataset/high_quality_human_id_folder/high_quality_human_id_set_May_5_2022.txt"
    # high_quality_property_str_to_num_file_path = "./dataset/FINAL_property_id_to_info_data/output/property_str_to_index_dict.json"
    # rc_relation_list_file = "./dataset/relation_information/FILTERED_ALL_relation_label_des_only_info.json"

    parser.add_argument("--high_quality_human_id_file_path",
                        default="./dataset/high_quality_human_id_folder/high_quality_human_id_set_May_5_2022.txt", type=str, help="dataset path")

    parser.add_argument("--high_quality_human_id_set_current_in_database",
                        default="./dataset/high_quality_human_id_folder/all_entity_id_IN_database.json", type=str, help="dataset path")

    parser.add_argument("--high_quality_property_str_to_num_file_path",
                        default="./dataset/FINAL_property_id_to_info_data/output/property_str_to_index_dict.json", type=str, help="dataset path")

    parser.add_argument("--train_rc_relation_list_file",
                        default="./training_config/20_relation_id_label_des.json", type=str, help="dataset path")

    parser.add_argument("--evaluation_rc_relation_list_file",
                        default="./training_config/20_relation_id_label_des.json", type=str, help="dataset path")

    # # relation id list
    # parser.add_argument("--rc_relation_list_file", default="./data/RC_relation_list_21.csv", type=str,
    #                     help="dataset path")
    # ##############################################################################################################################

    parser.add_argument("--no_cuda", default=False, type=bool, help="")

    # property information
    parser.add_argument("--property_embed_type",
                        default="label", type=str, help="")
    # type:
    # (1) label
    # (2) label_and_description

    # Note: dataset information
    parser.add_argument("--model_type", default="RGAT_Stack_With_Head_Tail_Attention_CrossEntropy", type=str,
                        help="The model type of interactions of subject/object entities.")
    # RGAT_MaxMargin_Stack_With_Head_Tail_Attention
    # RGAT_MaxMargin_Stack
    # RGAT_Stack_With_Head_Tail_Attention_CrossEntropy
    
    # -------------------------------------
    # - RGAT (classification)
    # - RGAT_Stack ->
    # -------------------------------------
    # - RGAT_MaxMargin (ranking)
    # - RGAT_MaxMargin_Stack (ranking) ->
    # - RGAT_MaxMargin_Stack_With_Head_Tail_Attention
    # -------------------------------------
    # - GAT_MaxMargin
    # -------------------------------------

    # 1. sup_contrastive_exp
    # 2. unsup_simple_transform
    # 3. relational_gat
    # 4. RGAT_MaxMargin

    # parser.add_argument("--data_folder", default="./dataset/valid_test_example_feature_folder", type=str,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # # "./data/output_contrastive_entity_pair"

    parser.add_argument("--dropout", default=0.0, type=float,
                        help="")

    parser.add_argument("--output_dir", default="./output_test", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--output_tag", type=str, default="test",
                        help="show this tag in the output dir, to show the purpose of this run. "
                             "For instance, batch_32_2021Jan11_13-28-35_lambda-quad means "
                             "it is for show result for batch size 32")

    parser.add_argument("--max_des", type=int, default=50,
                        help="the max length of description split by space.")

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
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
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


def train_contrastive_training_extract_feature_from_database_for_all_relation_by_relation_with_e1_and_e2(args,
                                                                                                         evaluation_processor,
                                                                                                         dataset_processor,
                                                                                                         model):
    """
    load pre-generated feature from database for training

    normal example features and novel example features are dynamically loaded in each batch
    so that contrastive training is performed.

    Training Strategy
    -------------------
    Each epoch, training relation by relation, each relation should be trained with 10240 instances
    For relation has more or less than 10240 instances, keep iterate the guid
    """

    # load valid and test raw feature list
    valid_feature_list = dataset_processor.load_and_cache_features_from_database(
        args, mode="valid")
    test_feature_list = dataset_processor.load_and_cache_features_from_database(
        args, mode="test")

    EACH_EPOCH_INSTANCE_NUM = 10240

    # ##### load all normal, novel example guid in database #####  # TIME: 6.222842220465342 mins to create
    all_relation_id_to_guid_list_dict = dataset_processor.parallel_load_relation_train_data_guid_dict(
        args)

    # ##### output training data stats ######
    train_stats_file = os.path.join(args.output_dir, "train_data_stats_db.csv")
    dataset_processor.write_train_data_stats(
        all_relation_id_to_guid_list_dict, train_stats_file)

    # set up tensorboard writer
    tb_writer_train = SummaryWriter(os.path.join(args.output_dir, "tb/train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.output_dir, "tb/valid"))
    tb_writer_test = SummaryWriter(os.path.join(args.output_dir, "tb/test"))

    tb_writer_normalized_train = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_train"))
    tb_writer_normalized_valid = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_valid"))
    tb_writer_normalized_test = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_test"))

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.per_gpu_train_batch_size
    batch_num = math.ceil(EACH_EPOCH_INSTANCE_NUM * 1.0 / batch_size)
    total_optimization_steps = args.num_train_epochs * \
        batch_num * len(dataset_processor.train_rc_str_list)

    print("***** Running training *****")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Num Batch_size = {batch_size}")
    print(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Each relation Num of steps per epoch = {batch_num}")
    print(f"  Total optimization steps = {total_optimization_steps}")

    pytorch_utils.set_seed(args)

    # ############ load resumed checkpoint ###########
    # checkpoint = {"epoch": epoch_index,
    #               "relation_id": relation_id,
    #               "batch_index": batch_index,
    #               "global_step": global_step,
    #               "train_loss": total_train_loss,
    #               "state_dict": model.state_dict(),
    #               "optimizer": optimizer.state_dict()}
    if args.resume_training:
        print("resume training ...")
        checkpoint = pytorch_utils.load_checkpoint(
            args, args.resume_checkpoint_dir)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_global_step = checkpoint["global_step"]
        resume_training_loss = checkpoint["train_loss"]

    # train_dataset is a feature list
    global_step = 0
    total_train_loss = 0
    best_eval_score = 0

    best_mean_auc_score = 0

    best_normalized_auc_score = 0

    model.train()
    model.zero_grad()

    for epoch_index in range(args.num_train_epochs):
        epoch_begin_time = time.time()

        for relation_id in dataset_processor.train_rc_str_list:

            # ########### (1) generate 10240 guid list ###########
            normal_guid_list = all_relation_id_to_guid_list_dict[relation_id]["normal"]
            novel_guid_list = all_relation_id_to_guid_list_dict[relation_id]["novel"]
            guid_list = list(
                set(normal_guid_list).intersection(set(novel_guid_list)))

            # shuffle guid
            random.shuffle(guid_list)

            epoch_guid_list = None
            if len(guid_list) >= EACH_EPOCH_INSTANCE_NUM:
                epoch_guid_list = guid_list[:EACH_EPOCH_INSTANCE_NUM]
            else:
                times_num = math.ceil(EACH_EPOCH_INSTANCE_NUM / len(guid_list))
                new_guid_list = guid_list * times_num
                epoch_guid_list = new_guid_list[:EACH_EPOCH_INSTANCE_NUM]
            # endif
            assert len(epoch_guid_list) == EACH_EPOCH_INSTANCE_NUM

            # ######### (2) get guid list by batch #############

            for batch_index in range(batch_num):
                global_step += 1

                # skip until the correct progress
                if args.resume_training:
                    total_train_loss = resume_training_loss
                    if global_step <= resume_global_step:
                        continue

                # !!! always put model.train() here before each batch training
                model.train()
                model.zero_grad()

                batch_begin_time = time.time()

                batch_guid_list = epoch_guid_list[batch_size *
                                                  batch_index: batch_size * (batch_index + 1)]

                head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
                    tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
                    nd_label_id_list, rc_label_id_list = dataset_processor.parallel_load_normal_novel_features_per_batch(
                        batch_guid_list)

                # load on device
                head_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                                head_property_id_list_of_list]
                tail_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                                tail_property_id_list_of_list]

                head_property_value_embeds_matrix_tensor_list = [
                    torch.tensor(item, dtype=torch.float).float().to(args.device) for
                    item in head_property_value_embed_matrix_list_of_list]

                tail_property_value_embeds_matrix_tensor_list = [
                    torch.tensor(item, dtype=torch.float).float().to(args.device) for
                    item in tail_property_value_embed_matrix_list_of_list]

                nd_label_id_list = torch.tensor(
                    nd_label_id_list, dtype=torch.long).to(args.device)
                # rc_label_id_list = torch.tensor(rc_label_id_list, dtype=torch.long).to(args.device)

                total_score, loss = model(args,
                                          head_property_id_tensor_list,
                                          tail_property_id_tensor_list,
                                          head_property_value_embeds_matrix_tensor_list,
                                          tail_property_value_embeds_matrix_tensor_list,
                                          nd_label_id_list,
                                          rc_label_id_list)
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()

                total_train_loss += loss.item()

                cur_train_loss = total_train_loss * 1.0 / global_step

                train_info_json = {"epoch": epoch_index,
                                   "relation_id": relation_id,
                                   "batch_index": f"{batch_index}/{batch_num}",
                                   "global_step": f"{global_step} / {total_optimization_steps}",
                                   "train_loss": cur_train_loss}

                print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

                # Note: tensorboard writer
                tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)

                with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
                    fout.write(json.dumps(train_info_json) + "\n")
                # endwith

                if global_step % args.logging_steps == 0 and batch_index >= 1:
                    if args.evaluate_during_training:

                        # Note: ============= VALID evaluation ==========
                        # valid log is saved inside method evaluate
                        valid_results_json_dict = evaluate_for_gat_21_relation(args,
                                                                               valid_feature_list,
                                                                               dataset_processor,
                                                                               model,
                                                                               mode="valid",
                                                                               epoch_index=epoch_index,
                                                                               step=global_step)

                        print(
                            f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
                        valid_auc_score = valid_results_json_dict["auc_score"]
                        # Note: tensorboard valid auc score
                        tb_writer_valid.add_scalar(
                            "valid_auc", valid_auc_score, global_step)
                        # ########
                        valid_normalized_auc_score = valid_results_json_dict["normalized_auc_score"]
                        tb_writer_normalized_valid.add_scalar("valid_normalized_auc", valid_normalized_auc_score,
                                                              global_step)

                        # ================ TEST DATA evaluaton =============
                        # test log is saved inside method evaluate
                        test_results_json_dict = evaluate_for_gat_21_relation(args,
                                                                              test_feature_list,
                                                                              dataset_processor,
                                                                              model,
                                                                              mode="test",
                                                                              epoch_index=epoch_index,
                                                                              if_write_pred_result=True,
                                                                              step=global_step)

                        print(
                            f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
                        assert test_results_json_dict is not None
                        test_auc_score = test_results_json_dict["auc_score"]
                        # Note: tensorboard test auc score
                        tb_writer_test.add_scalar(
                            "test_auc", test_auc_score, global_step)
                        # #########
                        test_normalized_auc_score = test_results_json_dict["normalized_auc_score"]
                        tb_writer_normalized_test.add_scalar("test_normalized_auc", test_normalized_auc_score,
                                                             global_step)

                        # ============= training information need to be saved ==============
                        # {"epoch": 0, "relation_id": "P241", "batch": "39/40", "global_step": "360 / 8400000"}

                        checkpoint = {"epoch": epoch_index,
                                      "relation_id": relation_id,
                                      "batch_index": batch_index,
                                      "global_step": global_step,
                                      "train_loss": total_train_loss,
                                      "state_dict": model.state_dict(),
                                      "optimizer": optimizer.state_dict()}

                        checkpoint_file_path = os.path.join(
                            args.output_dir, args.checkpoint_file_name)
                        torch.save(checkpoint, checkpoint_file_path)

                        # Note: ======== SAVE BEST normalized auc model ===========
                        current_normalized_auc_score = valid_results_json_dict["normalized_auc_score"]

                        if current_normalized_auc_score > best_normalized_auc_score:
                            best_normalized_auc_score = current_normalized_auc_score

                            normalized_sub_dir = os.path.join(
                                args.output_dir, "best_normalized_auc_model")
                            os.makedirs(normalized_sub_dir, exist_ok=True)
                            normalized_checkpoint_file = os.path.join(
                                normalized_sub_dir, args.checkpoint_file_name)
                            shutil.copyfile(
                                src=checkpoint_file_path, dst=normalized_checkpoint_file)

                            # >>>>>>>>>>> save the best eval information
                            with open(os.path.join(args.output_dir, "best_valid_normalized_auc_result.json"),
                                      mode="w") as fout:
                                fout.write(json.dumps(
                                    valid_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>>>> save the best test information
                            with open(os.path.join(args.output_dir, "best_test_normalized_auc_result.json"),
                                      mode="w") as fout:
                                fout.write(json.dumps(
                                    test_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>>>> save both valid / test mean auc
                            with open(os.path.join(args.output_dir, "best_valid_test_result_normalized_auc_log.json"),
                                      mode="a") as fout:
                                fout.write(
                                    "valid: " + json.dumps(valid_results_json_dict) + "\n")
                                fout.write(
                                    "test:  " + json.dumps(test_results_json_dict) + "\n")
                                fout.write("\n")
                            # endwith

                        # Note: ======== SAVE BEST MEAN AUC MODEL ===========
                        current_mean_auc_score = valid_results_json_dict["mean_of_each_relation_auc"]

                        if current_mean_auc_score > best_mean_auc_score:
                            best_mean_auc_score = current_mean_auc_score

                            mean_sub_dir = os.path.join(
                                args.output_dir, "best_mean_auc_model")
                            os.makedirs(mean_sub_dir, exist_ok=True)
                            mean_checkpoint_file = os.path.join(
                                mean_sub_dir, args.checkpoint_file_name)
                            shutil.copyfile(
                                src=checkpoint_file_path, dst=mean_checkpoint_file)

                            # >>>>>>>>>>> save the best eval information
                            with open(os.path.join(args.output_dir, "best_valid_mean_auc_result.json"),
                                      mode="w") as fout:
                                fout.write(json.dumps(
                                    valid_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>>>> save the best test information
                            with open(os.path.join(args.output_dir, "best_test_mean_auc_result.json"),
                                      mode="w") as fout:
                                fout.write(json.dumps(
                                    test_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>>>> save both valid / test mean auc
                            with open(os.path.join(args.output_dir, "best_valid_test_result_mean_auc_log.json"),
                                      mode="a") as fout:
                                fout.write(
                                    "valid: " + json.dumps(valid_results_json_dict) + "\n")
                                fout.write(
                                    "test:  " + json.dumps(test_results_json_dict) + "\n")
                                fout.write("\n")
                            # endwith

                        # Note: ================ SAVE THE BEST ===============

                        # save model if the model give the best metrics we care
                        current_eval_score = valid_results_json_dict[args.considered_metrics]

                        if current_eval_score > best_eval_score:
                            best_eval_score = current_eval_score

                            sub_dir = os.path.join(
                                args.output_dir, "best_auc_model")
                            os.makedirs(sub_dir, exist_ok=True)
                            best_checkpoint_file = os.path.join(
                                sub_dir, args.checkpoint_file_name)
                            shutil.copyfile(
                                src=checkpoint_file_path, dst=best_checkpoint_file)

                            # >>>>>>> save the best eval information
                            with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
                                fout.write(json.dumps(
                                    valid_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>> save the best test information, and history information
                            with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
                                fout.write(json.dumps(
                                    test_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>> save both valid test log
                            with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"),
                                      mode="a") as fout:
                                fout.write(
                                    "valid: " + json.dumps(valid_results_json_dict) + "\n")
                                fout.write(
                                    "test:  " + json.dumps(test_results_json_dict) + "\n")
                                fout.write("\n")
                            # endwith

                    # endif
                # end if evaluation

                batch_time_length_sec = (time.time() - batch_begin_time) * 1.0
                print(
                    f">>>>>>>>>>>>>>> This batch takes {batch_time_length_sec} sec")
                batch_time_json = {"batch_index": batch_index,
                                   "relation_id": relation_id,
                                   "time_sec": batch_time_length_sec}
                with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
                    fout.write(f"{json.dumps(batch_time_json)}\n")
                # endwith
                # end all batches

            # end one relation
        # endfor

        # end of epoch
        epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
        print(
            f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
        epoch_time_json = {"epoch_index": epoch_index,
                           "time_mins": epoch_time_length_min}
        with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
            fout.write(f"{json.dumps(epoch_time_json)}\n")
        # endwith

    #     # shuffle!
    #     random.shuffle(train_features)
    #     print("The train dataset is shuffling for epoch {}".format(epoch_index))
    #
    #     for batch_index in range(batch_num):
    #         global_step += 1
    #
    #         print(
    #             f"\n\n\n>>>>>>>>>>>>>> batch[{batch_index}/{batch_num}] -- epoch[{epoch_index}] -- global_step[{global_step}] <<<<<<<<<<<<<<<<<<<<<<")
    #         batch_begin_time = time.time()
    #
    #         model.train()
    #         model.zero_grad()
    #
    #         # (1) dynamically sample negative data examples
    #         # (2) create features for both positive/negative examples
    #         head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
    #         tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
    #         nd_label_id_list, rc_label_id_list = dataloading_processor.dynamically_get_positive_negative_batch_features(
    #             train_features[batch_index * batch_size: (batch_index + 1) * batch_size])
    #
    #         # load on device
    #         head_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
    #                                         head_property_id_list_of_list]
    #         tail_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
    #                                         tail_property_id_list_of_list]
    #
    #         head_property_value_embeds_matrix_tensor_list = [
    #             torch.tensor(item, dtype=torch.float).float().to(args.device) for
    #             item in head_property_value_embed_matrix_list_of_list]
    #
    #         tail_property_value_embeds_matrix_tensor_list = [
    #             torch.tensor(item, dtype=torch.float).float().to(args.device) for
    #             item in tail_property_value_embed_matrix_list_of_list]
    #
    #         nd_label_id_list = torch.tensor(nd_label_id_list, dtype=torch.long).to(args.device)
    #         # rc_label_id_list = torch.tensor(rc_label_id_list, dtype=torch.long).to(args.device)
    #
    #         total_score, loss = model(args,
    #                                   head_property_id_tensor_list,
    #                                   tail_property_id_tensor_list,
    #                                   head_property_value_embeds_matrix_tensor_list,
    #                                   tail_property_value_embeds_matrix_tensor_list,
    #                                   nd_label_id_list,
    #                                   rc_label_id_list)
    #         loss.backward()
    #
    #         # TODO: check this later
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #         optimizer.step()
    #         model.zero_grad()
    #
    #         total_train_loss += loss.item()
    #
    #         cur_train_loss = total_train_loss * 1.0 / global_step
    #
    #         train_info_json = {"epoch": epoch_index, "batch": f"{batch_index}/{batch_num}", "global_step": global_step,
    #                            "train_loss": cur_train_loss}
    #
    #         print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")
    #
    #         # Note: tensorboard writer
    #         tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)
    #
    #         with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
    #             fout.write(json.dumps(train_info_json) + "\n")
    #         # endwith
    #
    #         if global_step % args.logging_steps == 0 and global_step > 5:
    #             if args.evaluate_during_training:
    #
    #                 # Note: ============= VALID evaluation ==========
    #                 # valid log is saved inside method evaluate
    #                 valid_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                        dataloading_processor,
    #                                                                        model,
    #                                                                        mode="valid",
    #                                                                        epoch_index=epoch_index,
    #                                                                        step=global_step)
    #
    #                 print(
    #                     f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
    #                 valid_auc_score = valid_results_json_dict["auc_score"]
    #                 # Note: tensorboard valid auc score
    #                 tb_writer_valid.add_scalar(
    #                     "valid_auc", valid_auc_score, global_step)
    #
    #                 # Note: ========= SAVE THE BEST ==========
    #                 if_best_model = False
    #                 test_results_json_dict = None
    #                 # save model if the model give the best metrics we care
    #                 current_eval_score = valid_results_json_dict[args.considered_metrics]
    #                 if current_eval_score > best_eval_score:
    #                     if_best_model = True
    #                     # save the best model
    #                     best_eval_score = current_eval_score
    #                     subdir = os.path.join(args.output_dir, "best_model")
    #                     if not os.path.exists(subdir):
    #                         os.makedirs(subdir)
    #                     # endif
    #                     pytorch_utils.save_model(model,
    #                                              os.path.join(args.output_dir, "best_model", args.save_model_file_name))
    #
    #                     # save the best eval information
    #                     with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
    #                         fout.write(json.dumps(
    #                             valid_results_json_dict) + "\n")
    #                     # endwith
    #
    #                     # Note: ---------- test on the best model -------------
    #                     # use the current best model to evaluate on test data
    #                     test_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                           dataloading_processor,
    #                                                                           model,
    #                                                                           mode="test",
    #                                                                           epoch_index=epoch_index,
    #                                                                           if_write_pred_result=True,
    #                                                                           step=global_step)
    #                     print(
    #                         f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
    #
    #                     # save the best test information, and history information
    #                     with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
    #                         fout.write(json.dumps(
    #                             test_results_json_dict) + "\n")
    #                     # endwith
    #
    #                     with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"), mode="a") as fout:
    #                         fout.write(
    #                             "valid: " + json.dumps(valid_results_json_dict) + "\n")
    #                         fout.write(
    #                             "test:  " + json.dumps(test_results_json_dict) + "\n")
    #                         fout.write("\n")
    #                     # endwith
    #                 # endif
    #
    #                 # Note: ========== TEST evaluation ==========
    #                 # if not best model, then evaluate test, otherwise it is already evaluated
    #                 if not if_best_model:
    #                     # test log is saved inside method evaluate
    #                     test_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                           dataloading_processor,
    #                                                                           model,
    #                                                                           mode="test",
    #                                                                           epoch_index=epoch_index,
    #                                                                           if_write_pred_result=True,
    #                                                                           step=global_step)
    #                     print(
    #                         f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
    #                 # endif
    #                 assert test_results_json_dict is not None
    #                 test_auc_score = test_results_json_dict["auc_score"]
    #                 # Note: tensorboard test auc score
    #                 tb_writer_test.add_scalar(
    #                     "test_auc", test_auc_score, global_step)
    #
    #             # endif
    #         # endif
    #
    #         # end of batch
    #         batch_time_length_min = (time.time() - batch_begin_time) * 1.0 / 60
    #         print(
    #             f">>>>>>>>>>>>>>> This batch takes {batch_time_length_min} min")
    #         batch_time_json = {"batch_index": batch_index,
    #                            "time_mins": batch_time_length_min}
    #         with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
    #             fout.write(f"{json.dumps(batch_time_json)}\n")
    #         # endwith
    #     # endfor
    #
    #     # end of epoch
    #     epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
    #     print(
    #         f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
    #     epoch_time_json = {"epoch_index": epoch_index,
    #                        "time_mins": epoch_time_length_min}
    #     with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
    #         fout.write(f"{json.dumps(epoch_time_json)}\n")
    #     # endwith
    # # endfor


def train_contrastive_training_extract_feature_from_database_for_all_relation_by_relation(args,
                                                                                          dataset_processor,
                                                                                          model):
    """
    load pre-generated feature from database for training

    normal example features and novel example features are dynamically loaded in each batch
    so that contrastive training is performed.

    Training Strategy
    -------------------
    Each epoch, training relation by relation, each relation should be trained with 10240 instances
    For relation has more or less than 10240 instances, keep iterate the guid
    """

    # load valid and test raw feature list
    valid_feature_list = dataset_processor.load_and_cache_features_from_database(
        args, mode="valid")
    test_feature_list = dataset_processor.load_and_cache_features_from_database(
        args, mode="test")

    EACH_EPOCH_INSTANCE_NUM = 10240

    # ##### load all normal, novel example guid in database #####  # TIME: 6.222842220465342 mins to create
    all_relation_id_to_guid_list_dict = dataset_processor.parallel_load_relation_train_data_guid_dict(
        args)

    # ##### output training data stats ######
    train_stats_file = os.path.join(args.output_dir, "train_data_stats_db.csv")
    dataset_processor.write_train_data_stats(
        all_relation_id_to_guid_list_dict, train_stats_file)

    # set up tensorboard writer
    tb_writer_train = SummaryWriter(os.path.join(args.output_dir, "tb/train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.output_dir, "tb/valid"))
    tb_writer_test = SummaryWriter(os.path.join(args.output_dir, "tb/test"))

    tb_writer_normalized_train = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_train"))
    tb_writer_normalized_valid = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_valid"))
    tb_writer_normalized_test = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_test"))

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.per_gpu_train_batch_size
    batch_num = math.ceil(EACH_EPOCH_INSTANCE_NUM * 1.0 / batch_size)
    total_optimization_steps = args.num_train_epochs * \
        batch_num * len(dataset_processor.train_rc_str_list)

    print("***** Running training *****")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Num Batch_size = {batch_size}")
    print(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Each relation Num of steps per epoch = {batch_num}")
    print(f"  Total optimization steps = {total_optimization_steps}")

    pytorch_utils.set_seed(args)

    # ############ load resumed checkpoint ###########
    # checkpoint = {"epoch": epoch_index,
    #               "relation_id": relation_id,
    #               "batch_index": batch_index,
    #               "global_step": global_step,
    #               "train_loss": total_train_loss,
    #               "state_dict": model.state_dict(),
    #               "optimizer": optimizer.state_dict()}
    if args.resume_training:
        print("resume training ...")
        checkpoint = pytorch_utils.load_checkpoint(
            args, args.resume_checkpoint_dir)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_global_step = checkpoint["global_step"]
        resume_training_loss = checkpoint["train_loss"]

    # train_dataset is a feature list
    global_step = 0
    total_train_loss = 0
    best_eval_score = 0

    best_mean_auc_score = 0

    best_normalized_auc_score = 0

    model.train()
    model.zero_grad()

    for epoch_index in range(args.num_train_epochs):
        epoch_begin_time = time.time()

        for relation_id in dataset_processor.train_rc_str_list:

            # ########### (1) generate 10240 guid list ###########
            normal_guid_list = all_relation_id_to_guid_list_dict[relation_id]["normal"]
            novel_guid_list = all_relation_id_to_guid_list_dict[relation_id]["novel"]
            guid_list = list(
                set(normal_guid_list).intersection(set(novel_guid_list)))

            # shuffle guid
            random.shuffle(guid_list)

            epoch_guid_list = None
            if len(guid_list) >= EACH_EPOCH_INSTANCE_NUM:
                epoch_guid_list = guid_list[:EACH_EPOCH_INSTANCE_NUM]
            else:
                times_num = math.ceil(EACH_EPOCH_INSTANCE_NUM / len(guid_list))
                new_guid_list = guid_list * times_num
                epoch_guid_list = new_guid_list[:EACH_EPOCH_INSTANCE_NUM]
            # endif
            assert len(epoch_guid_list) == EACH_EPOCH_INSTANCE_NUM

            # ######### (2) get guid list by batch #############

            for batch_index in range(batch_num):
                global_step += 1

                # skip until the correct progress
                if args.resume_training:
                    total_train_loss = resume_training_loss
                    if global_step <= resume_global_step:
                        continue

                # !!! always put model.train() here before each batch training
                model.train()
                model.zero_grad()

                batch_begin_time = time.time()

                batch_guid_list = epoch_guid_list[batch_size *
                                                  batch_index: batch_size * (batch_index + 1)]

                head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
                    tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
                    nd_label_id_list, rc_label_id_list = dataset_processor.parallel_load_normal_novel_features_per_batch(
                        batch_guid_list)

                # load on device
                head_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                                head_property_id_list_of_list]
                tail_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                                tail_property_id_list_of_list]

                head_property_value_embeds_matrix_tensor_list = [
                    torch.tensor(item, dtype=torch.float).float().to(args.device) for
                    item in head_property_value_embed_matrix_list_of_list]

                tail_property_value_embeds_matrix_tensor_list = [
                    torch.tensor(item, dtype=torch.float).float().to(args.device) for
                    item in tail_property_value_embed_matrix_list_of_list]

                nd_label_id_list = torch.tensor(
                    nd_label_id_list, dtype=torch.long).to(args.device)
                # rc_label_id_list = torch.tensor(rc_label_id_list, dtype=torch.long).to(args.device)

                total_score, loss = model(args,
                                          head_property_id_tensor_list,
                                          tail_property_id_tensor_list,
                                          head_property_value_embeds_matrix_tensor_list,
                                          tail_property_value_embeds_matrix_tensor_list,
                                          nd_label_id_list,
                                          rc_label_id_list)
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()

                total_train_loss += loss.item()

                cur_train_loss = total_train_loss * 1.0 / global_step

                train_info_json = {"epoch": epoch_index,
                                   "relation_id": relation_id,
                                   "batch_index": f"{batch_index}/{batch_num}",
                                   "global_step": f"{global_step} / {total_optimization_steps}",
                                   "train_loss": cur_train_loss}

                print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

                # Note: tensorboard writer
                tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)

                with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
                    fout.write(json.dumps(train_info_json) + "\n")
                # endwith

                if global_step % args.logging_steps == 0 and batch_index >= 1:
                    if args.evaluate_during_training:

                        # Note: ============= VALID evaluation ==========
                        # valid log is saved inside method evaluate
                        valid_results_json_dict = evaluate_for_gat_21_relation(args,
                                                                               valid_feature_list,
                                                                               dataset_processor,
                                                                               model,
                                                                               mode="valid",
                                                                               epoch_index=epoch_index,
                                                                               step=global_step)

                        print(
                            f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
                        valid_auc_score = valid_results_json_dict["auc_score"]
                        # Note: tensorboard valid auc score
                        tb_writer_valid.add_scalar(
                            "valid_auc", valid_auc_score, global_step)
                        # ########
                        valid_normalized_auc_score = valid_results_json_dict["normalized_auc_score"]
                        tb_writer_normalized_valid.add_scalar("valid_normalized_auc", valid_normalized_auc_score,
                                                              global_step)

                        # ================ TEST DATA evaluaton =============
                        # test log is saved inside method evaluate
                        test_results_json_dict = evaluate_for_gat_21_relation(args,
                                                                              test_feature_list,
                                                                              dataset_processor,
                                                                              model,
                                                                              mode="test",
                                                                              epoch_index=epoch_index,
                                                                              if_write_pred_result=True,
                                                                              step=global_step)

                        print(
                            f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
                        assert test_results_json_dict is not None
                        test_auc_score = test_results_json_dict["auc_score"]
                        # Note: tensorboard test auc score
                        tb_writer_test.add_scalar(
                            "test_auc", test_auc_score, global_step)
                        # #########
                        test_normalized_auc_score = test_results_json_dict["normalized_auc_score"]
                        tb_writer_normalized_test.add_scalar("test_normalized_auc", test_normalized_auc_score,
                                                             global_step)

                        # ============= training information need to be saved ==============
                        # {"epoch": 0, "relation_id": "P241", "batch": "39/40", "global_step": "360 / 8400000"}

                        checkpoint = {"epoch": epoch_index,
                                      "relation_id": relation_id,
                                      "batch_index": batch_index,
                                      "global_step": global_step,
                                      "train_loss": total_train_loss,
                                      "state_dict": model.state_dict(),
                                      "optimizer": optimizer.state_dict()}

                        checkpoint_file_path = os.path.join(
                            args.output_dir, args.checkpoint_file_name)
                        torch.save(checkpoint, checkpoint_file_path)

                        # Note: ======== SAVE BEST normalized auc model ===========
                        current_normalized_auc_score = valid_results_json_dict["normalized_auc_score"]

                        if current_normalized_auc_score > best_normalized_auc_score:
                            best_normalized_auc_score = current_normalized_auc_score

                            normalized_sub_dir = os.path.join(
                                args.output_dir, "best_normalized_auc_model")
                            os.makedirs(normalized_sub_dir, exist_ok=True)
                            normalized_checkpoint_file = os.path.join(
                                normalized_sub_dir, args.checkpoint_file_name)
                            shutil.copyfile(
                                src=checkpoint_file_path, dst=normalized_checkpoint_file)

                            # >>>>>>>>>>> save the best eval information
                            with open(os.path.join(args.output_dir, "best_valid_normalized_auc_result.json"),
                                      mode="w") as fout:
                                fout.write(json.dumps(
                                    valid_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>>>> save the best test information
                            with open(os.path.join(args.output_dir, "best_test_normalized_auc_result.json"),
                                      mode="w") as fout:
                                fout.write(json.dumps(
                                    test_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>>>> save both valid / test mean auc
                            with open(os.path.join(args.output_dir, "best_valid_test_result_normalized_auc_log.json"),
                                      mode="a") as fout:
                                fout.write(
                                    "valid: " + json.dumps(valid_results_json_dict) + "\n")
                                fout.write(
                                    "test:  " + json.dumps(test_results_json_dict) + "\n")
                                fout.write("\n")
                            # endwith

                        # Note: ======== SAVE BEST MEAN AUC MODEL ===========
                        current_mean_auc_score = valid_results_json_dict["mean_of_each_relation_auc"]

                        if current_mean_auc_score > best_mean_auc_score:
                            best_mean_auc_score = current_mean_auc_score

                            mean_sub_dir = os.path.join(
                                args.output_dir, "best_mean_auc_model")
                            os.makedirs(mean_sub_dir, exist_ok=True)
                            mean_checkpoint_file = os.path.join(
                                mean_sub_dir, args.checkpoint_file_name)
                            shutil.copyfile(
                                src=checkpoint_file_path, dst=mean_checkpoint_file)

                            # >>>>>>>>>>> save the best eval information
                            with open(os.path.join(args.output_dir, "best_valid_mean_auc_result.json"),
                                      mode="w") as fout:
                                fout.write(json.dumps(
                                    valid_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>>>> save the best test information
                            with open(os.path.join(args.output_dir, "best_test_mean_auc_result.json"),
                                      mode="w") as fout:
                                fout.write(json.dumps(
                                    test_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>>>> save both valid / test mean auc
                            with open(os.path.join(args.output_dir, "best_valid_test_result_mean_auc_log.json"),
                                      mode="a") as fout:
                                fout.write(
                                    "valid: " + json.dumps(valid_results_json_dict) + "\n")
                                fout.write(
                                    "test:  " + json.dumps(test_results_json_dict) + "\n")
                                fout.write("\n")
                            # endwith

                        # Note: ================ SAVE THE BEST ===============

                        # save model if the model give the best metrics we care
                        current_eval_score = valid_results_json_dict[args.considered_metrics]

                        if current_eval_score > best_eval_score:
                            best_eval_score = current_eval_score

                            sub_dir = os.path.join(
                                args.output_dir, "best_auc_model")
                            os.makedirs(sub_dir, exist_ok=True)
                            best_checkpoint_file = os.path.join(
                                sub_dir, args.checkpoint_file_name)
                            shutil.copyfile(
                                src=checkpoint_file_path, dst=best_checkpoint_file)

                            # >>>>>>> save the best eval information
                            with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
                                fout.write(json.dumps(
                                    valid_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>> save the best test information, and history information
                            with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
                                fout.write(json.dumps(
                                    test_results_json_dict) + "\n")
                            # endwith

                            # >>>>>>>>> save both valid test log
                            with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"),
                                      mode="a") as fout:
                                fout.write(
                                    "valid: " + json.dumps(valid_results_json_dict) + "\n")
                                fout.write(
                                    "test:  " + json.dumps(test_results_json_dict) + "\n")
                                fout.write("\n")
                            # endwith

                    # endif
                # end if evaluation

                batch_time_length_sec = (time.time() - batch_begin_time) * 1.0
                print(
                    f">>>>>>>>>>>>>>> This batch takes {batch_time_length_sec} sec")
                batch_time_json = {"batch_index": batch_index,
                                   "relation_id": relation_id,
                                   "time_sec": batch_time_length_sec}
                with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
                    fout.write(f"{json.dumps(batch_time_json)}\n")
                # endwith
                # end all batches

            # end one relation
        # endfor

        # end of epoch
        epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
        print(
            f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
        epoch_time_json = {"epoch_index": epoch_index,
                           "time_mins": epoch_time_length_min}
        with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
            fout.write(f"{json.dumps(epoch_time_json)}\n")
        # endwith

    #     # shuffle!
    #     random.shuffle(train_features)
    #     print("The train dataset is shuffling for epoch {}".format(epoch_index))
    #
    #     for batch_index in range(batch_num):
    #         global_step += 1
    #
    #         print(
    #             f"\n\n\n>>>>>>>>>>>>>> batch[{batch_index}/{batch_num}] -- epoch[{epoch_index}] -- global_step[{global_step}] <<<<<<<<<<<<<<<<<<<<<<")
    #         batch_begin_time = time.time()
    #
    #         model.train()
    #         model.zero_grad()
    #
    #         # (1) dynamically sample negative data examples
    #         # (2) create features for both positive/negative examples
    #         head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
    #         tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
    #         nd_label_id_list, rc_label_id_list = dataloading_processor.dynamically_get_positive_negative_batch_features(
    #             train_features[batch_index * batch_size: (batch_index + 1) * batch_size])
    #
    #         # load on device
    #         head_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
    #                                         head_property_id_list_of_list]
    #         tail_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
    #                                         tail_property_id_list_of_list]
    #
    #         head_property_value_embeds_matrix_tensor_list = [
    #             torch.tensor(item, dtype=torch.float).float().to(args.device) for
    #             item in head_property_value_embed_matrix_list_of_list]
    #
    #         tail_property_value_embeds_matrix_tensor_list = [
    #             torch.tensor(item, dtype=torch.float).float().to(args.device) for
    #             item in tail_property_value_embed_matrix_list_of_list]
    #
    #         nd_label_id_list = torch.tensor(nd_label_id_list, dtype=torch.long).to(args.device)
    #         # rc_label_id_list = torch.tensor(rc_label_id_list, dtype=torch.long).to(args.device)
    #
    #         total_score, loss = model(args,
    #                                   head_property_id_tensor_list,
    #                                   tail_property_id_tensor_list,
    #                                   head_property_value_embeds_matrix_tensor_list,
    #                                   tail_property_value_embeds_matrix_tensor_list,
    #                                   nd_label_id_list,
    #                                   rc_label_id_list)
    #         loss.backward()
    #
    #         # TODO: check this later
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #         optimizer.step()
    #         model.zero_grad()
    #
    #         total_train_loss += loss.item()
    #
    #         cur_train_loss = total_train_loss * 1.0 / global_step
    #
    #         train_info_json = {"epoch": epoch_index, "batch": f"{batch_index}/{batch_num}", "global_step": global_step,
    #                            "train_loss": cur_train_loss}
    #
    #         print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")
    #
    #         # Note: tensorboard writer
    #         tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)
    #
    #         with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
    #             fout.write(json.dumps(train_info_json) + "\n")
    #         # endwith
    #
    #         if global_step % args.logging_steps == 0 and global_step > 5:
    #             if args.evaluate_during_training:
    #
    #                 # Note: ============= VALID evaluation ==========
    #                 # valid log is saved inside method evaluate
    #                 valid_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                        dataloading_processor,
    #                                                                        model,
    #                                                                        mode="valid",
    #                                                                        epoch_index=epoch_index,
    #                                                                        step=global_step)
    #
    #                 print(
    #                     f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
    #                 valid_auc_score = valid_results_json_dict["auc_score"]
    #                 # Note: tensorboard valid auc score
    #                 tb_writer_valid.add_scalar(
    #                     "valid_auc", valid_auc_score, global_step)
    #
    #                 # Note: ========= SAVE THE BEST ==========
    #                 if_best_model = False
    #                 test_results_json_dict = None
    #                 # save model if the model give the best metrics we care
    #                 current_eval_score = valid_results_json_dict[args.considered_metrics]
    #                 if current_eval_score > best_eval_score:
    #                     if_best_model = True
    #                     # save the best model
    #                     best_eval_score = current_eval_score
    #                     subdir = os.path.join(args.output_dir, "best_model")
    #                     if not os.path.exists(subdir):
    #                         os.makedirs(subdir)
    #                     # endif
    #                     pytorch_utils.save_model(model,
    #                                              os.path.join(args.output_dir, "best_model", args.save_model_file_name))
    #
    #                     # save the best eval information
    #                     with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
    #                         fout.write(json.dumps(
    #                             valid_results_json_dict) + "\n")
    #                     # endwith
    #
    #                     # Note: ---------- test on the best model -------------
    #                     # use the current best model to evaluate on test data
    #                     test_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                           dataloading_processor,
    #                                                                           model,
    #                                                                           mode="test",
    #                                                                           epoch_index=epoch_index,
    #                                                                           if_write_pred_result=True,
    #                                                                           step=global_step)
    #                     print(
    #                         f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
    #
    #                     # save the best test information, and history information
    #                     with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
    #                         fout.write(json.dumps(
    #                             test_results_json_dict) + "\n")
    #                     # endwith
    #
    #                     with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"), mode="a") as fout:
    #                         fout.write(
    #                             "valid: " + json.dumps(valid_results_json_dict) + "\n")
    #                         fout.write(
    #                             "test:  " + json.dumps(test_results_json_dict) + "\n")
    #                         fout.write("\n")
    #                     # endwith
    #                 # endif
    #
    #                 # Note: ========== TEST evaluation ==========
    #                 # if not best model, then evaluate test, otherwise it is already evaluated
    #                 if not if_best_model:
    #                     # test log is saved inside method evaluate
    #                     test_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                           dataloading_processor,
    #                                                                           model,
    #                                                                           mode="test",
    #                                                                           epoch_index=epoch_index,
    #                                                                           if_write_pred_result=True,
    #                                                                           step=global_step)
    #                     print(
    #                         f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
    #                 # endif
    #                 assert test_results_json_dict is not None
    #                 test_auc_score = test_results_json_dict["auc_score"]
    #                 # Note: tensorboard test auc score
    #                 tb_writer_test.add_scalar(
    #                     "test_auc", test_auc_score, global_step)
    #
    #             # endif
    #         # endif
    #
    #         # end of batch
    #         batch_time_length_min = (time.time() - batch_begin_time) * 1.0 / 60
    #         print(
    #             f">>>>>>>>>>>>>>> This batch takes {batch_time_length_min} min")
    #         batch_time_json = {"batch_index": batch_index,
    #                            "time_mins": batch_time_length_min}
    #         with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
    #             fout.write(f"{json.dumps(batch_time_json)}\n")
    #         # endwith
    #     # endfor
    #
    #     # end of epoch
    #     epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
    #     print(
    #         f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
    #     epoch_time_json = {"epoch_index": epoch_index,
    #                        "time_mins": epoch_time_length_min}
    #     with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
    #         fout.write(f"{json.dumps(epoch_time_json)}\n")
    #     # endwith
    # # endfor


def train_contrastive_training_extract_feature_from_database_for_all_relation_among_all_relations_with_e1_and_e2(args,
                                                                                                                 evaluation_processor,
                                                                                                                 dataset_processor,
                                                                                                                 model):
    """
    load pre-generated feature from database for training

    normal example features and novel example features are loaded in each batch
    so that contrastive training is performed.

    Training Strategy
    -------------------
    Each epoch, training relation by relation, each relation should be trained with 10240 instances
    For relation has more or less than 10240 instances, keep iterate the guid
    """
    # load valid and test raw feature list
    valid_example_list = dataset_processor.load_and_cache_examples_with_e1_e2(args, mode="valid")
    valid_feature_list = dataset_processor.load_and_cache_features_with_e1_e2(args, mode="valid")
    test_example_list = dataset_processor.load_and_cache_examples_with_e1_e2(args, mode="test")
    test_feature_list = dataset_processor.load_and_cache_features_with_e1_e2(args, mode="test")
    

    EACH_EPOCH_INSTANCE_NUM = 10240

    # ##### load all normal, novel example guid in database #####
    all_relation_id_to_guid_list_dict = dataset_processor.parallel_load_relation_train_data_guid_dict(
        args)

    # ##### output training data stats ######
    train_stats_file = os.path.join(args.output_dir, "train_data_stats_db.csv")
    dataset_processor.write_train_data_stats(
        all_relation_id_to_guid_list_dict, train_stats_file)

    # set up tensorboard writer
    tb_writer_train = SummaryWriter(os.path.join(args.output_dir, "tb/train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.output_dir, "tb/valid"))
    tb_writer_test = SummaryWriter(os.path.join(args.output_dir, "tb/test"))

    tb_writer_normalized_train = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_train"))
    tb_writer_normalized_valid = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_valid"))
    tb_writer_normalized_test = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_test"))

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.per_gpu_train_batch_size

    batch_num = math.ceil(EACH_EPOCH_INSTANCE_NUM *
                          len(dataset_processor.train_rc_str_list) * 1.0 / batch_size)

    total_optimization_steps = args.num_train_epochs * \
        batch_num * len(dataset_processor.train_rc_str_list)

    print("***** Running training *****")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Num Batch_size = {batch_size}")
    print(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Each relation Num of steps per epoch = {batch_num}")
    print(f"  Total optimization steps = {total_optimization_steps}")

    pytorch_utils.set_seed(args)

    # ############ load resumed checkpoint ###########
    # checkpoint = {"epoch": epoch_index,
    #               "relation_id": relation_id,
    #               "batch_index": batch_index,
    #               "global_step": global_step,
    #               "train_loss": total_train_loss,
    #               "state_dict": model.state_dict(),
    #               "optimizer": optimizer.state_dict()}
    if args.resume_training:
        checkpoint = pytorch_utils.load_checkpoint(
            args, args.resume_checkpoint_dir)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_global_step = checkpoint["global_step"]
        resume_training_loss = checkpoint["train_loss"]

    # train_dataset is a feature list
    global_step = 0
    total_train_loss = 0
    best_eval_score = 0

    best_mean_auc_score = 0

    best_normalized_auc_score = 0

    model.train()
    model.zero_grad()

    for epoch_index in range(args.num_train_epochs):
        epoch_begin_time = time.time()

        # ################## (1) get the guid ids for all relation #################

        all_relation_guid_list = []

        for relation_id in dataset_processor.train_rc_str_list:

            # ########### 1. generate 10240 guid list ###########
            normal_guid_list = all_relation_id_to_guid_list_dict[relation_id]["normal"]
            novel_guid_list = all_relation_id_to_guid_list_dict[relation_id]["novel"]
            guid_list = list(
                set(normal_guid_list).intersection(novel_guid_list))

            # shuffle guid
            random.shuffle(guid_list)

            epoch_guid_list = None
            if len(guid_list) >= EACH_EPOCH_INSTANCE_NUM:
                epoch_guid_list = guid_list[:EACH_EPOCH_INSTANCE_NUM]
            else:
                times_num = math.ceil(EACH_EPOCH_INSTANCE_NUM / len(guid_list))
                new_guid_list = guid_list * times_num
                epoch_guid_list = new_guid_list[:EACH_EPOCH_INSTANCE_NUM]
            # endif
            assert len(epoch_guid_list) == EACH_EPOCH_INSTANCE_NUM

            all_relation_guid_list.extend(epoch_guid_list)
        # endfor

        # ## important !! Must shuffle here
        random.shuffle(all_relation_guid_list)

        # #########################################################################

        # ######### (2) training for all relations ############

        for batch_index in range(batch_num):
            global_step += 1

            # skip until the correct progress
            if args.resume_training:
                total_train_loss = resume_training_loss
                if global_step <= resume_global_step:
                    continue

             # !!! always put model.train() here before each batch training
            model.train()
            model.zero_grad()

            batch_begin_time = time.time()

            batch_guid_list = all_relation_guid_list[batch_size *
                                                     batch_index: batch_size * (batch_index + 1)]

            head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
                tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
                nd_label_id_list, rc_label_id_list = dataset_processor.parallel_load_normal_novel_features_per_batch(
                    batch_guid_list)

            # load on device
            head_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                            head_property_id_list_of_list]
            tail_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                            tail_property_id_list_of_list]

            head_property_value_embeds_matrix_tensor_list = [
                torch.tensor(item, dtype=torch.float).float().to(args.device) for
                item in head_property_value_embed_matrix_list_of_list]

            tail_property_value_embeds_matrix_tensor_list = [
                torch.tensor(item, dtype=torch.float).float().to(args.device) for
                item in tail_property_value_embed_matrix_list_of_list]

            nd_label_id_list = torch.tensor(
                nd_label_id_list, dtype=torch.long).to(args.device)
            # rc_label_id_list = torch.tensor(rc_label_id_list, dtype=torch.long).to(args.device)

            total_score, loss = model(args,
                                      head_property_id_tensor_list,
                                      tail_property_id_tensor_list,
                                      head_property_value_embeds_matrix_tensor_list,
                                      tail_property_value_embeds_matrix_tensor_list,
                                      nd_label_id_list,
                                      rc_label_id_list)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

            total_train_loss += loss.item()

            cur_train_loss = total_train_loss * 1.0 / global_step

            train_info_json = {"epoch": epoch_index,
                               "batch_index": f"{batch_index}/{batch_num}",
                               "global_step": f"{global_step} / {total_optimization_steps}",
                               "train_loss": cur_train_loss}

            print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

            # Note: tensorboard writer
            tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)

            with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
                fout.write(json.dumps(train_info_json) + "\n")
            # endwith

            if (batch_index + 1) % args.logging_steps == 0 and batch_index >= 1:
                if args.evaluate_during_training:

                    # Note: ============= VALID evaluation ==========
                    # valid log is saved inside method evaluate
                
                    
                    # self,
                    # args,
                    # dataset_processor,
                    # model,
                    # mode=None,
                    # eval_example_list=None,
                    # eval_feature_list=None,
                    # epoch_index=None,
                    # step=None
                    result_output_dir = os.path.join(args.output_dir, "eval_results_during_training")
                    os.makedirs(result_output_dir, exist_ok=True)
                    valid_results_json_dict, _, _, _, _, _ = \
                        evaluation_processor.evaluate_with_predicted_rc_relation_label_with_e1_e2(args,
                                                                                                  result_output_dir,
                                                                                                  dataset_processor,
                                                                                                  model,
                                                                                                  mode="valid",
                                                                                                  eval_example_list=valid_example_list,
                                                                                                  eval_feature_list=valid_feature_list,
                                                                                                  epoch_index=epoch_index,
                                                                                                  step=global_step
                                                                                                  )
                    
                    # valid_results_json_dict = evaluate_for_gat_21_relation(args,
                    #                                                        valid_feature_list,
                    #                                                        dataset_processor,
                    #                                                        model,
                    #                                                        mode="valid",
                    #                                                        epoch_index=epoch_index,
                    #                                                        step=global_step)

                    print(
                        f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
                    valid_auc_score = valid_results_json_dict["auc_score"]
                    # Note: tensorboard valid auc score
                    tb_writer_valid.add_scalar(
                        "valid_auc", valid_auc_score, global_step)
                    # ########
                    valid_normalized_auc_score = valid_results_json_dict["normalized_auc_score"]
                    tb_writer_normalized_valid.add_scalar("valid_normalized_auc", valid_normalized_auc_score,
                                                          global_step)

                    # ================ TEST DATA evaluaton =============
                    # test log is saved inside method evaluate
                    test_results_json_dict, _, _, _, _, _ = \
                        evaluation_processor.evaluate_with_predicted_rc_relation_label_with_e1_e2(args,
                                                                                                  result_output_dir,
                                                                                                  dataset_processor,
                                                                                                  model,
                                                                                                  mode="test",
                                                                                                  eval_example_list=test_example_list,
                                                                                                  eval_feature_list=test_feature_list,
                                                                                                  epoch_index=epoch_index,
                                                                                                  step=global_step
                                                                                                  )
                    
                    
                    
                    # test_results_json_dict = evaluate_for_gat_21_relation(args,
                    #                                                       test_feature_list,
                    #                                                       dataset_processor,
                    #                                                       model,
                    #                                                       mode="test",
                    #                                                       epoch_index=epoch_index,
                    #                                                       if_write_pred_result=True,
                    #                                                       step=global_step)

                    print(f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
                    assert test_results_json_dict is not None
                    test_auc_score = test_results_json_dict["auc_score"]
                    # Note: tensorboard test auc score
                    tb_writer_test.add_scalar(
                        "test_auc", test_auc_score, global_step)
                    # #########
                    test_normalized_auc_score = test_results_json_dict["normalized_auc_score"]
                    tb_writer_normalized_test.add_scalar(
                        "test_normalized_auc", test_normalized_auc_score, global_step)

                    # ============= training information need to be saved ==============
                    # {"epoch": 0, "relation_id": "P241", "batch": "39/40", "global_step": "360 / 8400000"}

                    checkpoint = {"epoch": epoch_index,
                                  "batch_index": batch_index,
                                  "global_step": global_step,
                                  "train_loss": total_train_loss,
                                  "state_dict": model.state_dict(),
                                  "optimizer": optimizer.state_dict()}

                    checkpoint_file_path = os.path.join(
                        args.output_dir, args.checkpoint_file_name)
                    torch.save(checkpoint, checkpoint_file_path)

                    # Note: ======== SAVE BEST normalized auc model ===========
                    current_normalized_auc_score = valid_results_json_dict["normalized_auc_score"]

                    if current_normalized_auc_score > best_normalized_auc_score:
                        best_normalized_auc_score = current_normalized_auc_score

                        normalized_sub_dir = os.path.join(
                            args.output_dir, "best_normalized_auc_model")
                        os.makedirs(normalized_sub_dir, exist_ok=True)
                        normalized_checkpoint_file = os.path.join(
                            normalized_sub_dir, args.checkpoint_file_name)
                        shutil.copyfile(src=checkpoint_file_path,
                                        dst=normalized_checkpoint_file)

                        # >>>>>>>>>>> save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_normalized_auc_result.json"),
                                  mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>>>> save the best test information
                        with open(os.path.join(args.output_dir, "best_test_normalized_auc_result.json"),
                                  mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>>>> save both valid / test mean auc
                        with open(os.path.join(args.output_dir, "best_valid_test_result_normalized_auc_log.json"),
                                  mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith
                    # endif

                    # Note: ======== SAVE BEST MEAN AUC MODEL ===========
                    current_mean_auc_score = valid_results_json_dict["mean_of_each_relation_auc"]

                    if current_mean_auc_score > best_mean_auc_score:
                        best_mean_auc_score = current_mean_auc_score

                        mean_sub_dir = os.path.join(
                            args.output_dir, "best_mean_auc_model")
                        os.makedirs(mean_sub_dir, exist_ok=True)
                        mean_checkpoint_file = os.path.join(
                            mean_sub_dir, args.checkpoint_file_name)
                        shutil.copyfile(src=checkpoint_file_path,
                                        dst=mean_checkpoint_file)

                        # >>>>>>>>>>> save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_mean_auc_result.json"),
                                  mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>>>> save the best test information
                        with open(os.path.join(args.output_dir, "best_test_mean_auc_result.json"),
                                  mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>>>> save both valid / test mean auc
                        with open(os.path.join(args.output_dir, "best_valid_test_result_mean_auc_log.json"),
                                  mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith

                    # Note: ================ SAVE THE BEST ===============

                    # save model if the model give the best metrics we care
                    current_eval_score = valid_results_json_dict[args.considered_metrics]

                    if current_eval_score > best_eval_score:
                        best_eval_score = current_eval_score

                        sub_dir = os.path.join(
                            args.output_dir, "best_auc_model")
                        os.makedirs(sub_dir, exist_ok=True)
                        best_checkpoint_file = os.path.join(
                            sub_dir, args.checkpoint_file_name)
                        shutil.copyfile(src=checkpoint_file_path,
                                        dst=best_checkpoint_file)

                        # >>>>>>> save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>> save the best test information, and history information
                        with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>> save both valid test log
                        with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"),
                                  mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith

                # endif
            # end if evaluation

            batch_time_length_sec = (time.time() - batch_begin_time) * 1.0
            print(
                f">>>>>>>>>>>>>>> This batch takes {batch_time_length_sec} sec")
            batch_time_json = {"batch_index": batch_index,
                               "time_sec": batch_time_length_sec}
            with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
                fout.write(f"{json.dumps(batch_time_json)}\n")
            # endwith
            # end all batches

        # end one relation
        # endfor

        # end of epoch
        epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
        print(
            f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
        epoch_time_json = {"epoch_index": epoch_index,
                           "time_mins": epoch_time_length_min}
        with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
            fout.write(f"{json.dumps(epoch_time_json)}\n")
        # endwith


def train_contrastive_training_extract_feature_from_database_for_all_relation_among_all_relations(args,
                                                                                                  dataset_processor,
                                                                                                  model):
    """
    load pre-generated feature from database for training

    normal example features and novel example features are loaded in each batch
    so that contrastive training is performed.

    Training Strategy
    -------------------
    Each epoch, training relation by relation, each relation should be trained with 10240 instances
    For relation has more or less than 10240 instances, keep iterate the guid
    """
    # load valid and test raw feature list
    valid_feature_list = dataset_processor.load_and_cache_features_from_database(
        args, mode="valid")
    test_feature_list = dataset_processor.load_and_cache_features_from_database(
        args, mode="test")

    EACH_EPOCH_INSTANCE_NUM = 10240

    # ##### load all normal, novel example guid in database #####
    all_relation_id_to_guid_list_dict = dataset_processor.parallel_load_relation_train_data_guid_dict(
        args)

    # ##### output training data stats ######
    train_stats_file = os.path.join(args.output_dir, "train_data_stats_db.csv")
    dataset_processor.write_train_data_stats(
        all_relation_id_to_guid_list_dict, train_stats_file)

    # set up tensorboard writer
    tb_writer_train = SummaryWriter(os.path.join(args.output_dir, "tb/train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.output_dir, "tb/valid"))
    tb_writer_test = SummaryWriter(os.path.join(args.output_dir, "tb/test"))

    tb_writer_normalized_train = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_train"))
    tb_writer_normalized_valid = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_valid"))
    tb_writer_normalized_test = SummaryWriter(
        os.path.join(args.output_dir, "tb/normalized_test"))

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.per_gpu_train_batch_size

    batch_num = math.ceil(EACH_EPOCH_INSTANCE_NUM *
                          len(dataset_processor.train_rc_str_list) * 1.0 / batch_size)

    total_optimization_steps = args.num_train_epochs * \
        batch_num * len(dataset_processor.train_rc_str_list)

    print("***** Running training *****")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Num Batch_size = {batch_size}")
    print(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Each relation Num of steps per epoch = {batch_num}")
    print(f"  Total optimization steps = {total_optimization_steps}")

    pytorch_utils.set_seed(args)

    # ############ load resumed checkpoint ###########
    # checkpoint = {"epoch": epoch_index,
    #               "relation_id": relation_id,
    #               "batch_index": batch_index,
    #               "global_step": global_step,
    #               "train_loss": total_train_loss,
    #               "state_dict": model.state_dict(),
    #               "optimizer": optimizer.state_dict()}
    if args.resume_training:
        checkpoint = pytorch_utils.load_checkpoint(
            args, args.resume_checkpoint_dir)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_global_step = checkpoint["global_step"]
        resume_training_loss = checkpoint["train_loss"]

    # train_dataset is a feature list
    global_step = 0
    total_train_loss = 0
    best_eval_score = 0

    best_mean_auc_score = 0

    best_normalized_auc_score = 0

    model.train()
    model.zero_grad()

    for epoch_index in range(args.num_train_epochs):
        epoch_begin_time = time.time()

        # ################## (1) get the guid ids for all relation #################

        all_relation_guid_list = []

        for relation_id in dataset_processor.train_rc_str_list:

            # ########### 1. generate 10240 guid list ###########
            normal_guid_list = all_relation_id_to_guid_list_dict[relation_id]["normal"]
            novel_guid_list = all_relation_id_to_guid_list_dict[relation_id]["novel"]
            guid_list = list(
                set(normal_guid_list).intersection(novel_guid_list))

            # shuffle guid
            random.shuffle(guid_list)

            epoch_guid_list = None
            if len(guid_list) >= EACH_EPOCH_INSTANCE_NUM:
                epoch_guid_list = guid_list[:EACH_EPOCH_INSTANCE_NUM]
            else:
                times_num = math.ceil(EACH_EPOCH_INSTANCE_NUM / len(guid_list))
                new_guid_list = guid_list * times_num
                epoch_guid_list = new_guid_list[:EACH_EPOCH_INSTANCE_NUM]
            # endif
            assert len(epoch_guid_list) == EACH_EPOCH_INSTANCE_NUM

            all_relation_guid_list.extend(epoch_guid_list)
        # endfor

        # ## important !! Must shuffle here
        random.shuffle(all_relation_guid_list)

        # #########################################################################

        # ######### (2) training for all relations ############

        for batch_index in range(batch_num):
            global_step += 1

            # skip until the correct progress
            if args.resume_training:
                total_train_loss = resume_training_loss
                if global_step <= resume_global_step:
                    continue

             # !!! always put model.train() here before each batch training
            model.train()
            model.zero_grad()

            batch_begin_time = time.time()

            batch_guid_list = all_relation_guid_list[batch_size *
                                                     batch_index: batch_size * (batch_index + 1)]

            head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
                tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
                nd_label_id_list, rc_label_id_list = dataset_processor.parallel_load_normal_novel_features_per_batch(
                    batch_guid_list)

            # load on device
            head_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                            head_property_id_list_of_list]
            tail_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                            tail_property_id_list_of_list]

            head_property_value_embeds_matrix_tensor_list = [
                torch.tensor(item, dtype=torch.float).float().to(args.device) for
                item in head_property_value_embed_matrix_list_of_list]

            tail_property_value_embeds_matrix_tensor_list = [
                torch.tensor(item, dtype=torch.float).float().to(args.device) for
                item in tail_property_value_embed_matrix_list_of_list]

            nd_label_id_list = torch.tensor(
                nd_label_id_list, dtype=torch.long).to(args.device)
            # rc_label_id_list = torch.tensor(rc_label_id_list, dtype=torch.long).to(args.device)

            total_score, loss = model(args,
                                      head_property_id_tensor_list,
                                      tail_property_id_tensor_list,
                                      head_property_value_embeds_matrix_tensor_list,
                                      tail_property_value_embeds_matrix_tensor_list,
                                      nd_label_id_list,
                                      rc_label_id_list)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

            total_train_loss += loss.item()

            cur_train_loss = total_train_loss * 1.0 / global_step

            train_info_json = {"epoch": epoch_index,
                               "batch_index": f"{batch_index}/{batch_num}",
                               "global_step": f"{global_step} / {total_optimization_steps}",
                               "train_loss": cur_train_loss}

            print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

            # Note: tensorboard writer
            tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)

            with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
                fout.write(json.dumps(train_info_json) + "\n")
            # endwith

            if (batch_index + 1) % args.logging_steps == 0 and batch_index >= 1:
                if args.evaluate_during_training:

                    # Note: ============= VALID evaluation ==========
                    # valid log is saved inside method evaluate
                    valid_results_json_dict = evaluate_for_gat_21_relation(args,
                                                                           valid_feature_list,
                                                                           dataset_processor,
                                                                           model,
                                                                           mode="valid",
                                                                           epoch_index=epoch_index,
                                                                           step=global_step)

                    print(
                        f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
                    valid_auc_score = valid_results_json_dict["auc_score"]
                    # Note: tensorboard valid auc score
                    tb_writer_valid.add_scalar(
                        "valid_auc", valid_auc_score, global_step)
                    # ########
                    valid_normalized_auc_score = valid_results_json_dict["normalized_auc_score"]
                    tb_writer_normalized_valid.add_scalar("valid_normalized_auc", valid_normalized_auc_score,
                                                          global_step)

                    # ================ TEST DATA evaluaton =============
                    # test log is saved inside method evaluate
                    test_results_json_dict = evaluate_for_gat_21_relation(args,
                                                                          test_feature_list,
                                                                          dataset_processor,
                                                                          model,
                                                                          mode="test",
                                                                          epoch_index=epoch_index,
                                                                          if_write_pred_result=True,
                                                                          step=global_step)

                    print(f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
                    assert test_results_json_dict is not None
                    test_auc_score = test_results_json_dict["auc_score"]
                    # Note: tensorboard test auc score
                    tb_writer_test.add_scalar(
                        "test_auc", test_auc_score, global_step)
                    # #########
                    test_normalized_auc_score = test_results_json_dict["normalized_auc_score"]
                    tb_writer_normalized_test.add_scalar(
                        "test_normalized_auc", test_normalized_auc_score, global_step)

                    # ============= training information need to be saved ==============
                    # {"epoch": 0, "relation_id": "P241", "batch": "39/40", "global_step": "360 / 8400000"}

                    checkpoint = {"epoch": epoch_index,
                                  "batch_index": batch_index,
                                  "global_step": global_step,
                                  "train_loss": total_train_loss,
                                  "state_dict": model.state_dict(),
                                  "optimizer": optimizer.state_dict()}

                    checkpoint_file_path = os.path.join(
                        args.output_dir, args.checkpoint_file_name)
                    torch.save(checkpoint, checkpoint_file_path)

                    # Note: ======== SAVE BEST normalized auc model ===========
                    current_normalized_auc_score = valid_results_json_dict["normalized_auc_score"]

                    if current_normalized_auc_score > best_normalized_auc_score:
                        best_normalized_auc_score = current_normalized_auc_score

                        normalized_sub_dir = os.path.join(
                            args.output_dir, "best_normalized_auc_model")
                        os.makedirs(normalized_sub_dir, exist_ok=True)
                        normalized_checkpoint_file = os.path.join(
                            normalized_sub_dir, args.checkpoint_file_name)
                        shutil.copyfile(src=checkpoint_file_path,
                                        dst=normalized_checkpoint_file)

                        # >>>>>>>>>>> save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_normalized_auc_result.json"),
                                  mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>>>> save the best test information
                        with open(os.path.join(args.output_dir, "best_test_normalized_auc_result.json"),
                                  mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>>>> save both valid / test mean auc
                        with open(os.path.join(args.output_dir, "best_valid_test_result_normalized_auc_log.json"),
                                  mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith
                    # endif

                    # Note: ======== SAVE BEST MEAN AUC MODEL ===========
                    current_mean_auc_score = valid_results_json_dict["mean_of_each_relation_auc"]

                    if current_mean_auc_score > best_mean_auc_score:
                        best_mean_auc_score = current_mean_auc_score

                        mean_sub_dir = os.path.join(
                            args.output_dir, "best_mean_auc_model")
                        os.makedirs(mean_sub_dir, exist_ok=True)
                        mean_checkpoint_file = os.path.join(
                            mean_sub_dir, args.checkpoint_file_name)
                        shutil.copyfile(src=checkpoint_file_path,
                                        dst=mean_checkpoint_file)

                        # >>>>>>>>>>> save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_mean_auc_result.json"),
                                  mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>>>> save the best test information
                        with open(os.path.join(args.output_dir, "best_test_mean_auc_result.json"),
                                  mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>>>> save both valid / test mean auc
                        with open(os.path.join(args.output_dir, "best_valid_test_result_mean_auc_log.json"),
                                  mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith

                    # Note: ================ SAVE THE BEST ===============

                    # save model if the model give the best metrics we care
                    current_eval_score = valid_results_json_dict[args.considered_metrics]

                    if current_eval_score > best_eval_score:
                        best_eval_score = current_eval_score

                        sub_dir = os.path.join(
                            args.output_dir, "best_auc_model")
                        os.makedirs(sub_dir, exist_ok=True)
                        best_checkpoint_file = os.path.join(
                            sub_dir, args.checkpoint_file_name)
                        shutil.copyfile(src=checkpoint_file_path,
                                        dst=best_checkpoint_file)

                        # >>>>>>> save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>> save the best test information, and history information
                        with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        # >>>>>>>>> save both valid test log
                        with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"),
                                  mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith

                # endif
            # end if evaluation

            batch_time_length_sec = (time.time() - batch_begin_time) * 1.0
            print(
                f">>>>>>>>>>>>>>> This batch takes {batch_time_length_sec} sec")
            batch_time_json = {"batch_index": batch_index,
                               "time_sec": batch_time_length_sec}
            with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
                fout.write(f"{json.dumps(batch_time_json)}\n")
            # endwith
            # end all batches

        # end one relation
        # endfor

        # end of epoch
        epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
        print(
            f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
        epoch_time_json = {"epoch_index": epoch_index,
                           "time_mins": epoch_time_length_min}
        with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
            fout.write(f"{json.dumps(epoch_time_json)}\n")
        # endwith

    #     # shuffle!
    #     random.shuffle(train_features)
    #     print("The train dataset is shuffling for epoch {}".format(epoch_index))
    #
    #     for batch_index in range(batch_num):
    #         global_step += 1
    #
    #         print(
    #             f"\n\n\n>>>>>>>>>>>>>> batch[{batch_index}/{batch_num}] -- epoch[{epoch_index}] -- global_step[{global_step}] <<<<<<<<<<<<<<<<<<<<<<")
    #         batch_begin_time = time.time()
    #
    #         model.train()
    #         model.zero_grad()
    #
    #         # (1) dynamically sample negative data examples
    #         # (2) create features for both positive/negative examples
    #         head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
    #         tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
    #         nd_label_id_list, rc_label_id_list = dataloading_processor.dynamically_get_positive_negative_batch_features(
    #             train_features[batch_index * batch_size: (batch_index + 1) * batch_size])
    #
    #         # load on device
    #         head_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
    #                                         head_property_id_list_of_list]
    #         tail_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
    #                                         tail_property_id_list_of_list]
    #
    #         head_property_value_embeds_matrix_tensor_list = [
    #             torch.tensor(item, dtype=torch.float).float().to(args.device) for
    #             item in head_property_value_embed_matrix_list_of_list]
    #
    #         tail_property_value_embeds_matrix_tensor_list = [
    #             torch.tensor(item, dtype=torch.float).float().to(args.device) for
    #             item in tail_property_value_embed_matrix_list_of_list]
    #
    #         nd_label_id_list = torch.tensor(nd_label_id_list, dtype=torch.long).to(args.device)
    #         # rc_label_id_list = torch.tensor(rc_label_id_list, dtype=torch.long).to(args.device)
    #
    #         total_score, loss = model(args,
    #                                   head_property_id_tensor_list,
    #                                   tail_property_id_tensor_list,
    #                                   head_property_value_embeds_matrix_tensor_list,
    #                                   tail_property_value_embeds_matrix_tensor_list,
    #                                   nd_label_id_list,
    #                                   rc_label_id_list)
    #         loss.backward()
    #
    #         # TODO: check this later
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #         optimizer.step()
    #         model.zero_grad()
    #
    #         total_train_loss += loss.item()
    #
    #         cur_train_loss = total_train_loss * 1.0 / global_step
    #
    #         train_info_json = {"epoch": epoch_index, "batch": f"{batch_index}/{batch_num}", "global_step": global_step,
    #                            "train_loss": cur_train_loss}
    #
    #         print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")
    #
    #         # Note: tensorboard writer
    #         tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)
    #
    #         with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
    #             fout.write(json.dumps(train_info_json) + "\n")
    #         # endwith
    #
    #         if global_step % args.logging_steps == 0 and global_step > 5:
    #             if args.evaluate_during_training:
    #
    #                 # Note: ============= VALID evaluation ==========
    #                 # valid log is saved inside method evaluate
    #                 valid_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                        dataloading_processor,
    #                                                                        model,
    #                                                                        mode="valid",
    #                                                                        epoch_index=epoch_index,
    #                                                                        step=global_step)
    #
    #                 print(
    #                     f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
    #                 valid_auc_score = valid_results_json_dict["auc_score"]
    #                 # Note: tensorboard valid auc score
    #                 tb_writer_valid.add_scalar(
    #                     "valid_auc", valid_auc_score, global_step)
    #
    #                 # Note: ========= SAVE THE BEST ==========
    #                 if_best_model = False
    #                 test_results_json_dict = None
    #                 # save model if the model give the best metrics we care
    #                 current_eval_score = valid_results_json_dict[args.considered_metrics]
    #                 if current_eval_score > best_eval_score:
    #                     if_best_model = True
    #                     # save the best model
    #                     best_eval_score = current_eval_score
    #                     subdir = os.path.join(args.output_dir, "best_model")
    #                     if not os.path.exists(subdir):
    #                         os.makedirs(subdir)
    #                     # endif
    #                     pytorch_utils.save_model(model,
    #                                              os.path.join(args.output_dir, "best_model", args.save_model_file_name))
    #
    #                     # save the best eval information
    #                     with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
    #                         fout.write(json.dumps(
    #                             valid_results_json_dict) + "\n")
    #                     # endwith
    #
    #                     # Note: ---------- test on the best model -------------
    #                     # use the current best model to evaluate on test data
    #                     test_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                           dataloading_processor,
    #                                                                           model,
    #                                                                           mode="test",
    #                                                                           epoch_index=epoch_index,
    #                                                                           if_write_pred_result=True,
    #                                                                           step=global_step)
    #                     print(
    #                         f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
    #
    #                     # save the best test information, and history information
    #                     with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
    #                         fout.write(json.dumps(
    #                             test_results_json_dict) + "\n")
    #                     # endwith
    #
    #                     with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"), mode="a") as fout:
    #                         fout.write(
    #                             "valid: " + json.dumps(valid_results_json_dict) + "\n")
    #                         fout.write(
    #                             "test:  " + json.dumps(test_results_json_dict) + "\n")
    #                         fout.write("\n")
    #                     # endwith
    #                 # endif
    #
    #                 # Note: ========== TEST evaluation ==========
    #                 # if not best model, then evaluate test, otherwise it is already evaluated
    #                 if not if_best_model:
    #                     # test log is saved inside method evaluate
    #                     test_results_json_dict = evaluate_for_gat_21_relation(args,
    #                                                                           dataloading_processor,
    #                                                                           model,
    #                                                                           mode="test",
    #                                                                           epoch_index=epoch_index,
    #                                                                           if_write_pred_result=True,
    #                                                                           step=global_step)
    #                     print(
    #                         f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
    #                 # endif
    #                 assert test_results_json_dict is not None
    #                 test_auc_score = test_results_json_dict["auc_score"]
    #                 # Note: tensorboard test auc score
    #                 tb_writer_test.add_scalar(
    #                     "test_auc", test_auc_score, global_step)
    #
    #             # endif
    #         # endif
    #
    #         # end of batch
    #         batch_time_length_min = (time.time() - batch_begin_time) * 1.0 / 60
    #         print(
    #             f">>>>>>>>>>>>>>> This batch takes {batch_time_length_min} min")
    #         batch_time_json = {"batch_index": batch_index,
    #                            "time_mins": batch_time_length_min}
    #         with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
    #             fout.write(f"{json.dumps(batch_time_json)}\n")
    #         # endwith
    #     # endfor
    #
    #     # end of epoch
    #     epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
    #     print(
    #         f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
    #     epoch_time_json = {"epoch_index": epoch_index,
    #                        "time_mins": epoch_time_length_min}
    #     with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
    #         fout.write(f"{json.dumps(epoch_time_json)}\n")
    #     # endwith
    # # endfor


def train_contrastive_entity_pair_for_gat(args, target_relation_id, processor, property_cache, model):
    """
    Train data contains contrastive entity pair
    """
    # load train contrastive dataset
    train_dataset = load_and_cache_examples_for_gat(args,
                                                    processor,
                                                    mode="train")

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # endif

    total_label_list = processor.get_labels()

    # set up tensorboard writer
    tb_writer_train = SummaryWriter(os.path.join(args.output_dir, "tb/train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.output_dir, "tb/valid"))
    tb_writer_test = SummaryWriter(os.path.join(args.output_dir, "tb/test"))

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.per_gpu_train_batch_size
    total_train_size = len(train_dataset)
    batch_num = math.ceil(total_train_size * 1.0 / batch_size)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Num Batch_size = {batch_size}")
    print(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Num of steps per epoch = {batch_num}")
    print(f"  Total optimization steps = {args.num_train_epochs * batch_num}")

    pytorch_utils.set_seed(args)

    # train_dataset is a feature list
    global_step = 0
    total_train_loss = 0
    best_eval_score = 0

    for epoch_index in range(args.num_train_epochs):
        epoch_begin_time = time.time()

        # shuffle
        random.shuffle(train_dataset)

        # for item in train_dataset:
        #     if item.label_id == 0:
        #         print("hello")

        print("The train dataset is shuffling for epoch {}".format(epoch_index))

        for batch_index in range(batch_num):
            global_step += 1

            print(
                f"\n\n\n>>>>>>>>>>>>>> batch[{batch_index}/{batch_num}] -- epoch[{epoch_index}] -- global_step[{global_step}] <<<<<<<<<<<<<<<<<<<<<<")
            batch_begin_time = time.time()

            model.train()
            model.zero_grad()

            subj_property_ids_list, subj_property_value_embeds_matrix_list, \
                obj_property_ids_list, obj_property_value_embeds_matrix_list, \
                label_id_list = \
                processor.get_model_input(
                    train_dataset[batch_index * batch_size: (batch_index + 1) * batch_size])

            # load on device
            subj_property_ids_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                      subj_property_ids_list]
            obj_property_ids_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                     obj_property_ids_list]
            subj_property_value_embeds_matrix_list = [torch.tensor(item, dtype=torch.float).to(args.device) for item in
                                                      subj_property_value_embeds_matrix_list]
            obj_property_value_embeds_matrix_list = [torch.tensor(item, dtype=torch.float).to(args.device) for item in
                                                     obj_property_value_embeds_matrix_list]
            label_id_list = torch.tensor(
                label_id_list, dtype=torch.long).to(args.device)

            total_score, loss = model(args,
                                      target_relation_id,
                                      subj_property_ids_list,
                                      obj_property_ids_list,
                                      subj_property_value_embeds_matrix_list,
                                      obj_property_value_embeds_matrix_list,
                                      label_id_list)
            loss.backward()

            # TODO: check this later
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

            total_train_loss += loss.item()

            cur_train_loss = total_train_loss * 1.0 / global_step
            train_info_json = {"epoch": epoch_index, "batch": f"{batch_index}/{batch_num}", "global_step": global_step,
                               "train_loss": cur_train_loss}
            print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

            # Note: tensorboard writer
            tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)

            with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
                fout.write(json.dumps(train_info_json) + "\n")
            # endwith

            if global_step % args.logging_steps == 0:
                if args.evaluate_during_training:

                    # Note: ============= VALID evaluation ==========
                    # valid log is saved inside method evaluate
                    valid_results_json_dict = evaluate_for_gat(args, property_cache, target_relation_id, processor,
                                                               model,
                                                               mode="valid", epoch_index=epoch_index, step=global_step)
                    print(
                        f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
                    valid_auc_score = valid_results_json_dict["auc_score"]
                    # Note: tensorboard valid auc score
                    tb_writer_valid.add_scalar(
                        "valid_auc", valid_auc_score, global_step)

                    # Note: ========= SAVE THE BEST ==========
                    if_best_model = False
                    test_results_json_dict = None
                    # save model if the model give the best metrics we care
                    current_eval_score = valid_results_json_dict[args.considered_metrics]
                    if current_eval_score > best_eval_score:
                        if_best_model = True
                        # save the best model
                        best_eval_score = current_eval_score
                        subdir = os.path.join(args.output_dir, "best_model")
                        if not os.path.exists(subdir):
                            os.makedirs(subdir)
                        # endif
                        pytorch_utils.save_model(model,
                                                 os.path.join(args.output_dir, "best_model", args.save_model_file_name))

                        # save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # Note: ---------- test on the best model -------------
                        # use the current best model to evaluate on test data
                        test_results_json_dict = evaluate_for_gat(args, property_cache, target_relation_id, processor,
                                                                  model,
                                                                  mode="test", if_write_pred_result=True,
                                                                  epoch_index=epoch_index, step=global_step)
                        print(
                            f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")

                        # save the best test information, and history information
                        with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"), mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith
                    # endif

                    # Note: ========== TEST evaluation ==========
                    # if not best model, then evaluate test, otherwise it is already evaluated
                    if not if_best_model:
                        # test log is saved inside method evaluate
                        test_results_json_dict = evaluate_for_gat(args, property_cache, target_relation_id, processor,
                                                                  model,
                                                                  mode="test", if_write_pred_result=True,
                                                                  epoch_index=epoch_index, step=global_step)
                        print(
                            f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
                    # endif
                    assert test_results_json_dict is not None
                    test_auc_score = test_results_json_dict["auc_score"]
                    # Note: tensorboard test auc score
                    tb_writer_test.add_scalar(
                        "test_auc", test_auc_score, global_step)

                # endif
            # endif

            # end of batch
            batch_time_length_min = (time.time() - batch_begin_time) * 1.0 / 60
            print(
                f">>>>>>>>>>>>>>> This batch takes {batch_time_length_min} min")
            batch_time_json = {"batch_index": batch_index,
                               "time_mins": batch_time_length_min}
            with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
                fout.write(f"{json.dumps(batch_time_json)}\n")
            # endwith
        # endfor

        # end of epoch
        epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
        print(
            f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
        epoch_time_json = {"epoch_index": epoch_index,
                           "time_mins": epoch_time_length_min}
        with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
            fout.write(f"{json.dumps(epoch_time_json)}\n")
        # endwith
    # endfor


def train_contrastive_entity_pair(args, processor, model):
    """
    Train data contains contrastive entity pair
    """
    # load train contrastive dataset
    train_dataset = load_and_cache_examples(args,
                                            processor,
                                            mode="train_contrastive")

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # endif

    total_label_list = processor.get_labels()

    # set up tensorboard writer
    tb_writer_train = SummaryWriter(os.path.join(args.output_dir, "tb/train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.output_dir, "tb/valid"))
    tb_writer_test = SummaryWriter(os.path.join(args.output_dir, "tb/test"))

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.per_gpu_train_batch_size
    total_train_size = len(train_dataset)
    batch_num = math.ceil(total_train_size * 1.0 / batch_size)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Num Batch_size = {batch_size}")
    print(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Num of steps per epoch = {batch_num}")
    print(f"  Total optimization steps = {args.num_train_epochs * batch_num}")

    pytorch_utils.set_seed(args)

    # train_dataset is a feature list
    global_step = 0
    total_train_loss = 0
    best_eval_score = 0

    for epoch_index in range(args.num_train_epochs):
        epoch_begin_time = time.time()

        # shuffle
        random.shuffle(train_dataset)

        # for item in train_dataset:
        #     if item.label_id == 0:
        #         print("hello")

        print("The train dataset is shuffling for epoch {}".format(epoch_index))

        for batch_index in range(batch_num):
            global_step += 1

            print(
                f"\n\n\n>>>>>>>>>>>>>> batch[{batch_index}/{batch_num}] -- epoch[{epoch_index}] -- global_step[{global_step}] <<<<<<<<<<<<<<<<<<<<<<")
            batch_begin_time = time.time()

            model.train()
            model.zero_grad()

            subj_embed_matrix, obj_embed_matrix, label_id_list = \
                processor.get_model_input(
                    train_dataset[batch_index * batch_size: (batch_index + 1) * batch_size])

            # load on device
            subj_embed_matrix = torch.tensor(
                subj_embed_matrix, dtype=torch.float).to(args.device)
            obj_embed_matrix = torch.tensor(
                obj_embed_matrix, dtype=torch.float).to(args.device)
            label_id_list = torch.tensor(
                label_id_list, dtype=torch.long).to(args.device)

            total_score, loss = model(
                args, subj_embed_matrix, obj_embed_matrix, label_id_list)
            loss.backward()

            # TODO: check this later
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

            total_train_loss += loss.item()

            cur_train_loss = total_train_loss * 1.0 / global_step
            train_info_json = {"epoch": epoch_index, "batch": f"{batch_index}/{batch_num}", "global_step": global_step,
                               "train_loss": cur_train_loss}
            print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

            # Note: tensorboard writer
            tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)

            with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
                fout.write(json.dumps(train_info_json) + "\n")
            # endwith

            if global_step % args.logging_steps == 0:
                if args.evaluate_during_training:

                    # Note: ============= VALID evaluation ==========
                    # valid log is saved inside method evaluate
                    valid_results_json_dict = evaluate(args, processor, model,
                                                       mode="valid", epoch_index=epoch_index, step=global_step)
                    print(
                        f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
                    valid_auc_score = valid_results_json_dict["auc_score"]
                    # Note: tensorboard valid auc score
                    tb_writer_valid.add_scalar(
                        "valid_auc", valid_auc_score, global_step)

                    # Note: ========= SAVE THE BEST ==========
                    if_best_model = False
                    test_results_json_dict = None
                    # save model if the model give the best metrics we care
                    current_eval_score = valid_results_json_dict[args.considered_metrics]
                    if current_eval_score > best_eval_score:
                        if_best_model = True
                        # save the best model
                        best_eval_score = current_eval_score
                        subdir = os.path.join(args.output_dir, "best_model")
                        if not os.path.exists(subdir):
                            os.makedirs(subdir)
                        # endif
                        pytorch_utils.save_model(model,
                                                 os.path.join(args.output_dir, "best_model", args.save_model_file_name))

                        # save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # Note: ---------- test on the best model -------------
                        # use the current best model to evaluate on test data
                        test_results_json_dict = evaluate(args, processor, model,
                                                          mode="test", if_write_pred_result=True,
                                                          epoch_index=epoch_index, step=global_step)
                        print(
                            f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")

                        # save the best test information, and history information
                        with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"), mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith
                    # endif

                    # Note: ========== TEST evaluation ==========
                    # if not best model, then evaluate test, otherwise it is already evaluated
                    if not if_best_model:
                        # test log is saved inside method evaluate
                        test_results_json_dict = evaluate(args, processor, model,
                                                          mode="test", if_write_pred_result=True,
                                                          epoch_index=epoch_index, step=global_step)
                        print(
                            f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
                    # endif
                    assert test_results_json_dict is not None
                    test_auc_score = test_results_json_dict["auc_score"]
                    # Note: tensorboard test auc score
                    tb_writer_test.add_scalar(
                        "test_auc", test_auc_score, global_step)

                # endif
            # endif

            # end of batch
            batch_time_length_min = (time.time() - batch_begin_time) * 1.0 / 60
            print(
                f">>>>>>>>>>>>>>> This batch takes {batch_time_length_min} min")
            batch_time_json = {"batch_index": batch_index,
                               "time_mins": batch_time_length_min}
            with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
                fout.write(f"{json.dumps(batch_time_json)}\n")
            # endwith
        # endfor

        # end of epoch
        epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
        print(
            f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
        epoch_time_json = {"epoch_index": epoch_index,
                           "time_mins": epoch_time_length_min}
        with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
            fout.write(f"{json.dumps(epoch_time_json)}\n")
        # endwith
    # endfor


def train_dynamic_sampled_GAT(args, model_class, tokenizer_class, model, processor):
    """ Train the model
    """
    train_dataset = processor.get_train_examples()
    total_label_list = processor.get_labels()

    # set up tensorboard writer
    tb_writer_train = SummaryWriter(os.path.join(args.output_dir, "tb/train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.output_dir, "tb/valid"))
    tb_writer_test = SummaryWriter(os.path.join(args.output_dir, "tb/test"))

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.per_gpu_train_batch_size
    total_train_size = len(train_dataset)
    batch_num = math.ceil(total_train_size * 1.0 / batch_size)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Num Batch_size = {batch_size}")
    print(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Num of steps per epoch = {batch_num}")
    print(f"  Total optimization steps = {args.num_train_epochs * batch_num}")

    pytorch_utils.set_seed(args)

    # train_dataset is a feature list
    global_step = 0
    total_train_loss = 0
    best_eval_score = 0

    # Note: ----------------------------------------------------------------------------------------------------------
    #       if there are some synset sampling list is not precompute, use following code to precompute
    #       and save into file
    all_synset_set = processor.get_entity_in_training_data_sampling_negative_example_dictionary(
        train_dataset)
    print(
        f"There are totally {len(all_synset_set)} synset in the training data")
    # Note: ----------------------------------------------------------------------------------------------------------

    for epoch_index in range(args.num_train_epochs):
        epoch_begin_time = time.time()

        # shuffle
        random.shuffle(train_dataset)
        print("The train dataset is shuffling for epoch {}".format(epoch_index))

        for batch_index in range(batch_num):
            global_step += 1

            print(
                f"\n\n\n>>>>>>>>>>>>>> batch[{batch_index}/{batch_num}] -- epoch[{epoch_index}] -- global_step[{global_step}] <<<<<<<<<<<<<<<<<<<<<<")
            batch_begin_time = time.time()

            model.train()
            model.zero_grad()

            cur_batch_examples = train_dataset[batch_index *
                                               batch_size: (batch_index + 1) * batch_size]

            total_model_input = processor.dynamically_get_positive_and_negative_batch_without_online_parsing(args,
                                                                                                             cur_batch_examples)

            word_embed_matrix, graph_edge_list, target_mask_list, label_id_list, input_token_size_list = total_model_input

            # load on device
            word_embed_matrix = torch.tensor(
                word_embed_matrix, dtype=torch.float).to(args.device)
            # graph_edge_list = torch.tensor(np.array(graph_edge_list).transpose(), dtype=torch.long).to(args.device)
            graph_edge_list = torch.tensor(np.array(graph_edge_list), dtype=torch.long).t().contiguous().to(
                args.device)
            label_id_list = torch.tensor(
                label_id_list, dtype=torch.long).to(args.device)

            total_score, loss = model(
                args, word_embed_matrix, target_mask_list, graph_edge_list, label_id_list)
            loss.backward()

            # TODO: check this later
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

            total_train_loss += loss.item()

            cur_train_loss = total_train_loss * 1.0 / global_step
            train_info_json = {"epoch": epoch_index, "batch": f"{batch_index}/{batch_num}", "global_step": global_step,
                               "train_loss": cur_train_loss}
            print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

            # Note: tensorboard writer
            tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)

            with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
                fout.write(json.dumps(train_info_json) + "\n")
            # endwith

            if global_step % args.logging_steps == 0:
                if args.evaluate_during_training:

                    # Note: ============= VALID evaluation ==========
                    # valid log is saved inside method evaluate
                    valid_results_json_dict = evaluate(args, processor, model_class, tokenizer_class, model,
                                                       mode="valid", epoch_index=epoch_index, step=global_step)
                    print(
                        f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
                    valid_auc_score = valid_results_json_dict["auc_score"]
                    # Note: tensorboard valid auc score
                    tb_writer_valid.add_scalar(
                        "valid_auc", valid_auc_score, global_step)

                    # Note: ========= SAVE THE BEST ==========
                    if_best_model = False
                    test_results_json_dict = None
                    # save model if the model give the best metrics we care
                    current_eval_score = valid_results_json_dict[args.considered_metrics]
                    if current_eval_score > best_eval_score:
                        if_best_model = True
                        # save the best model
                        best_eval_score = current_eval_score
                        subdir = os.path.join(args.output_dir, "best_model")
                        if not os.path.exists(subdir):
                            os.makedirs(subdir)
                        # endif
                        pytorch_utils.save_model(model,
                                                 os.path.join(args.output_dir, "best_model", args.save_model_file_name))

                        # save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                valid_results_json_dict) + "\n")
                        # endwith

                        # Note: ---------- test on the best model -------------
                        # use the current best model to evaluate on test data
                        test_results_json_dict = evaluate(args, processor, model_class, tokenizer_class, model,
                                                          mode="test", if_write_pred_result=True,
                                                          epoch_index=epoch_index, step=global_step)
                        print(
                            f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")

                        # save the best test information, and history information
                        with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
                            fout.write(json.dumps(
                                test_results_json_dict) + "\n")
                        # endwith

                        with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"), mode="a") as fout:
                            fout.write(
                                "valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write(
                                "test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith
                    # endif

                    # Note: ========== TEST evaluation ==========
                    # if not best model, then evaluate test, otherwise it is already evaluated
                    if not if_best_model:
                        # test log is saved inside method evaluate
                        test_results_json_dict = evaluate(args, processor, model_class, tokenizer_class, model,
                                                          mode="test", if_write_pred_result=True,
                                                          epoch_index=epoch_index, step=global_step)
                        print(
                            f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
                    # endif
                    assert test_results_json_dict is not None
                    test_auc_score = test_results_json_dict["auc_score"]
                    # Note: tensorboard test auc score
                    tb_writer_test.add_scalar(
                        "test_auc", test_auc_score, global_step)

                # endif
            # endif

            # end of batch
            batch_time_length_min = (time.time() - batch_begin_time) * 1.0 / 60
            print(
                f">>>>>>>>>>>>>>> This batch takes {batch_time_length_min} min")
            batch_time_json = {"batch_index": batch_index,
                               "time_mins": batch_time_length_min}
            with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
                fout.write(f"{json.dumps(batch_time_json)}\n")
            # endwith
        # endfor

        # end of epoch
        epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
        print(
            f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
        epoch_time_json = {"epoch_index": epoch_index,
                           "time_mins": epoch_time_length_min}
        with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
            fout.write(f"{json.dumps(epoch_time_json)}\n")
        # endwith


def evaluate_unsupervised(args, processor, model, mode=None,
                          if_write_pred_result=False,
                          epoch_index=None,
                          step=None,
                          output_file=None):
    assert epoch_index is not None
    assert mode in ["train", "valid", "test"]

    # eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}

    # eval dataset could be train, dev, test
    eval_dataset = load_and_cache_examples(args,
                                           processor,
                                           mode=mode)  # mode decides which dataset to evaluate

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = args.per_gpu_eval_batch_size
    total_eval_size = len(eval_dataset)
    batch_num = math.ceil(total_eval_size * 1.0 / eval_batch_size)

    # Eval!
    print(f"***** Running evaluation for {mode} dataset*****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.per_gpu_eval_batch_size}")
    print(f"  Batch num per epoch = {batch_num}")

    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    all_score_list = []
    all_label_list = []
    for batch_index in tqdm(range(batch_num), desc="Evaluating"):
        model.eval()

        subj_embed_matrix, obj_embed_matrix, label_id_list = \
            processor.get_model_input(
                eval_dataset[batch_index * eval_batch_size: (batch_index + 1) * eval_batch_size])

        # load on device
        subj_embed_matrix = torch.tensor(
            subj_embed_matrix, dtype=torch.float).to(args.device)
        obj_embed_matrix = torch.tensor(
            obj_embed_matrix, dtype=torch.float).to(args.device)
        # label_id_list = torch.tensor(label_id_list, dtype=torch.long).to(args.device)

        with torch.no_grad():
            score_list, loss = model(
                args, subj_embed_matrix, obj_embed_matrix, label_id_list=None)
            all_score_list.extend(score_list)
            all_label_list.extend(label_id_list)
        # endwith

        # nb_eval_steps += 1
        # if preds is None:
        #     preds = total_score.detach().cpu().numpy()
        #     out_label_ids = label_id_list.detach().cpu().numpy()
        # else:
        #     preds = np.append(preds, total_score.detach().cpu().numpy(), axis=0)
        #     out_label_ids = np.append(out_label_ids, label_id_list.detach().cpu().numpy(), axis=0)
        # # endif
    # endfor

    # preds = np.argmax(preds, axis=1)

    auc_score = evaluation_metrics.get_auc_score(
        all_label_list, all_score_list)  # roc_auc_score(y_true, pred_score)
    results["auc_score"] = auc_score

    if output_file is None:
        output_eval_file = os.path.join(eval_output_dir, mode + "_results.txt")
    else:
        output_eval_file = output_file
    # endif

    with open(output_eval_file, "a") as writer:
        output_json = {'epoch': epoch_index,
                       'step': step,
                       'mode': mode}
        output_json.update(results)
        # print("{}".format(output_json))
        writer.write(json.dumps(output_json) + '\n')
    # endwith

    # Note: write test prediction file
    # if if_write_pred_result and mode == "test":
    #     write_test_prediction(args, preds)
    # # endif
    return output_json


def evaluate_for_gat_21_relation(args,
                                 raw_feature_list,
                                 dataset_processor,
                                 model,
                                 mode=None,
                                 if_write_pred_result=False,
                                 epoch_index=None,
                                 step=None,
                                 output_file=None):
    assert epoch_index is not None
    assert mode in ["train", "valid", "test"]

    # eval_task = args.task_name
    eval_output_dir = args.output_dir

    # eval dataset could be train, dev, test
    # mode decides which dataset to evaluate
    # eval_dataset = load_and_cache_features_from_database(args,
    #                                                      dataloading_processor,
    #                                                      mode=mode)
    eval_dataset = raw_feature_list

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = args.per_gpu_eval_batch_size
    total_eval_size = len(eval_dataset)
    batch_num = math.ceil(total_eval_size * 1.0 / eval_batch_size)

    # Eval!
    print(f"***** Running evaluation for {mode} dataset*****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.per_gpu_eval_batch_size}")
    print(f"  Batch num per epoch = {batch_num}")

    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    data_guid_list = []

    for batch_index in tqdm(range(batch_num), desc="Evaluating"):
        model.eval()

        eval_dataset_batch = eval_dataset[batch_index *
                                          eval_batch_size: (batch_index + 1) * eval_batch_size]

        for item in eval_dataset_batch:
            data_guid_list.append(item.guid)
        # endfor

        head_property_id_list_of_list, head_property_value_embed_matrix_list_of_list, \
            tail_property_id_list_of_list, tail_property_value_embed_matrix_list_of_list, \
            nd_label_id_list, rc_label_id_list = dataset_processor.get_model_input(
                eval_dataset_batch)

        # load on device
        head_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                        head_property_id_list_of_list]
        tail_property_id_tensor_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                        tail_property_id_list_of_list]

        head_property_value_embeds_matrix_tensor_list = [torch.tensor(item, dtype=torch.float).to(args.device) for item
                                                         in head_property_value_embed_matrix_list_of_list]
        tail_property_value_embeds_matrix_tensor_list = [torch.tensor(item, dtype=torch.float).to(args.device) for item
                                                         in tail_property_value_embed_matrix_list_of_list]

        nd_label_id_list = torch.tensor(
            nd_label_id_list, dtype=torch.long).to(args.device)
        # do not make rc_label_id_list tensor
        # rc_label_id_list = torch.tensor(rc_label_id_list, dtype=torch.long).to(args.device)

        with torch.no_grad():
            total_score, loss = model(args,
                                      head_property_id_tensor_list,
                                      tail_property_id_tensor_list,
                                      head_property_value_embeds_matrix_tensor_list,
                                      tail_property_value_embeds_matrix_tensor_list,
                                      None,
                                      rc_label_id_list)
        # endwith

        nb_eval_steps += 1
        if preds is None:
            preds = total_score.detach().cpu().numpy()
            out_label_ids = nd_label_id_list.detach().cpu().numpy()
        else:
            preds = np.append(
                preds, total_score.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, nd_label_id_list.detach().cpu().numpy(), axis=0)
        # endif
    # endfor

    if args.model_type in {"RGAT", "RGAT_Stack"}:
        # classification
        preds = np.argmax(preds, axis=1)

    if args.model_type in ["RGAT_MaxMargin", "RGAT_MaxMargin_Stack", "GAT_MaxMargin", "RGAT_MaxMargin_Stack_With_Head_Tail_Attention"]:
        # use preds directly as score
        pass

    # ########### evaluation cross all relation #######

    result_dict = {}
    auc_score = evaluation_metrics.get_auc_score(
        out_label_ids, preds)  # roc_auc_score(y_true, pred_score)
    result_dict["auc_score"] = auc_score

    # ########### evaluation result by relation #########
    relation_to_auc_score_dict = {}
    pred_list = preds.tolist()
    out_label_ids_list = out_label_ids.tolist()
    assert len(pred_list) == len(out_label_ids_list)

    all_relation_normalized_score_dict = {}

    for relation_id in dataset_processor.evaluation_rc_str_list:
        cur_guid_pred_list = []
        cur_pred_list = []
        cur_label_list = []
        for i in range(len(pred_list)):
            guid = data_guid_list[i]
            cur_relation_id = guid.split("-")[0]
            if cur_relation_id == relation_id:
                # guid: pred_score pair
                cur_guid_pred_list.append((guid, pred_list[i]))
                cur_pred_list.append(pred_list[i])
                cur_label_list.append(out_label_ids_list[i])
            # endif
        # endfor
        assert len(cur_pred_list) == 100
        cur_pred_arr = np.array(cur_pred_list)
        cur_label_arr = np.array(cur_label_list)

        # normalized cur_pre_arr into 0 and 1
        pred_min = np.amin(cur_pred_arr)
        pred_max = np.amax(cur_pred_arr)
        for tmp_guid, tmp_pred_score in cur_guid_pred_list:

            if args.model_type in ["RGAT_MaxMargin", "RGAT_MaxMargin_Stack", "GAT_MaxMargin", "RGAT_MaxMargin_Stack_With_Head_Tail_Attention"]:
                tmp_normalized_pred_score = (
                    tmp_pred_score - pred_min) / (pred_max - pred_min)
            else:
                tmp_normalized_pred_score = tmp_pred_score
            # endif

            all_relation_normalized_score_dict[tmp_guid] = tmp_normalized_pred_score
        # endfor

        # roc_auc_score(y_true, pred_score)
        cur_auc_score = evaluation_metrics.get_auc_score(
            cur_label_arr, cur_pred_arr)

        relation_to_auc_score_dict[relation_id] = cur_auc_score
    # endfor

    # calculate normalized among all relation auc score
    all_relation_normalized_pred_score_list = []
    for tmp_guid in data_guid_list:
        all_relation_normalized_pred_score_list.append(
            all_relation_normalized_score_dict[tmp_guid])
    # endfor
    all_relation_normalized_pred_score_arr = np.array(
        all_relation_normalized_pred_score_list)

    normalized_auc = evaluation_metrics.get_auc_score(
        out_label_ids, all_relation_normalized_pred_score_arr)
    result_dict["normalized_auc_score"] = normalized_auc

    # calculate avg auc score
    all_auc_scores = np.array(list(relation_to_auc_score_dict.values()))
    mean_auc_score = np.mean(all_auc_scores)
    relation_to_auc_score_dict["mean_of_each_relation_auc"] = mean_auc_score

    train_info_json = {'epoch': epoch_index,
                       'step': step,
                       'mode': mode}
    relation_to_auc_score_dict.update(train_info_json)

    # ----------- output relation to auc score dict -----------
    relation_auc_json_file_path = os.path.join(
        args.output_dir, f"relation_auc_score_dict_{mode}.json")
    with open(relation_auc_json_file_path, mode="a") as fout:
        fout.write(f"{json.dumps(relation_to_auc_score_dict)}\n")
    # endwith

    # ---------- output relation in a table format ------------
    relation_auc_json_file_path_in_tabel = os.path.join(
        args.output_dir, f"relation_aut_score_dict_in_table_{mode}.txt")
    with open(relation_auc_json_file_path_in_tabel, mode="a") as fout:
        fout.write(
            f"------------- epoch: {epoch_index} - step: {step} - mode: {mode} -------------\n")
        for k, v in relation_to_auc_score_dict.items():
            fout.write(f"{k:<10}\t{v:<10}\n")
        # endfor
        fout.write("*" * 30 + "\n\n")
    # endwith

    # ----------- output to valid/test_result --------
    if output_file is None:
        output_eval_file = os.path.join(eval_output_dir, mode + "_results.txt")
    else:
        output_eval_file = output_file
    # endif

    result_dict["mean_of_each_relation_auc"] = mean_auc_score
    with open(output_eval_file, "a") as writer:
        output_json = {'epoch': epoch_index,
                       'step': step,
                       'mode': mode}
        output_json.update(result_dict)
        # print("{}".format(output_json))
        writer.write(json.dumps(output_json) + '\n')
    # endwith

    # Note: write test prediction file
    # if if_write_pred_result and mode == "test":
    #     write_test_prediction(args, preds)
    # # endif
    return output_json


def evaluate_for_gat_for_interpretation(args, property_cache, target_relation_id, processor, model, mode=None,
                                        if_write_pred_result=False,
                                        epoch_index=None,
                                        step=None,
                                        output_file=None):
    assert epoch_index is not None
    assert mode in ["train", "valid", "test"]

    # eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}

    # eval dataset could be train, dev, test
    eval_dataset = load_and_cache_examples_for_gat(args,
                                                   processor,
                                                   mode=mode)  # mode decides which dataset to evaluate

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = args.per_gpu_eval_batch_size
    total_eval_size = len(eval_dataset)
    batch_num = math.ceil(total_eval_size * 1.0 / eval_batch_size)

    # Eval!
    print(f"***** Running evaluation for {mode} dataset*****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.per_gpu_eval_batch_size}")
    print(f"  Batch num per epoch = {batch_num}")

    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    all_subj_attention_pro_list = []
    all_obj_attention_pro_list = []
    for batch_index in tqdm(range(batch_num), desc="Evaluating"):
        model.eval()

        subj_property_ids_list, subj_property_value_embeds_matrix_list, \
            obj_property_ids_list, obj_property_value_embeds_matrix_list, \
            label_id_list = \
            processor.get_model_input(
                eval_dataset[batch_index * eval_batch_size: (batch_index + 1) * eval_batch_size])

        # load on device
        subj_property_ids_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                  subj_property_ids_list]
        obj_property_ids_list = [torch.tensor(item, dtype=torch.long).to(args.device) for item in
                                 obj_property_ids_list]
        subj_property_value_embeds_matrix_list = [torch.tensor(item, dtype=torch.float).to(args.device) for item in
                                                  subj_property_value_embeds_matrix_list]
        obj_property_value_embeds_matrix_list = [torch.tensor(item, dtype=torch.float).to(args.device) for item in
                                                 obj_property_value_embeds_matrix_list]
        label_id_list = torch.tensor(
            label_id_list, dtype=torch.long).to(args.device)

        # subj_embed_matrix, obj_embed_matrix, label_id_list = \
        #     processor.get_model_input(eval_dataset[batch_index * eval_batch_size: (batch_index + 1) * eval_batch_size])
        #
        # # load on device
        # subj_embed_matrix = torch.tensor(subj_embed_matrix, dtype=torch.float).to(args.device)
        # obj_embed_matrix = torch.tensor(obj_embed_matrix, dtype=torch.float).to(args.device)
        # label_id_list = torch.tensor(label_id_list, dtype=torch.long).to(args.device)

        with torch.no_grad():
            total_score, loss, subj_attention_pro_mean_list, obj_attention_pro_mean_list = model(args,
                                                                                                 target_relation_id,
                                                                                                 subj_property_ids_list,
                                                                                                 obj_property_ids_list,
                                                                                                 subj_property_value_embeds_matrix_list,
                                                                                                 obj_property_value_embeds_matrix_list,
                                                                                                 label_id_list)
            all_subj_attention_pro_list.extend(subj_attention_pro_mean_list)
            all_obj_attention_pro_list.extend(obj_attention_pro_mean_list)
            # total_score, loss = model(args, subj_embed_matrix, obj_embed_matrix, label_id_list=None)
        # endwith

        nb_eval_steps += 1
        if preds is None:
            preds = total_score.detach().cpu().numpy()
            out_label_ids = label_id_list.detach().cpu().numpy()
        else:
            preds = np.append(
                preds, total_score.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_id_list.detach().cpu().numpy(), axis=0)
        # endif
    # endfor

    preds = np.argmax(preds, axis=1)
    auc_score = evaluation_metrics.get_auc_score(
        out_label_ids, preds)  # roc_auc_score(y_true, pred_score)
    results["auc_score"] = auc_score

    if output_file is None:
        output_eval_file = os.path.join(eval_output_dir, mode + "_results.txt")
    else:
        output_eval_file = output_file
    # endif

    with open(output_eval_file, "a") as writer:
        output_json = {'epoch': epoch_index,
                       'step': step,
                       'mode': mode}
        output_json.update(results)
        # print("{}".format(output_json))
        writer.write(json.dumps(output_json) + '\n')
    # endwith

    # ================ output attention information =============
    data_example_list = None
    if mode == "valid":
        data_example_list = processor.load_valid_examples()
    # endif
    if mode == "test":
        data_example_list = processor.load_test_examples()
    # endif

    if step % 50 == 0 or step in {0, 1, 2, 3, 4, 5}:
        attention_info_file = os.path.join(
            eval_output_dir, f"{mode}_attention_info.txt")
        with open(attention_info_file, mode="w") as fout:
            fout.write(f"----> step: {step} <----\n")

            # data_example_list have the same size and order
            # as all_subj_attention_pro_list, all_obj_attention_pro_list
            for i in range(len(data_example_list)):
                entity_pair_exp = data_example_list[i]
                label_text = entity_pair_exp.label
                subj_entity = entity_pair_exp.subj_wikidata_entity_obj
                obj_entity = entity_pair_exp.obj_wikidata_entity_obj
                # -----------
                subj_entity_id = subj_entity.wikidata_id
                subj_entity_label = subj_entity.label
                subj_entity_des = subj_entity.description
                subj_property_to_value_pair_list = subj_entity.property_to_value_pair_list
                subj_property_value_attention_triple_list = []
                for j, (property_id, value_text_list) in enumerate(subj_property_to_value_pair_list):
                    subj_property_value_attention_triple_list.append((property_id,
                                                                      value_text_list,
                                                                      all_subj_attention_pro_list[i][j]))
                # -----------
                obj_entity_id = obj_entity.wikidata_id
                obj_entity_label = obj_entity.label
                obj_entity_des = obj_entity.description
                obj_property_to_value_pair_list = obj_entity.property_to_value_pair_list
                obj_property_value_attention_triple_list = []
                for k, (property_id, value_text_list) in enumerate(obj_property_to_value_pair_list):
                    obj_property_value_attention_triple_list.append((property_id,
                                                                     value_text_list,
                                                                     all_obj_attention_pro_list[i][k]))

                # =================
                fout.write(f"------------ label: {label_text}--------\n")
                fout.write(
                    f"subj: {subj_entity_id} || {subj_entity_label} || {subj_entity_des}\n")
                for property_id, value_text, prob in sorted(subj_property_value_attention_triple_list,
                                                            key=lambda x: x[2], reverse=True):
                    fout.write(
                        f"{prob}\t{value_text}\t{property_id}\t{property_cache.get_property_label_and_des(property_id)}\n")
                # endfor
                fout.write(f"-------------------------------\n")
                fout.write(
                    f"obj: {obj_entity_id} || {obj_entity_label} || {obj_entity_des}\n")
                for property_id, value_text, prob in sorted(obj_property_value_attention_triple_list,
                                                            key=lambda x: x[2], reverse=True):
                    fout.write(
                        f"{prob}\t{value_text}\t{property_id}\t{property_cache.get_property_label_and_des(property_id)}\n")
                # endfor
                fout.write(f"###############################\n\n")
            # endfor
        # endwith

    # Note: write test prediction file
    # if if_write_pred_result and mode == "test":
    #     write_test_prediction(args, preds)
    # # endif
    return output_json


def evaluate(args, processor, model, mode=None,
             if_write_pred_result=False,
             epoch_index=None,
             step=None,
             output_file=None):
    assert epoch_index is not None
    assert mode in ["train", "valid", "test"]

    # eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}

    # eval dataset could be train, dev, test
    eval_dataset = load_and_cache_examples(args,
                                           processor,
                                           mode=mode)  # mode decides which dataset to evaluate

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = args.per_gpu_eval_batch_size
    total_eval_size = len(eval_dataset)
    batch_num = math.ceil(total_eval_size * 1.0 / eval_batch_size)

    # Eval!
    print(f"***** Running evaluation for {mode} dataset*****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.per_gpu_eval_batch_size}")
    print(f"  Batch num per epoch = {batch_num}")

    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch_index in tqdm(range(batch_num), desc="Evaluating"):
        model.eval()

        subj_embed_matrix, obj_embed_matrix, label_id_list = \
            processor.get_model_input(
                eval_dataset[batch_index * eval_batch_size: (batch_index + 1) * eval_batch_size])

        # load on device
        subj_embed_matrix = torch.tensor(
            subj_embed_matrix, dtype=torch.float).to(args.device)
        obj_embed_matrix = torch.tensor(
            obj_embed_matrix, dtype=torch.float).to(args.device)
        label_id_list = torch.tensor(
            label_id_list, dtype=torch.long).to(args.device)

        with torch.no_grad():
            total_score, loss = model(
                args, subj_embed_matrix, obj_embed_matrix, label_id_list=None)
        # endwith

        nb_eval_steps += 1
        if preds is None:
            preds = total_score.detach().cpu().numpy()
            out_label_ids = label_id_list.detach().cpu().numpy()
        else:
            preds = np.append(
                preds, total_score.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_id_list.detach().cpu().numpy(), axis=0)
        # endif
    # endfor

    preds = np.argmax(preds, axis=1)
    auc_score = evaluation_metrics.get_auc_score(
        out_label_ids, preds)  # roc_auc_score(y_true, pred_score)
    results["auc_score"] = auc_score

    if output_file is None:
        output_eval_file = os.path.join(eval_output_dir, mode + "_results.txt")
    else:
        output_eval_file = output_file
    # endif

    with open(output_eval_file, "a") as writer:
        output_json = {'epoch': epoch_index,
                       'step': step,
                       'mode': mode}
        output_json.update(results)
        # print("{}".format(output_json))
        writer.write(json.dumps(output_json) + '\n')
    # endwith

    # Note: write test prediction file
    # if if_write_pred_result and mode == "test":
    #     write_test_prediction(args, preds)
    # # endif
    return output_json


def write_test_prediction(args, pred_score_list):
    """
    Sort by predication score, and then check, how the label are distributed
    :param data_input_dir:
    :param pred_label_list:
    :param output_dir:
    :param epoch_index:
    :param step_index:
    :return:
    """
    # pred_score_list is np array
    tmp_pred_score_list = pred_score_list.to_list()

    data_folder = args.data_folder
    test_data_file_path = os.path.join(data_folder, args.test_file_name)
    test_data_pred_result_file_path = os.path.join(
        args.output_dir, "test_data_pred_file.txt")

    with open(test_data_file_path, mode="r") as fin, \
            open(test_data_pred_result_file_path, mode="w") as fout:
        test_data_tuple_list = []
        line_list = fin.readlines()
        for index, line in enumerate(line_list):
            line = line.strip()
            if len(line) == 0:
                continue
            # endif
            json_obj = json.loads(line)
            instance_id = json_obj["id"]
            sent_text = json_obj["text_for_parsing"]
            label = json_obj["label"]
            compatibility_score = tmp_pred_score_list[index]
            test_data_tuple_list.append(
                [instance_id, sent_text, label, compatibility_score])
        # endfor

        sorted_test_data_tuple_list = sorted(
            test_data_tuple_list, key=lambda x: x[3])

        for entry in sorted_test_data_tuple_list:
            entry_line = "\t".join(entry)
            fout.write(entry_line + "\n")
        # endfor


def save_model(model, tokenizer, args):
    """
    The pytorch recommended way is to save/load state_dict
    :return:
    """
    model_to_save = model.module if hasattr(model,
                                            'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    pass


def load_model(model_class, tokenizer_class, args):
    """
    :param saved_model_file:
    :return:
    """
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    model.to(args.device)
    return model, tokenizer


# def save_GAT_model(model, path):
#     torch.save(model.state_dict(), path)
#     pass


def load_GAT_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
    pass


def _validate_do_lower_case(args):
    """
    Since, do lower case or not is based on the tokenizer of pretrained bert
    So args.do_lower_case should be consistent with the pretrained bert tokenizer config file
    :return:
    """
    if args.pretrained_bert_model_folder_for_feature_creation is not None:
        bert_tokenizer_config_file = os.path.join(args.pretrained_bert_model_folder_for_feature_creation,
                                                  "tokenizer_config.json")
        with open(bert_tokenizer_config_file, mode="r") as fin:
            tokenizer_do_lower_case = json.load(fin)["do_lower_case"]
        # endwith

        if tokenizer_do_lower_case != args.do_lower_case:
            print("args.do_lower_case={} does not consistent with tokenizer config do_lower_case={}".format(
                args.do_lower_case, tokenizer_do_lower_case))
            sys.exit(1)
        # endif


def _validate_input_size(args):
    if args.feature_type == "glove":
        assert args.glove_embed_size == args.input_size == 300

    if "bert" in args.feature_type:
        if "bert-base" in args.pretrained_transformer_model_name_or_path:
            assert args.input_size == 768
        # endif
        if "bert-large" in args.pretrained_transformer_model_name_or_path:
            assert args.input_size == 1024
        # endif
    # endif


def run_app_for_gat_24_relations():
    begin = time.time()
    args = argument_parser()

    # NOTE: valid, test data folder
    valid_test_data_root_folder = args.valid_test_root_folder
    args.valid_normal_folder = os.path.join(
        valid_test_data_root_folder, "valid_normal")
    args.valid_novel_folder = os.path.join(
        valid_test_data_root_folder, "valid_novel")
    args.test_normal_folder = os.path.join(
        valid_test_data_root_folder, "test_normal")
    args.test_novel_folder = os.path.join(
        valid_test_data_root_folder, "test_novel")

    assert args.valid_normal_folder is not None
    assert args.valid_novel_folder is not None
    assert args.test_normal_folder is not None
    assert args.test_novel_folder is not None

    # NOTE: valid and test dataset and feature reference folder
    args.valid_test_example_and_feature_reference_folder = "./dataset/valid_test_example_feature_folder_reference"

    # NOTE: folder for example and feature
    os.makedirs(args.valid_test_example_and_feature_folder, exist_ok=True)
    assert args.valid_test_example_and_feature_folder is not None

    # ######## tmp training file #######
    # (1) all relation to guid in training data folder
    # This folder is related to training data, it should be verified that self.train_tmp_folder is changed
    relation_id_to_guid_list_dict_file = os.path.join(
        args.train_tmp_folder, "all_relation_id_to_guid_list_dict_db.json")
    os.makedirs(args.train_tmp_folder, exist_ok=True)

    # validate the do_lower_case, make sure that, tokenizer do_lower_case is consistent with args.do_lower_case
    _validate_do_lower_case(args)
    _validate_input_size(args)

    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # Set seed
    pytorch_utils.set_seed(args)

    if not args.resume_training:
        # # create unique output folder based on time
        pytorch_utils.set_output_folder(args)
        print(
            f"training from scratch >>>>>>>>>>>>>>> output folder is {args.output_dir}")
    else:
        args.output_dir = args.resume_checkpoint_dir
    # endif

    # #### write parameter ####
    pytorch_utils.write_params(args)

    dataset_processor = WikidataProcessor(
        args=args,
        train_data_folder=args.train_data_folder,
        valid_normal_folder=args.valid_normal_folder,
        valid_novel_folder=args.valid_novel_folder,
        test_normal_folder=args.test_normal_folder,
        test_novel_folder=args.test_novel_folder,

        high_quality_human_id_file_path=args.high_quality_human_id_file_path,
        high_quality_human_id_set_current_in_database=args.high_quality_human_id_set_current_in_database,
        high_quality_property_str_to_num_file_path=args.high_quality_property_str_to_num_file_path,

        total_rc_relation_list_file=args.total_rc_relation_list_file,
        train_rc_relation_list_file=args.train_rc_relation_list_file,
        evaluation_rc_relation_list_file=args.evaluation_rc_relation_list_file
    )

    # #################################### load property information #############################################
    property_info_folder = "./dataset/FINAL_property_id_to_info_data/output"
    property_processor = PropertyProcessor(
        args, output_dir=property_info_folder)
    property_index_to_str_list, property_str_to_index_dict, property_str_to_info_dict, property_embedding_label_arr, property_embedding_label_and_description_arr = \
        property_processor.load_property_info()

    property_embed_arr = None
    if args.property_embed_type == "label":
        property_embed_arr = property_embedding_label_arr
    # endif
    if args.property_embed_type == "label_and_description":
        property_embed_arr = property_embedding_label_and_description_arr
    # endif
    assert property_embed_arr is not None

    # ############################################################################################################

    # data folder
    # './data/data_examples_and_features' contains test/valid_cached_example, test/valid_cached_feature_bert
    print(f"data folder: {args.valid_test_example_and_feature_folder}")
    os.makedirs(args.valid_test_example_and_feature_folder, exist_ok=True)
    #

    # Training
    if args.do_train:
        # (1) loading valid and test examples and features in "e1, e2 format"
        valid_examples = dataset_processor.load_and_cache_examples_with_e1_e2(
            args, mode="valid")
        test_examples = dataset_processor.load_and_cache_examples_with_e1_e2(
            args, mode="test")

        valid_features = dataset_processor.load_and_cache_features_with_e1_e2(
            args, mode="valid")
        test_features = dataset_processor.load_and_cache_features_with_e1_e2(
            args, mode="test")

        # (1) loading valid and test examples and features in "h, t format"
        # --
        # test_examples = dataset_processor.load_and_cache_examples_from_database(args, mode="test")
        # # Old
        # test_features = dataset_processor.load_and_cache_features_from_database(args, mode="test")
        # #test_features = dataset_processor.load_and_cache_features_from_database_based_on_existing_dump(args, test_examples, mode="test")
        # dataset_processor.validate_consistency_of_cached_examples_and_features(test_examples, test_features)

        # # # --
        # valid_examples = dataset_processor.load_and_cache_examples_from_database(args, mode="valid")
        # # Old
        # valid_features = dataset_processor.load_and_cache_features_from_database(args, mode="valid")
        # #valid_features = dataset_processor.load_and_cache_features_from_database_based_on_existing_dump(args, valid_examples, mode="valid")
        # dataset_processor.validate_consistency_of_cached_examples_and_features(valid_examples, valid_features)

        sys.exit(0)

        ############### model information ############
        num_of_rc_relation = len(dataset_processor.train_rc_str_list)

        model = None
        if args.model_type == "RGAT":
            from model_utils import RGAT
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)

            model = RGAT(args, property_embed_matrix_tensor,
                         num_of_rc_relation=num_of_rc_relation)

            model.to(args.device)

        if args.model_type == "RGAT_Stack":
            from model_utils import RGAT_Stack
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_Stack(args, property_embed_matrix_tensor,
                               num_of_rc_relation=num_of_rc_relation)
            model.to(args.device)

        if args.model_type == "RGAT_MaxMargin":
            from model_utils import RGAT_MaxMargin
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_MaxMargin(
                args, property_embed_matrix_tensor, num_of_rc_relation=num_of_rc_relation)
            model.to(args.device)

        if args.model_type == "RGAT_MaxMargin_Stack":
            from model_utils import RGAT_MaxMargin_Stack
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_MaxMargin_Stack(
                args, property_embed_matrix_tensor, num_of_rc_relation=num_of_rc_relation)
            model.to(args.device)

        if args.model_type == "RGAT_MaxMargin_Stack_With_Head_Tail_Attention":
            from model_utils import RGAT_MaxMargin_Stack_With_Head_Tail_Attention
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_MaxMargin_Stack_With_Head_Tail_Attention(
                args, property_embed_matrix_tensor, num_of_rc_relation=num_of_rc_relation)
            model.to(args.device)

        # one variation
        if args.model_type == "GAT_MaxMargin":
            from model_utils import GAT_MaxMargin
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = GAT_MaxMargin(
                args, property_embed_matrix_tensor, num_of_rc_relation)
            model.to(args.device)
        # endif

        assert model is not None

        # ########## the negative sample during training is generated dynamically ##########
        # property_cache = PropertyInfoCache()

        if args.training_sampling_mode == "by_each_relation":
            train_contrastive_training_extract_feature_from_database_for_all_relation_by_relation(args,
                                                                                                  dataset_processor,
                                                                                                  model)
        # endif

        if args.training_sampling_mode == "among_all_relation":
            train_contrastive_training_extract_feature_from_database_for_all_relation_among_all_relations(args,
                                                                                                          dataset_processor,
                                                                                                          model)
        # endif

    # endif

    time_length = time.time() - begin
    print(f"Total time: {time_length / 60} mins")

    # endif
    try:
        move_log_file_to_output_directory(args.output_dir)
    except:
        print("cannot find log.txt file")
    # endfor

    # load the save model and check that it could reproduce the dev/test dataset result
    # saved_model, saved_tokenizer = self.load_model(model_class, tokenizer_class, args)
    #
    # #
    # best_epoch, best_step = None, None
    # with open(self.best_dev_eval_file, mode="r") as fin:
    #     for line in fin:
    #         line = line.strip()
    #         json_obj = json.loads(line)
    #         best_epoch = json_obj['epoch']
    #         best_step = json_obj['step']
    #     # endfor
    # # endwith
    #
    # self.reproduce_dev_eval_file = os.path.join(args.output_dir, "reproduce_dev_eval_file.txt")
    # self.reproduce_test_eval_file = os.path.join(args.output_dir, "reproduce_test_eval_file.txt")
    #
    # self.evaluate(args, saved_model, saved_tokenizer, label_list, mode="valid",
    #               epoch_index=best_epoch,
    #               step=best_step, output_file=self.reproduce_dev_eval_file)
    # self.evaluate(args, saved_model, saved_tokenizer, label_list, mode="test",
    #               epoch_index=best_epoch,
    #               step=best_step, output_file=self.reproduce_test_eval_file)
    #
    # self.move_log_file_to_output_directory(args.output_dir)


def run_app_for_gat_20_relations_Oct_2022():
    """
    Test data is "e1, e2 version"
    """
    begin = time.time()
    args = argument_parser()

    # NOTE: valid, test data folder
    valid_test_data_root_folder = args.valid_test_root_folder
    args.valid_normal_folder = os.path.join(
        valid_test_data_root_folder, "valid_normal")
    args.valid_novel_folder = os.path.join(
        valid_test_data_root_folder, "valid_novel")
    args.test_normal_folder = os.path.join(
        valid_test_data_root_folder, "test_normal")
    args.test_novel_folder = os.path.join(
        valid_test_data_root_folder, "test_novel")

    assert args.valid_normal_folder is not None
    assert args.valid_novel_folder is not None
    assert args.test_normal_folder is not None
    assert args.test_novel_folder is not None

    # NOTE: valid and test dataset and feature reference folder
    # args.valid_test_example_and_feature_reference_folder = "./dataset/valid_test_example_feature_folder_reference"

    # NOTE: folder for example and feature
    os.makedirs(args.valid_test_example_and_feature_folder, exist_ok=True)
    assert args.valid_test_example_and_feature_folder is not None

    # ######## tmp training file #######
    # (1) all relation to guid in training data folder
    # This folder is related to training data, it should be verified that self.train_tmp_folder is changed
    relation_id_to_guid_list_dict_file = os.path.join(
        args.train_tmp_folder, "all_relation_id_to_guid_list_dict_db.json")
    os.makedirs(args.train_tmp_folder, exist_ok=True)

    # validate the do_lower_case, make sure that, tokenizer do_lower_case is consistent with args.do_lower_case
    _validate_do_lower_case(args)
    _validate_input_size(args)

    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # Set seed
    pytorch_utils.set_seed(args)

    if not args.resume_training:
        # # create unique output folder based on time
        pytorch_utils.set_output_folder(args)
        print(
            f"training from scratch >>>>>>>>>>>>>>> output folder is {args.output_dir}")
    else:
        args.output_dir = args.resume_checkpoint_dir
    # endif

    # #### write parameter ####
    pytorch_utils.write_params(args)

    dataset_processor = WikidataProcessor(
        args=args,
        train_data_folder=args.train_data_folder,
        valid_normal_folder=args.valid_normal_folder,
        valid_novel_folder=args.valid_novel_folder,
        test_normal_folder=args.test_normal_folder,
        test_novel_folder=args.test_novel_folder,

        high_quality_human_id_file_path=args.high_quality_human_id_file_path,
        high_quality_human_id_set_current_in_database=args.high_quality_human_id_set_current_in_database,
        high_quality_property_str_to_num_file_path=args.high_quality_property_str_to_num_file_path,

        train_rc_relation_list_file=args.train_rc_relation_list_file,
        evaluation_rc_relation_list_file=args.evaluation_rc_relation_list_file
    )

    # #################################### load property information #############################################
    property_info_folder = "./dataset/FINAL_property_id_to_info_data/output"
    property_processor = PropertyProcessor(
        args, output_dir=property_info_folder)
    property_index_to_str_list, property_str_to_index_dict, property_str_to_info_dict, property_embedding_label_arr, property_embedding_label_and_description_arr = \
        property_processor.load_property_info()

    property_embed_arr = None
    if args.property_embed_type == "label":
        property_embed_arr = property_embedding_label_arr
    # endif
    if args.property_embed_type == "label_and_description":
        property_embed_arr = property_embedding_label_and_description_arr
    # endif
    assert property_embed_arr is not None

    # ############################################################################################################

    # data folder
    # './data/data_examples_and_features' contains test/valid_cached_example, test/valid_cached_feature_bert
    print(f"data folder: {args.valid_test_example_and_feature_folder}")
    os.makedirs(args.valid_test_example_and_feature_folder, exist_ok=True)
    #

    # Training
    if args.do_train:
        # (1) loading valid and test examples and features in "e1, e2 format"
        # valid_examples = dataset_processor.load_and_cache_examples_with_e1_e2(
        #     args, mode="valid")
        # valid_features = dataset_processor.load_and_cache_features_with_e1_e2(
        #     args, mode="valid")
        
        # test_examples = dataset_processor.load_and_cache_examples_with_e1_e2(
        #     args, mode="test")
        # test_features = dataset_processor.load_and_cache_features_with_e1_e2(
        #     args, mode="test")
        
        # dataset_processor.validate_check_guid_list(valid_examples, valid_features, mode="valid")
        # dataset_processor.validate_check_guid_list(test_examples, test_features, mode="test")
        
        # predicted_guid_set = set()
        # with open(args.predicted_rc_label_file, mode="r") as fin:
        #     for line in fin:
        #         line = line.strip()
        #         if len(line) == 0:
        #             continue
        #         #endif
        #         json_obj = json.loads(line)
        #         guid = json_obj["guid"]
        #         predicted_guid_set.add(guid)
        #     #endfor
        # #endwith
        
        # # get test guid set
        # test_guid_set = set()
        # for tmp_example in test_examples:
        #     tmp_guid = tmp_example.guid
        #     test_guid_set.add(tmp_guid)
        # #endfor
        
        # assert predicted_guid_set == test_guid_set
        
            
        
        # dataset_processor.validate_consistency_of_cached_examples_and_features(valid_examples, valid_features)        
        # dataset_processor.validate_consistency_of_cached_examples_and_features(test_examples, test_features)
        
        # print("hello")
        # (1) loading valid and test examples and features in "h, t format"
        # --
        # test_examples = dataset_processor.load_and_cache_examples_from_database(args, mode="test")
        # # Old
        # test_features = dataset_processor.load_and_cache_features_from_database(args, mode="test")
        # #test_features = dataset_processor.load_and_cache_features_from_database_based_on_existing_dump(args, test_examples, mode="test")
        # dataset_processor.validate_consistency_of_cached_examples_and_features(test_examples, test_features)

        # # # --
        # valid_examples = dataset_processor.load_and_cache_examples_from_database(args, mode="valid")
        # # Old
        # valid_features = dataset_processor.load_and_cache_features_from_database(args, mode="valid")
        # #valid_features = dataset_processor.load_and_cache_features_from_database_based_on_existing_dump(args, valid_examples, mode="valid")
        # dataset_processor.validate_consistency_of_cached_examples_and_features(valid_examples, valid_features)

        # sys.exit(0)

        ############### model information ############
        num_of_rc_relation = len(dataset_processor.train_rc_str_list)

        model = None
        if args.model_type == "RGAT":
            from model_utils import RGAT
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)

            model = RGAT(args, property_embed_matrix_tensor,
                         num_of_rc_relation=num_of_rc_relation)

            model.to(args.device)

        if args.model_type == "RGAT_Stack":
            from model_utils import RGAT_Stack
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_Stack(args, property_embed_matrix_tensor,
                               num_of_rc_relation=num_of_rc_relation)
            model.to(args.device)

        if args.model_type == "RGAT_MaxMargin":
            from model_utils import RGAT_MaxMargin
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_MaxMargin(
                args, property_embed_matrix_tensor, num_of_rc_relation=num_of_rc_relation)
            model.to(args.device)

        if args.model_type == "RGAT_MaxMargin_Stack":
            from model_utils import RGAT_MaxMargin_Stack
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_MaxMargin_Stack(
                args, property_embed_matrix_tensor, num_of_rc_relation=num_of_rc_relation)
            model.to(args.device)

        if args.model_type == "RGAT_MaxMargin_Stack_With_Head_Tail_Attention":
            from model_utils import RGAT_MaxMargin_Stack_With_Head_Tail_Attention
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_MaxMargin_Stack_With_Head_Tail_Attention(
                args, 
                property_embed_matrix_tensor,
                relation_str_to_index_dict=dataset_processor.train_rc_str_to_num_dict,
                index_to_relation_str_dict=dataset_processor.train_num_to_rc_str_dict
                )
            model.to(args.device)
        
        if args.model_type == "RGAT_Stack_With_Head_Tail_Attention_CrossEntropy":
            from model_utils import RGAT_Stack_With_Head_Tail_Attention_CrossEntropy
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = RGAT_Stack_With_Head_Tail_Attention_CrossEntropy(
                args, 
                property_embed_matrix_tensor,
                relation_str_to_index_dict=dataset_processor.train_rc_str_to_num_dict,
                index_to_relation_str_dict=dataset_processor.train_num_to_rc_str_dict
                )
            model.to(args.device)
        

        # one variation
        if args.model_type == "GAT_MaxMargin":
            from model_utils import GAT_MaxMargin
            property_embed_matrix_tensor = torch.tensor(
                property_embed_arr).float().to(args.device)
            model = GAT_MaxMargin(
                args, property_embed_matrix_tensor, num_of_rc_relation)
            model.to(args.device)
        # endif

        assert model is not None

        # ########## the negative sample during training is generated dynamically ##########
        # property_cache = PropertyInfoCache()

        evaluation_processor = VisualizeAttention(args)

        if args.training_sampling_mode == "by_each_relation":
            train_contrastive_training_extract_feature_from_database_for_all_relation_by_relation_with_e1_and_e2(args,
                                                                                                                 evaluation_processor,
                                                                                                                 dataset_processor,
                                                                                                                 model)

            # train_contrastive_training_extract_feature_from_database_for_all_relation_by_relation(args,
            #                                                                                       dataset_processor,
            #                                                                                       model)
        # endif

        if args.training_sampling_mode == "among_all_relation":
            # train_contrastive_training_extract_feature_from_database_for_all_relation_among_all_relations(args,
            #                                                                                               dataset_processor,
            #                                                                                               model)
            train_contrastive_training_extract_feature_from_database_for_all_relation_among_all_relations_with_e1_and_e2(args,
                                                                                                                         evaluation_processor,
                                                                                                                         dataset_processor,
                                                                                                                         model)
        # endif

    # endif

    time_length = time.time() - begin
    print(f"Total time: {time_length / 60} mins")

    # endif
    try:
        move_log_file_to_output_directory(args.output_dir)
    except:
        print("cannot find log.txt file")
    # endfor

    # load the save model and check that it could reproduce the dev/test dataset result
    # saved_model, saved_tokenizer = self.load_model(model_class, tokenizer_class, args)
    #
    # #
    # best_epoch, best_step = None, None
    # with open(self.best_dev_eval_file, mode="r") as fin:
    #     for line in fin:
    #         line = line.strip()
    #         json_obj = json.loads(line)
    #         best_epoch = json_obj['epoch']
    #         best_step = json_obj['step']
    #     # endfor
    # # endwith
    #
    # self.reproduce_dev_eval_file = os.path.join(args.output_dir, "reproduce_dev_eval_file.txt")
    # self.reproduce_test_eval_file = os.path.join(args.output_dir, "reproduce_test_eval_file.txt")
    #
    # self.evaluate(args, saved_model, saved_tokenizer, label_list, mode="valid",
    #               epoch_index=best_epoch,
    #               step=best_step, output_file=self.reproduce_dev_eval_file)
    # self.evaluate(args, saved_model, saved_tokenizer, label_list, mode="test",
    #               epoch_index=best_epoch,
    #               step=best_step, output_file=self.reproduce_test_eval_file)
    #
    # self.move_log_file_to_output_directory(args.output_dir)


def run_app_for_gat_20_relations_evaluation():
    begin = time.time()

    args = argument_parser()

    # validate the do_lower_case, make sure that, tokenizer do_lower_case is consistent with args.do_lower_case
    _validate_do_lower_case(args)
    _validate_input_size(args)
    #

    # data folder
    print(f"data folder: {args.data_folder}")
    os.makedirs(args.data_folder, exist_ok=True)
    #
    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # Set seed
    pytorch_utils.set_seed(args)

    if not args.resume_training:
        # # create unique output folder based on time
        pytorch_utils.set_output_folder(args)
        print(
            f"training from scratch >>>>>>>>>>>>>>> output folder is {args.output_dir}")

    else:
        args.output_dir = args.resume_checkpoint_dir

    # write parameter
    pytorch_utils.write_params(args)

    #
    # # load main model
    # # model = GraphAttentionNetwork(args, args.layer_node_feature_size, num_class, args.num_graph_layers)
    # model = Net(args)
    # model.to(args.device)

    # ############ load property feature ##############
    # property_loader = WikidataPropertyRepresentationLoader()
    # property_loader.load_property_feature_matrix(feature_type="label")

    # ################## FILE PATHS ################
    # 10000 -> 188251
    # 100 -> 2050

    evaluation_relation_list_file = "./data/RC_evaluation_relations.json"

    dataloading_processor = Wikidata_DataLoading_Processor(args=args,
                                                           train_data_folder=args.train_data_folder,
                                                           valid_normal_folder=args.valid_normal_folder,
                                                           valid_novel_folder=args.valid_novel_folder,
                                                           test_normal_folder=args.test_normal_folder,
                                                           test_novel_folder=args.test_novel_folder,
                                                           rc_relation_list_file=args.rc_relation_list_file,
                                                           evaluation_relation_list_file=evaluation_relation_list_file,
                                                           )

    relation_id_to_guid_list_dict_file = "all_relation_id_to_guid_list_dict_db.json"

    database_processor = Wikidata_Database_Processor(args=args,
                                                     rc_relation_list_file=args.rc_relation_list_file,
                                                     evaluation_relation_list_file=evaluation_relation_list_file,
                                                     relation_id_to_guid_list_dict_file=relation_id_to_guid_list_dict_file)

    # -------------- load property information --------------
    # Two types of property embedding:
    # Embedding created based on
    # (1) label
    # (2) label and description

    property_index_to_str_list, property_str_to_index_dict, property_str_to_info_dict, \
        property_embedding_label_arr, property_embedding_label_and_description_arr = dataloading_processor.load_property_info()

    property_embed_arr = None
    if args.property_embed_type == "label":
        property_embed_arr = property_embedding_label_arr
    # endif
    if args.property_embed_type == "label_and_description":
        property_embed_arr = property_embedding_label_and_description_arr
    # endif
    assert property_embed_arr is not None

    model = None
    if args.model_type == "RGAT":
        from model_utils import RGAT
        num_of_rc_relation = len(
            dataloading_processor.rc_label_index_to_str_list)
        property_embed_matrix_tensor = torch.tensor(
            property_embed_arr).float().to(args.device)
        model = RGAT(args, property_embed_matrix_tensor,
                     num_of_rc_relation=num_of_rc_relation)
        model.to(args.device)

    if args.model_type == "RGAT_MaxMargin":
        from model_utils import RGAT_MaxMargin
        num_of_rc_relation = len(
            dataloading_processor.rc_label_index_to_str_list)
        property_embed_matrix_tensor = torch.tensor(
            property_embed_arr).float().to(args.device)
        model = RGAT_MaxMargin(
            args, property_embed_matrix_tensor, num_of_rc_relation=num_of_rc_relation)
        model.to(args.device)

    if args.model_type == "RGAT_MaxMargin_Stack":
        from model_utils import RGAT_MaxMargin_Stack
        num_of_rc_relation = len(
            dataloading_processor.rc_label_index_to_str_list)
        property_embed_matrix_tensor = torch.tensor(
            property_embed_arr).float().to(args.device)
        model = RGAT_MaxMargin_Stack(
            args, property_embed_matrix_tensor, num_of_rc_relation=num_of_rc_relation)
        model.to(args.device)

    if args.model_type == "RGAT_MaxMargin_Stack_With_Head_Tail_Attention":
        from model_utils import RGAT_MaxMargin_Stack_With_Head_Tail_Attention
        num_of_rc_relation = len(
            dataloading_processor.rc_label_index_to_str_list)
        property_embed_matrix_tensor = torch.tensor(
            property_embed_arr).float().to(args.device)
        model = RGAT_MaxMargin_Stack_With_Head_Tail_Attention(
            args, property_embed_matrix_tensor, num_of_rc_relation=num_of_rc_relation)
        model.to(args.device)

    assert model is not None

    if args.resume_training:
        device = args.device
        model_file = os.path.join(
            args.resume_checkpoint_dir, "best_mean_auc_model", "best_model.pt")
        model = pytorch_utils.load_model(model, device, model_file)
        saved_epoch_index = -1
        saved_global_step = -1

    evaluate_for_gat_21_relation(args,
                                 dataloading_processor,
                                 model,
                                 mode="test",
                                 if_write_pred_result=False,
                                 epoch_index=-1,
                                 step=-1,
                                 )

    # endif

    time_length = time.time() - begin
    print(f"Total time: {time_length / 60} mins")

    # endif
    try:
        move_log_file_to_output_directory(args.output_dir)
    except:
        print("cannot find log.txt file")
    # endfor

    # load the save model and check that it could reproduce the dev/test dataset result
    # saved_model, saved_tokenizer = self.load_model(model_class, tokenizer_class, args)
    #
    # #
    # best_epoch, best_step = None, None
    # with open(self.best_dev_eval_file, mode="r") as fin:
    #     for line in fin:
    #         line = line.strip()
    #         json_obj = json.loads(line)
    #         best_epoch = json_obj['epoch']
    #         best_step = json_obj['step']
    #     # endfor
    # # endwith
    #
    # self.reproduce_dev_eval_file = os.path.join(args.output_dir, "reproduce_dev_eval_file.txt")
    # self.reproduce_test_eval_file = os.path.join(args.output_dir, "reproduce_test_eval_file.txt")
    #
    # self.evaluate(args, saved_model, saved_tokenizer, label_list, mode="valid",
    #               epoch_index=best_epoch,
    #               step=best_step, output_file=self.reproduce_dev_eval_file)
    # self.evaluate(args, saved_model, saved_tokenizer, label_list, mode="test",
    #               epoch_index=best_epoch,
    #               step=best_step, output_file=self.reproduce_test_eval_file)
    #
    # self.move_log_file_to_output_directory(args.output_dir)


def _delete_cached_feature_file(args):
    for dir, subdir, file_list in os.walk(args.data_folder):
        for file in file_list:
            if "cached" in file:
                file_path = os.path.join(dir, file)
                os.remove(file_path)
            # endif
        # endfor
    # endfor


def move_log_file_to_output_directory(output_dir):
    """
    The output file name is at the bash file.
    Redirect all the output to the log file.
    The reason do not use file handler of logging module is that, it could only redirect the message from
    user write script to file. For the logging information in API/packages, it will not be blocked and not
    shown in the stdout.
    Using redirect in linux console will redirect all the information printed out on the screen to the file
    not matter where it is from.
    :return:
    """
    shutil.move("./log.txt", output_dir)


def determine_max_length_for_bert_feature():
    """
    Total sentences: 2778

    ============= whitespace tokenizer ==============
    Sent length mean: 5.677105831533478
    Sent length std: 5.309206720503423
    Sent length median: 4.0
    Sent length max: 46
    loading bert model: bert-base-cased to create feature for sent tokens
    Downloading: 100%|██████████| 213k/213k [00:00<00:00, 3.94MB/s]
    bert-base-cased tokenizer is successfully loaded.
    bert tokenizer: 100%|██████████| 2778/2778 [00:00<00:00, 4929.25it/s]

    ============ bert tokenizer ==============
    Sent length mean: 7.558675305975522
    Sent length std: 7.363955370991114
    Sent length median: 5.0
    Sent length max: 71

    ########################################################################
    ########################################################################

    Total sentences: 4756
    ============= whitespace tokenizer ==============
    Sent length mean: 5.823591253153911
    Sent length std: 5.243502789315089
    Sent length median: 4.0
    Sent length max: 46
    loading bert model: bert-base-cased to create feature for sent tokens
    bert-base-cased tokenizer is successfully loaded.
    bert tokenizer: 100%|██████████| 4756/4756 [00:00<00:00, 4804.91it/s]
    ============ bert tokenizer ==============
    Sent length mean: 7.669890664423885
    Sent length std: 7.26012138752769
    Sent length median: 6.0
    Sent length max: 71
    """

    args = argument_parser()
    processor = Entity_Compatibility_Processor(args)

    train_examples = processor.get_train_examples()
    valid_examples = processor.get_valid_examples()
    test_examples = processor.get_test_examples()

    total_example = []
    total_example.extend(train_examples)
    total_example.extend(valid_examples)
    total_example.extend(test_examples)

    sent_len_list = []
    sent_text_list = []
    for exp in total_example:
        subj_sent_len = len(exp.subj_des.split())
        obj_sent_len = len(exp.obj_des.split())

        sent_len_list.append(subj_sent_len)
        sent_len_list.append(obj_sent_len)

        sent_text_list.append(exp.subj_des)
        sent_text_list.append(exp.obj_des)
    # endfor

    sent_len_array = np.array(sent_len_list)

    print("\n\nTotal sentences: {}".format(len(sent_len_list)))

    print("============= whitespace tokenizer ==============")
    print("Sent length mean: {}".format(sent_len_array.mean()))
    print("Sent length std: {}".format(sent_len_array.std()))
    print("Sent length median: {}".format(np.median(sent_len_array)))
    print("Sent length max: {}".format(np.max(sent_len_array)))

    # Note: check bert tokenizer
    from transformers import BertModel, BertTokenizer
    print("loading bert model: {} to create feature for sent tokens".format(
        args.pretrained_transformer_model_name_or_path))
    # bert_model = BertModel.from_pretrained(args.pretrained_transformer_model_name_or_path)
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_transformer_model_name_or_path)
    # load the pretrained bert model. The loaded tokenizer will follow the original pretrained config
    # For "do_lower_case" will decide whether or not make the sent_text lower case
    # It should consistent with the programs argument config
    # bert_pretrained_model, bert_pretrained_tokenizer = self._load_pretrained_bert_model(args, model_class,
    # tokenizer_class)
    # bert_model.eval()
    # bert_model.to(args.device)
    print(f"{args.pretrained_transformer_model_name_or_path} tokenizer is successfully loaded.")

    bert_wordpiece_len_list = []
    for text in tqdm(sent_text_list, desc="bert tokenizer"):
        token_list = text.split()
        word_pieces_list = []
        for w in token_list:
            word_pieces = bert_tokenizer.tokenize(w)
            word_pieces_list += word_pieces
        # endfor
        bert_wordpiece_len_list.append(len(word_pieces_list))
    # endfor

    bert_wordpiece_len_arr = np.array(bert_wordpiece_len_list)

    print("============ bert tokenizer ==============")
    print("Sent length mean: {}".format(bert_wordpiece_len_arr.mean()))
    print("Sent length std: {}".format(bert_wordpiece_len_arr.std()))
    print("Sent length median: {}".format(np.median(bert_wordpiece_len_arr)))
    print("Sent length max: {}".format(np.max(bert_wordpiece_len_arr)))
    pass


def determine_max_length_for_bert_feature_for_gat_model():
    """
    In the gat model, we will create features for all the properties of subject/object wikidata entity

    Total sentences: 233452
    ============= whitespace tokenizer ==============
    Sent length mean: 4.53661566403372
    Sent length std: 5.1357778789831965
    Sent length median: 2.0
    Sent length max: 77
    loading bert model: bert-base-cased to create feature for sent tokens
    bert-base-cased tokenizer is successfully loaded.
    bert tokenizer: 100%|██████████| 233452/233452 [04:16<00:00, 910.98it/s]
    ============ bert tokenizer ==============
    Sent length mean: 10.292672583657454
    Sent length std: 9.723351489240112
    Sent length median: 7.0
    Sent length max: 229
    """

    begin = time.time()

    args = argument_parser()

    # validate the do_lower_case, make sure that, tokenizer do_lower_case is consistent with args.do_lower_case
    _validate_do_lower_case(args)
    _validate_input_size(args)
    #
    # # create unique output folder based on time
    pytorch_utils.set_output_folder(args)
    print(f"output folder is {args.output_dir}")
    #
    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # Set seed
    pytorch_utils.set_seed(args)

    # all dataset file path
    train_normal_file_path = "../data/output_contrastive_entity_pair/inventor_train_normal.txt"
    train_novel_file_path = "../data/output_contrastive_entity_pair/inventor_train_novel.txt"
    valid_normal_file_path = "../data/output_contrastive_entity_pair/inventor_valid_normal.txt"
    valid_novel_file_path = "../data/output_contrastive_entity_pair/inventor_valid_novel.txt"
    test_normal_file_path = "../data/output_contrastive_entity_pair/inventor_test_normal.txt"
    test_novel_file_path = "../data/output_contrastive_entity_pair/inventor_test_novel.txt"

    processor = WikidataEntityRepresentationProcessor(train_normal_file=train_normal_file_path,
                                                      train_novel_file=train_novel_file_path,
                                                      valid_normal_file=valid_normal_file_path,
                                                      valid_novel_file=valid_novel_file_path,
                                                      test_normal_file=test_normal_file_path,
                                                      test_novel_file=test_novel_file_path,
                                                      args=args)

    train_examples = processor.load_train_examples()
    valid_examples = processor.load_valid_examples()
    test_examples = processor.load_test_examples()

    total_example = []
    total_example.extend(train_examples)
    total_example.extend(valid_examples)
    total_example.extend(test_examples)

    time_length_load_exp = time.time() - begin
    # load all train, valid, test examples takes 3.187604820728302 mins.
    print(
        f"load all train, valid, test examples take {time_length_load_exp / 60} mins.")

    sent_len_list = []
    sent_text_list = []
    for exp in total_example:
        subj_sent_label_len = len(exp.subj_wikidata_entity_obj.label.split())
        obj_sent_label_len = len(exp.obj_wikidata_entity_obj.label.split())
        sent_len_list.append(subj_sent_label_len)
        sent_len_list.append(obj_sent_label_len)
        sent_text_list.append(exp.subj_wikidata_entity_obj.label)
        sent_text_list.append(exp.obj_wikidata_entity_obj.label)

        subj_sent_des_len = len(
            exp.subj_wikidata_entity_obj.description.split())
        obj_sent_des_len = len(exp.obj_wikidata_entity_obj.description.split())
        sent_len_list.append(subj_sent_des_len)
        sent_len_list.append(obj_sent_des_len)
        sent_text_list.append(exp.subj_wikidata_entity_obj.description)
        sent_text_list.append(exp.obj_wikidata_entity_obj.description)

        # traverse subj attribute
        for property_id, value_text_list in exp.subj_wikidata_entity_obj.property_to_value_dict.items():
            for value_text in value_text_list:
                subj_value_parts_len = len(value_text.split())
                sent_len_list.append(subj_value_parts_len)
                sent_text_list.append(value_text)
        # endfor

        # traverse obj attribute
        for property_id, value_text_list in exp.obj_wikidata_entity_obj.property_to_value_dict.items():
            for value_text in value_text_list:
                obj_value_parts_len = len(value_text.split())
                sent_len_list.append(obj_value_parts_len)
                sent_text_list.append(value_text)
        # endfor
    # endfor

    sent_len_array = np.array(sent_len_list)

    print("\n\nTotal sentences: {}".format(len(sent_len_list)))

    print("============= whitespace tokenizer ==============")
    print("Sent length mean: {}".format(sent_len_array.mean()))
    print("Sent length std: {}".format(sent_len_array.std()))
    print("Sent length median: {}".format(np.median(sent_len_array)))
    print("Sent length max: {}".format(np.max(sent_len_array)))

    # Note: check bert tokenizer
    from transformers import BertModel, BertTokenizer
    print("loading bert model: {} to create feature for sent tokens".format(
        args.pretrained_transformer_model_name_or_path))
    # bert_model = BertModel.from_pretrained(args.pretrained_transformer_model_name_or_path)
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_transformer_model_name_or_path)
    # load the pretrained bert model. The loaded tokenizer will follow the original pretrained config
    # For "do_lower_case" will decide whether or not make the sent_text lower case
    # It should consistent with the programs argument config
    # bert_pretrained_model, bert_pretrained_tokenizer = self._load_pretrained_bert_model(args, model_class,
    # tokenizer_class)
    # bert_model.eval()
    # bert_model.to(args.device)
    print(f"{args.pretrained_transformer_model_name_or_path} tokenizer is successfully loaded.")

    bert_wordpiece_len_list = []
    for text in tqdm(sent_text_list, desc="bert tokenizer"):
        token_list = text.split()
        word_pieces_list = []
        for w in token_list:
            word_pieces = bert_tokenizer.tokenize(w)
            word_pieces_list += word_pieces
        # endfor
        bert_wordpiece_len_list.append(len(word_pieces_list))
    # endfor

    bert_wordpiece_len_arr = np.array(bert_wordpiece_len_list)

    print("============ bert tokenizer ==============")
    print("Sent length mean: {}".format(bert_wordpiece_len_arr.mean()))
    print("Sent length std: {}".format(bert_wordpiece_len_arr.std()))
    print("Sent length median: {}".format(np.median(bert_wordpiece_len_arr)))
    print("Sent length max: {}".format(np.max(bert_wordpiece_len_arr)))
    pass


def create_examples_for_train_valid_test_dataset():
    begin = time.time()

    args = argument_parser()

    # validate the do_lower_case, make sure that, tokenizer do_lower_case is consistent with args.do_lower_case
    _validate_do_lower_case(args)
    _validate_input_size(args)
    #
    # # create unique output folder based on time
    pytorch_utils.set_output_folder(args)
    print(f"output folder is {args.output_dir}")
    #
    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # Set seed
    pytorch_utils.set_seed(args)

    # write parameter
    pytorch_utils.write_params(args)
    #

    train_normal_file_path = "../data/output_contrastive_entity_pair/inventor_train_normal.txt"
    train_novel_file_path = "../data/output_contrastive_entity_pair/inventor_train_novel.txt"
    valid_normal_file_path = "../data/output_contrastive_entity_pair/inventor_valid_normal.txt"
    valid_novel_file_path = "../data/output_contrastive_entity_pair/inventor_valid_novel.txt"
    test_normal_file_path = "../data/output_contrastive_entity_pair/inventor_test_normal.txt"
    test_novel_file_path = "../data/output_contrastive_entity_pair/inventor_test_novel.txt"

    entity_processor = WikidataEntityRepresentationProcessor(
        train_normal_file_path,
        train_novel_file_path,
        valid_normal_file_path,
        valid_novel_file_path,
        test_normal_file_path,
        test_novel_file_path,
        None,
        None,
        None,
        args)

    property_cache = PropertyInfoCache()

    # -------- create examples -----------
    # validation data
    valid_examples = entity_processor.load_valid_examples()
    print(f"There are {len(valid_examples)} valid examples.")
    valid_exp_output_file = os.path.join(
        args.data_folder, "valid_example_file.txt")
    entity_processor.write_examples_to_file(
        property_cache, valid_examples, valid_exp_output_file)
    valid_pickle_file = os.path.join(args.data_folder, "valid_example.pkl")
    with open(valid_pickle_file, mode="wb") as fout:
        pickle.dump(valid_examples, fout)
    # endwith

    # Test data
    test_examples = entity_processor.load_test_examples()
    print(f"There are {len(test_examples)} test examples.")
    test_exp_output_file = os.path.join(
        args.data_folder, "test_example_file.txt")
    entity_processor.write_examples_to_file(
        property_cache, test_examples, test_exp_output_file)
    test_pickle_file = os.path.join(args.data_folder, "test_example.pkl")
    with open(test_pickle_file, mode="wb") as fout:
        pickle.dump(test_examples, fout)
    # endwith

    # training data
    train_examples = entity_processor.load_train_examples()
    print(f"There are {len(train_examples)} training examples.")
    train_exp_output_file = os.path.join(
        args.data_folder, "train_example_file.txt")
    entity_processor.write_examples_to_file(
        property_cache, train_examples, train_exp_output_file)
    train_pickle_file = os.path.join(args.data_folder, "train_example.pkl")
    with open(train_pickle_file, mode="wb") as fout:
        pickle.dump(train_examples, fout)
    # endwith

    # -------------- check pickle files ----------
    train_pickle_file = os.path.join(args.data_folder, "train_example.pkl")
    with open(train_pickle_file, mode="rb") as fin:
        train_examples = pickle.load(fin)  # currently only has normal examples
    # endwith


def main_wikidata_create_all_entity_features():
    args = argument_parser()

    # validate the do_lower_case, make sure that, tokenizer do_lower_case is consistent with args.do_lower_case
    _validate_do_lower_case(args)
    _validate_input_size(args)
    #
    # # create unique output folder based on time
    pytorch_utils.set_output_folder(args)
    print(f"output folder is {args.output_dir}")
    #
    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # Set seed
    pytorch_utils.set_seed(args)

    # all dataset file path
    train_normal_file_path = "../data/output_contrastive_entity_pair/inventor_train_normal.txt"
    train_novel_file_path = "../data/output_contrastive_entity_pair/inventor_train_novel.txt"
    valid_normal_file_path = "../data/output_contrastive_entity_pair/inventor_valid_normal.txt"
    valid_novel_file_path = "../data/output_contrastive_entity_pair/inventor_valid_novel.txt"
    test_normal_file_path = "../data/output_contrastive_entity_pair/inventor_test_normal.txt"
    test_novel_file_path = "../data/output_contrastive_entity_pair/inventor_test_novel.txt"

    processor = WikidataEntityRepresentationProcessor(train_normal_file=train_normal_file_path,
                                                      train_novel_file=train_novel_file_path,
                                                      valid_normal_file=valid_normal_file_path,
                                                      valid_novel_file=valid_novel_file_path,
                                                      test_normal_file=test_normal_file_path,
                                                      test_novel_file=test_novel_file_path,
                                                      args=args)

    # (1) get all the entity id
    # processor.get_all_wikidata_entity_one_hop()
    # output: all_related_wikidata_entity_id_list.txt

    # (2) wikidata_utils
    # wikidata_processor = WikidataDumpUtils(
    #     split_data_folder="/hdd_1/nma4/python_project/Name_Entity_Novelty_Detection_Dataset_Preparation/data/wikidata/wikidata_json_split_fast",
    #     entity_id_list_file="all_related_wikidata_entity_id_list.txt")
    # wikidata_processor.parallelly_insert_entity_id_into_mongodb()

    # (2.1) validate make sure everything is inserted into mongodb
    # wikidata_processor.parallel_validate_mongodb_contains_entity_id_list()

    # (3) start to create feature for entities
    # # TODO: what if attention attend on "human", need to dynamically sample any entity and put it there.
    processor.load_train_examples()
    processor.load_valid_examples()
    processor.load_test_examples()
    pass


def main_create_feature_for_entities_in_train_valid_test_dataset():
    """
    (1) Extract the subject and object entities in train, valid, test dataset
    (2) Create feature for all the entity set

    Note: !!!
    The entity should remove the target relation id in its relation_it_to_feature pair

    There are totally 770 entity ids from valid and test dataset
    """
    # ################ load all entity id ##############
    args = argument_parser()

    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # all dataset file path
    train_normal_file_path = "../data/output_contrastive_entity_pair/inventor_train_normal.txt"
    train_novel_file_path = "../data/output_contrastive_entity_pair/inventor_train_novel.txt"
    valid_normal_file_path = "../data/output_contrastive_entity_pair/inventor_valid_normal.txt"
    valid_novel_file_path = "../data/output_contrastive_entity_pair/inventor_valid_novel.txt"
    test_normal_file_path = "../data/output_contrastive_entity_pair/inventor_test_normal.txt"
    test_novel_file_path = "../data/output_contrastive_entity_pair/inventor_test_novel.txt"

    file_path_list = [train_normal_file_path,
                      valid_normal_file_path,
                      valid_novel_file_path,
                      test_normal_file_path,
                      test_novel_file_path]

    entity_set = set()
    for file_path in file_path_list:
        tmp_entity_set = WikidataEntityRepresentationProcessor.load_datafile_entity_id(
            file_path)
        entity_set.update(tmp_entity_set)
    # endfor
    print(f"There are totally {len(entity_set)} entity ids")

    # ######################## create feature for all entities ################
    from feature_creation_utils import Wikidata_Entity_Example_and_Feature_DBProcessor
    entity_processor = Wikidata_Entity_Example_and_Feature_DBProcessor(args)
    entity_processor._single_worker_create_wikidata_entity_example_and_feature_object_and_store_to_database(
        list(entity_set))
    pass


def validate_create_feature_for_entities_in_train_valid_test_dataset():
    """
    (1) Extract the subject and object entities in train, valid, test dataset
    (2) Create feature for all the entity set

    Note: !!!
    The entity should remove the target relation id in its relation_it_to_feature pair

    There are totally 770 entity ids from valid and test dataset
    """
    # ################ load all entity id ##############
    args = argument_parser()

    # # create unique output folder based on time
    pytorch_utils.set_output_folder(args)
    print(f"output folder is {args.output_dir}")
    #
    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # all dataset file path
    train_normal_file_path = "../data/output_contrastive_entity_pair/inventor_train_normal.txt"
    train_novel_file_path = "../data/output_contrastive_entity_pair/inventor_train_novel.txt"
    valid_normal_file_path = "../data/output_contrastive_entity_pair/inventor_valid_normal.txt"
    valid_novel_file_path = "../data/output_contrastive_entity_pair/inventor_valid_novel.txt"
    test_normal_file_path = "../data/output_contrastive_entity_pair/inventor_test_normal.txt"
    test_novel_file_path = "../data/output_contrastive_entity_pair/inventor_test_novel.txt"

    file_path_list = [train_normal_file_path,
                      valid_normal_file_path,
                      valid_novel_file_path,
                      test_normal_file_path,
                      test_novel_file_path]

    entity_set = set()
    for file_path in file_path_list:
        tmp_entity_set = WikidataEntityRepresentationProcessor.load_datafile_entity_id(
            file_path)
        entity_set.update(tmp_entity_set)
    # endfor
    print(f"There are totally {len(entity_set)} entity ids")

    # ######################## create feature for all entities ################
    # from feature_creation_utils import Wikidata_Entity_Example_and_Feature_DBProcessor
    # entity_processor = Wikidata_Entity_Example_and_Feature_DBProcessor(args)
    # entity_processor._single_worker_create_wikidata_entity_feature_and_store_to_database(list(entity_set))

    # ############# query database ############
    from pymongo import MongoClient

    # Creating a pymongo client
    client = MongoClient('localhost', 27017)

    # Getting the database instance
    db = client['wikidata_feature']
    print("List of databases after creating new one")
    print(client.list_database_names())

    database_feature_id_set = set()
    database_example_id_set = set()

    for wikidata_id in entity_set:
        # >>>> check feature <<<<
        feature_result_list = []
        for item in db.entity_feature.find({"wikidata_id": wikidata_id}):
            feature_result_list.append(item)
        # endfor
        if len(feature_result_list) > 0:
            database_feature_id_set.add(wikidata_id)
        # endif

        # >>>>> check example <<<<<
        example_result_list = []
        for item in db.entity_example.find({"wikidata_id": wikidata_id}):
            example_result_list.append(item)
        # endfor
        if len(example_result_list) > 0:
            database_example_id_set.add(wikidata_id)
        # endif
    # endfor

    print(len(database_example_id_set))
    print(len(database_feature_id_set))
    print(len(entity_set))

    # check difference
    example_diff = entity_set - database_example_id_set
    print(f"These {len(example_diff)} id has NO example:")
    print(example_diff)

    feature_diff = entity_set - database_feature_id_set
    print(f"These {len(feature_diff)} id has NO feature:")
    print(feature_diff)

    pass


if __name__ == '__main__':
    # main_create_args_obj()

    # ############## (1) create examples and features for entities in train valid test ################
    # main_create_feature_for_entities_in_train_valid_test_dataset()
    # validate_create_feature_for_entities_in_train_valid_test_dataset()

    # run_app_for_gat_24_relations()

    run_app_for_gat_20_relations_Oct_2022()

    # run_app_for_gat_20_relations_evaluation()
    # visualize_attention()

    # main_create_feature_for_train_valid_test_dataset()

    # create_examples_for_train_valid_test_dataset()

    # determine_max_length_for_bert_feature()
    # main_create_property_embedding()
    # main_wikidata_create_all_entity_features()
    # determine_max_length_for_bert_feature_for_gat_model()

    # main_create_feature_for_entities_in_train_valid_test_dataset()
    pass
