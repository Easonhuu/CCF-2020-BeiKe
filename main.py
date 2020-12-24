# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification
from torchviz import make_dot

import time
import numpy as np
from tqdm import trange
from args import args
from dataset_ss import ss_df, SsDataset, DataProcessor
from sklearn.metrics import roc_auc_score, matthews_corrcoef, recall_score, accuracy_score, \
    r2_score, mean_squared_error, mean_absolute_error, precision_score, precision_recall_curve, auc, f1_score
from sklearn.model_selection import KFold
import gc
import sys
import pickle
import random
import csv

import copy
import pandas as pd


import seaborn as sns
from torchsummary import summary
from train_eval_test import train, evaluate, predict
from plot_results import plot_results

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # set seed for CPU
    torch.cuda.manual_seed(seed)  # set seed for current GPU
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)  # set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # CPU/GPU results are consistent
    torch.backends.cudnn.benchmark = False   # the training is accelerated when the training set changes little

if __name__ == "__main__":
    set_seed(args.seed)
    processor = DataProcessor()
    args.device = torch.device("cuda:{}".format(args.gpu_start) if torch.cuda.is_available() else "cpu")
    args.device_ids = []  # IDs of GPUs to be used
    for i in range(args.n_gpu):
        args.device_ids.append(args.gpu_start+i)
    config = AutoConfig.from_pretrained(args.config_name, num_labels=len(processor.get_labels()), cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir if args.cache_dir else None)
    train_df, test_df = ss_df(args, tokenizer)
    print("train data nums: %d, test data nums: %d" % (len(train_df), len(test_df)))
    
    if args.is_cv:
        weight = []
        Kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        for fold_id, (train_idx, valid_idx) in enumerate(Kfold.split(train_df)):
            valid_df_kfold = train_df.iloc[valid_idx]
            train_df_kfold = train_df.iloc[train_idx]
            # train_df_kfold = data_Augment(train_df_kfold, args.seed, prob=0.2)
            print("kfold %d - train data nums: %d, valid data nums: %d" % ((fold_id+1), len(train_df_kfold), len(valid_df_kfold)))

            train_df_kfold = processor.convert_examples_to_features(train_df_kfold, tokenizer, label_list=processor.get_labels(),
                max_length=args.max_seq_length, output_mode=args.output_mode,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )
            valid_df_kfold = processor.convert_examples_to_features(valid_df_kfold, tokenizer, label_list=processor.get_labels(),
                max_length=args.max_seq_length, output_mode=args.output_mode,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )
            test_df_kfold = processor.convert_examples_to_features(test_df, tokenizer, label_list=processor.get_labels(),
                max_length=args.max_seq_length, output_mode=args.output_mode,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )

            train_dataset = SsDataset(train_df_kfold)
            valid_dataset = SsDataset(valid_df_kfold)
            test_dataset = SsDataset(test_df_kfold)
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir if args.cache_dir else None)
            model = model.to(args.device)
            model = nn.DataParallel(model, device_ids=args.device_ids)
            plot_loss, plot_f1, best_param, best_model, real_epochs, dir_save = train(args, model, [train_dataset, valid_dataset], kfold=fold_id+1)
            
            # # plot results
            # train_loss, train_f1, label_pred, label_true = evaluate(args, best_model, train_dataset)
            # plot_results(plot_loss, plot_f1, dir_save, label_pred, label_true, 'train')
            # eval_loss, eval_f1, label_pred, label_true = evaluate(args, best_model, valid_dataset)
            # plot_results(plot_loss, plot_f1, dir_save, label_pred, label_true, 'valid')

            preds_label, preds = predict(args, best_model, test_dataset)
            if fold_id == 0:
                preds_avg = preds / Kfold.n_splits * best_param['valid_f1']
            else:
                preds_avg += preds / Kfold.n_splits * best_param['valid_f1']

            weight.append(best_param['valid_f1'])

            del model
            del best_model
            gc.collect()

        # preds_avg = (preds_avg > 0.5).astype(np.int)
        exp_preds = np.exp(preds_avg)
        softmax_preds = []
        for exp_pred in exp_preds:
            softmax_pred = exp_pred / np.sum(exp_pred)
            softmax_preds.append(softmax_pred)
        preds_avg = np.array(softmax_preds).T
        weight = [np.mean(weight)] * len(test_df)

        save_file_name = "test_submit_%dfold_%s_len-%d_epoch-%d_gpu-%d_perbatch-%d_lossfun-%s_possibility.tsv" \
            % (args.kfold, args.model_specific, args.max_seq_length, args.epochs, args.n_gpu, args.per_gpu_batch_size, args.loss_type)

        # test_df_submit = pd.DataFrame({"query_id": test_df["query_id"], "reply_id": test_df["reply_id"], "label": preds_avg})
        test_df_submit = pd.DataFrame({"query_id": test_df["query_id"], "reply_id": test_df["reply_id"], "label_0": preds_avg[0], "label_1": preds_avg[-1], "weight": weight})
        print("test_df_submit num: %d" % len(test_df_submit))
        test_df_submit.to_csv(os.path.join(args.data_dir, save_file_name), sep='\t', header=None, index=None)
        print("The results of test-set is saved at %s" % os.path.join(args.data_dir, save_file_name))

    else:
        if args.do_train:
            train_df_feature = processor.convert_examples_to_features(train_df, tokenizer, label_list=processor.get_labels(),
                max_length=args.max_seq_length, output_mode=args.output_mode,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )

            train_dataset = SsDataset(train_df_feature)
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir if args.cache_dir else None)
            model = model.to(args.device)
            model = nn.DataParallel(model, device_ids=args.device_ids)
            
            plot_loss, plot_f1, best_param, best_model, real_epochs, dir_save = train(args, model, train_dataset)
            
            # # plot results
            # eval_loss, eval_f1, label_pred, label_true = evaluate(args, best_model, train_dataset)
            # plot_results(plot_loss, plot_f1, dir_save, label_pred, label_true, 'train')

            model_to_save = best_model.module if hasattr(best_model, "module") else best_model # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), args.data_dir+'/best-model.pth')
        
        if args.do_eval:
            state_dict_path = args.data_dir+'/best-model.pth'
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir if args.cache_dir else None)
            model.load_state_dict(state_dict)
            model = model.to(args.device)
            model = nn.DataParallel(model, device_ids=args.device_ids)

            train_df_feature = processor.convert_examples_to_features(train_df, tokenizer, label_list=processor.get_labels(),
                max_length=args.max_seq_length, output_mode=args.output_mode,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )
            train_dataset = SsDataset(train_df_feature)
            print("-----------running evaluate-----------")

            # # plot results
            eval_loss, eval_f1, label_pred, label_true = evaluate(args, model, train_dataset)
            # plot_results(plot_loss, plot_f1, dir_save, label_pred, label_true, 'train')
            
            eval_log = 'valid_loss:{:.3f}, valid_f1:{:.3f}'.format(eval_loss, eval_f1)
            print(eval_log)

        if args.do_test:
            save_file_name = "test_submit_%s_len-%d_epoch-%d_gpu-%d_perbatch-%d_lossfun-%s.tsv" \
                % (args.model_specific, args.max_seq_length, args.epochs, args.n_gpu, args.per_gpu_batch_size, args.loss_type)
            state_dict_path = args.data_dir+'/best-model.pth'
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir if args.cache_dir else None)
            model.load_state_dict(state_dict)
            model = model.to(args.device)
            model = nn.DataParallel(model, device_ids=args.device_ids)

            test_df_feature = processor.convert_examples_to_features(test_df, tokenizer, label_list=processor.get_labels(),
                max_length=args.max_seq_length, output_mode=args.output_mode,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )

            test_dataset = SsDataset(test_df_feature)
            print("-----------running test-----------")
            preds, _ = predict(args, model, test_dataset)

            test_df_submit = pd.DataFrame({"query_id": test_df["query_id"], "reply_id": test_df["reply_id"], "label": preds})
            print("test_df_submit num: %d" % len(test_df_submit))
            test_df_submit.to_csv(os.path.join(args.data_dir, save_file_name), sep='\t', header=None, index=None)

            print("The results of test-set is saved at %s" % os.path.join(args.data_dir, save_file_name))

