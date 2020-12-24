# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
import logging
import torch
import pandas as pd
# from data_augment import data_Augment 

from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class SsDataset(Dataset):
    def __init__(self, df):
        self.examples = df

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        df = self.examples.iloc[index]
        input_ids = torch.tensor(df['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(df['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(df['token_type_ids'], dtype=torch.long)
        labels = torch.tensor(df['label'], dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, labels

class DataProcessor():
    def convert_examples_to_features(
        self,
        examples,
        tokenizer,
        max_length=512,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
    ):

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        input_ids_list=[]
        attention_mask_list=[]
        token_type_ids_list=[]
        label_list=[]
        for _, example in examples.iterrows():
            """
            inputs:
            {
                input_ids : 每个token的id
                attention_mask : 非填充部分的token对应1
                token_type_ids : 分段token索引，类似segment embedding（对于句子对任务 属于句子A的token为0，句子B的token为1，对于分类任务，只有一个输入句子 全为0）
            }
            """
            if pd.isna(example['question']) or pd.isna(example['sentence']):
                continue
            inputs = tokenizer.encode_plus(
                example['question'], example['sentence'], add_special_tokens=True, max_length=max_length, return_token_type_ids=True, truncation=True
            )

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            if output_mode == "classification":
                label = label_map[example['label']]
            elif output_mode == "regression":
                label = float(example['label'])
            else:
                raise KeyError(output_mode)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            token_type_ids_list.append(token_type_ids)
            label_list.append(label)
        features = pd.DataFrame({'input_ids': input_ids_list, 'attention_mask': attention_mask_list, 
                                'token_type_ids': token_type_ids_list, 'label': label_list})

        return features

    def get_examples_from_file(self, args, train_dev_test):
        if train_dev_test == 'train':
            path = args.data_train_file
        elif train_dev_test == 'dev':
            path = args.data_dev_file
        elif train_dev_test == 'test':
            path = args.data_test_file
        df = pd.read_csv(path, sep='\t')
        if train_dev_test == 'test':
            df['label'] = 0
        return df

    def get_labels(self):
        """See base class."""
        return [0, 1]

def load_and_cache_examples(args, tokenizer, train_dev_test, output_mode):
    processor = DataProcessor()
    logger.info("Reading raw data from dataset file at %s", args.data_dir)
    examples_df = processor.get_examples_from_file(args, train_dev_test)

    return examples_df

def load_and_cache_features(args, task, tokenizer, train_dev_test, output_mode):
    processor = DataProcessor()
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            train_dev_test,
            args.model_type,
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features_df = pickle.load(f)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        df = processor.get_examples_from_file(args, train_dev_test)
        features_df = processor.convert_examples_to_features(
            df,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        with open(cached_features_file, 'wb') as f:
            pickle.dump(features_df, f)

    return features_df

def ss_df(args, tokenizer):
    train_df = load_and_cache_examples(args, tokenizer, train_dev_test = "train", output_mode = "regression")
    test_df = load_and_cache_examples(args, tokenizer, train_dev_test = "test", output_mode = "regression")

    return train_df, test_df

