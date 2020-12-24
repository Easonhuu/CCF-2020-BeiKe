import argparse
import time
import os
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from transformers import glue_processors as processors

data_dir = "./data"

data_train_file = os.path.join(data_dir, "train.tsv")
data_dev_file = os.path.join(data_dir, "dev.tsv")
data_test_file = os.path.join(data_dir, "test.tsv")
output_dir = "./output"
log_dir = "./log"

# # debug
# data_train_file = os.path.join(data_dir, "train_small.tsv")
# data_dev_file = os.path.join(data_dir, "dev_small.tsv")
# data_test_file = os.path.join(data_dir, "test_small.tsv")
# output_dir = "./output-debug"
# log_dir = "./log-debug"

# model_type = "bert"
# model_specific = "chinese-bert-wwm-ext"
# tokenizer_name = "./pretrained_model/chinese-bert-wwm-ext/"
# model_name_or_path = "./pretrained_model/chinese-bert-wwm-ext/pytorch_model.bin"
# config_name = "./pretrained_model/chinese-bert-wwm-ext/config.json"

model_type = "xlnet"
model_specific = "chinese-xlnet-base"
tokenizer_name = "./pretrained_model/chinese-xlnet-base/"
model_name_or_path = "./pretrained_model/chinese-xlnet-base/pytorch_model.bin"
config_name = "./pretrained_model/chinese-xlnet-base/config.json"

# model_type = "roberta"
# model_specific = "chinese-roberta-wwm-ext-large"
# tokenizer_name = "./pretrained_model/chinese-roberta-wwm-ext-large/"
# model_name_or_path = "./pretrained_model/chinese-roberta-wwm-ext-large/pytorch_model.bin"
# config_name = "./pretrained_model/chinese-roberta-wwm-ext-large/config.json"

log_file = os.path.join(log_dir, model_specific + '_' + time.strftime("%Y-%m-%d_%H-%M", time.localtime()) + '.txt')

output_mode = "classification"
cache_dir = ""
do_train = True
do_eval = False
do_test = True
is_cv = True
do_lower_case = True
evaluate_during_training = False
max_seq_length = 64
learning_rate = 2e-5
epochs = 5
gpu_start = 0
n_gpu = 2
per_gpu_batch_size = 16
batch_size = per_gpu_batch_size * n_gpu 
early_stop_scale = 0.02
early_stop_num = 10
kfold = 5
gradient_accumulation_steps = 1
seed = 24
weight_decay = 0.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
warmup_steps = 0
local_rank = -1
loss_type = "ce"  # ['ce', 'fl']

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--data_dir",
    default=data_dir,
    type=str,
    # required=True,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
    "--data_train_file",
    default=data_train_file,
    type=str,
    help="",
)
parser.add_argument(
    "--data_dev_file",
    default=data_dev_file,
    type=str,
    help="",
)
parser.add_argument(
    "--data_test_file",
    default=data_test_file,
    type=str,
    help="",
)
parser.add_argument(
    "--output_dir",
    default=output_dir,
    type=str,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--log_dir",
    default=log_dir,
    type=str,
    help="The log directory where the model results will be written.",
)
parser.add_argument(
    "--model_type",
    default=model_type,
    type=str,
    help="Model type selected",
)
parser.add_argument(
    "--model_specific", 
    default=model_specific, 
    type=str
)
parser.add_argument(
    "--model_name_or_path",
    default=model_name_or_path,
    type=str,
    help="Path to pre-trained model or shortcut name selected",
)
parser.add_argument(
    "--config_name", 
    default=config_name, 
    type=str, 
    help="Pretrained config name or path if not the same as model_name",
)
parser.add_argument(
    "--tokenizer_name",
    default=tokenizer_name,
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--log_file", 
    default=log_file, 
    type=str, 
)
parser.add_argument(
    "--output_mode",
    default=output_mode,
    type=str,
    help="regression or classification",
)
parser.add_argument(
    "--cache_dir",
    default=cache_dir,
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--do_train", 
    default=do_train, 
    action="store_true", 
    help="Whether to run training."
)
parser.add_argument(
    "--do_eval", 
    default=do_eval, 
    action="store_true", 
    help="Whether to run eval on the dev set."
)
parser.add_argument(
    "--do_test", 
    default=do_test, 
    action="store_true", 
    help="Whether to run testing."
)
parser.add_argument(
    "--is_cv", 
    default=is_cv, 
    action="store_true", 
    help="Whether to run cross validation."
)
parser.add_argument(
    "--do_lower_case", 
    default=do_lower_case, 
    action="store_true", 
    help="Set this flag if you are using an uncased model.",
)
parser.add_argument(
    "--evaluate_during_training", 
    default=evaluate_during_training, 
    action="store_true", 
    help="Run evaluation during training at each logging step.",
)
parser.add_argument(
    "--max_seq_length",
    default=max_seq_length,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--learning_rate", 
    default=learning_rate, 
    type=float, 
    help="The initial learning rate for Adam."
)
parser.add_argument(
    "--epochs", 
    default=epochs, 
    type=int, 
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--gpu_start", 
    default=gpu_start, 
    type=int, 
)
parser.add_argument(
    "--n_gpu", 
    default=n_gpu, 
    type=int, 
)
parser.add_argument(
    "--per_gpu_batch_size", 
    default=per_gpu_batch_size, 
    type=int, 
    help="Batch size per GPU/CPU.",
)
parser.add_argument(
    "--batch_size", 
    default=batch_size, 
    type=int, 
    help="Batch size.",
)
parser.add_argument(
    "--early_stop_scale", 
    default=early_stop_scale, 
    type=float,
)
parser.add_argument(
    "--early_stop_num", 
    default=early_stop_num, 
    type=int,
)
parser.add_argument(
    "--kfold", 
    default=kfold, 
    type=int,
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=gradient_accumulation_steps,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--seed", 
    default=seed, 
    type=int, 
    help="random seed for initialization"
)
parser.add_argument(
    "--weight_decay", 
    default=weight_decay, 
    type=float, 
    help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon", 
    default=adam_epsilon, 
    type=float, 
    help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--max_grad_norm", 
    default=max_grad_norm, 
    type=float, 
    help="Max gradient norm."
)
parser.add_argument(
    "--warmup_steps", 
    default=warmup_steps, 
    type=int, 
    help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--local_rank", 
    type=int, 
    default=local_rank, 
    help="For distributed training: local_rank"
)
parser.add_argument(
    "--loss_type", 
    type=str, 
    default=loss_type, 
    help=""
)

args = parser.parse_args()
