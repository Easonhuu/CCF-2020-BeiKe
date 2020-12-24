python main.py --kfold 5 --max_seq_length 64 --epochs 5 --n_gpu 2 --gpu_start 0 --per_gpu_batch_size 16 --loss_type 'ce' \
               --model_type "bert" --model_specific "chinese-bert-wwm-ext" --tokenizer_name "./pretrained_model/chinese-bert-wwm-ext/" \
               --model_name_or_path "./pretrained_model/chinese-bert-wwm-ext/pytorch_model.bin" \
               --config_name "./pretrained_model/chinese-bert-wwm-ext/config.json"

python main.py --kfold 5 --max_seq_length 64 --epochs 5 --n_gpu 2 --gpu_start 0 --per_gpu_batch_size 16 --loss_type 'ce' \
               --model_type "roberta" --model_specific "chinese-roberta-wwm-ext" --tokenizer_name "./pretrained_model/chinese-roberta-wwm-ext/" \
               --model_name_or_path "./pretrained_model/chinese-roberta-wwm-ext/pytorch_model.bin" \
               --config_name "./pretrained_model/chinese-roberta-wwm-ext/config.json"

python main.py --kfold 5 --max_seq_length 128 --epochs 5 --n_gpu 2 --gpu_start 0 --per_gpu_batch_size 16 --loss_type 'ce' \
               --model_type "roberta" --model_specific "chinese-roberta-wwm-ext-large" --tokenizer_name "./pretrained_model/chinese-roberta-wwm-ext-large/" \
               --model_name_or_path "./pretrained_model/chinese-roberta-wwm-ext-large/pytorch_model.bin" \
               --config_name "./pretrained_model/chinese-roberta-wwm-ext-large/config.json"

python main.py --kfold 5 --max_seq_length 128 --epochs 5 --n_gpu 2 --gpu_start 0 --per_gpu_batch_size 16 --loss_type 'fl' \
               --model_type "bert" --model_specific "chinese-bert-wwm-ext" --tokenizer_name "./pretrained_model/chinese-bert-wwm-ext/" \
               --model_name_or_path "./pretrained_model/chinese-bert-wwm-ext/pytorch_model.bin" \
               --config_name "./pretrained_model/chinese-bert-wwm-ext/config.json"

python main.py --kfold 5 --max_seq_length 64 --epochs 5 --n_gpu 2 --gpu_start 0 --per_gpu_batch_size 16 --loss_type 'fl' \
               --model_type "roberta" --model_specific "chinese-roberta-wwm-ext" --tokenizer_name "./pretrained_model/chinese-roberta-wwm-ext/" \
               --model_name_or_path "./pretrained_model/chinese-roberta-wwm-ext/pytorch_model.bin" \
               
               --config_name "./pretrained_model/chinese-roberta-wwm-ext/config.json"

