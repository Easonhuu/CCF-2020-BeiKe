import copy
import time
import os
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

def train(args, model, dataset_list, kfold=None):
    is_eval = args.evaluate_during_training
    if kfold or is_eval:
        train_dataset, valid_dataset = dataset_list
    else:
        train_dataset = dataset_list
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # optimizer = optim.Adam(optimizer_grouped_parameters, 10 ** -args.learning_rate, weight_decay=10 ** -args.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    best_param = {}
    if kfold or is_eval:
        best_model = copy.deepcopy(model)
        best_param["epoch"] = 0
        best_param["train_loss"] = 9e8
        best_param["valid_loss"] = 9e8
        best_param["train_f1"] = -9e8
        best_param["valid_f1"] = -9e8

    plot_loss = []
    plot_f1 = []

    early_stop = 0  # used to stop early
    real_epochs = 0  # used to record real training epochs
    model.zero_grad()
    for epoch in range(args.epochs):
        tr_loss = 0.0
        preds = None
        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            model.train()
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            labels = labels.to(args.device)

            if args.model_type != "distilbert":
                token_type_ids = (
                    token_type_ids if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

            if args.loss_type == "ce":
                loss, logits = outputs[:2]
            elif args.loss_type == "fl":
                _, logits = outputs[:2]
                focal_loss = 0
                for true_label, pred_label in zip(labels, logits):
                    pred_label = pred_label - torch.max(pred_label)
                    exp_pred_label = torch.exp(pred_label)
                    softmax_pred_label = exp_pred_label / torch.sum(exp_pred_label)
                    p = softmax_pred_label[true_label]
                    focal_loss += -0.6 * (1-p)**3 * torch.log(p)
                loss = focal_loss
            else:
                sys.exit(-1)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(train_dataloader) <= args.gradient_accumulation_steps
                and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        train_loss = tr_loss / (step+1)
        train_f1 = f1_score(out_label_ids, preds)

        real_epoch = epoch+1
        if kfold or is_eval:
            eval_loss, eval_f1, _, _ = evaluate(args, model, valid_dataset)
            if eval_f1 > best_param["valid_f1"]:
                best_model = copy.deepcopy(model)
                best_param["epoch"] = real_epoch
                best_param["train_loss"] = train_loss
                best_param["valid_loss"] = eval_loss
                best_param["train_f1"] = train_f1
                best_param["valid_f1"] = eval_f1
        end_time = time.time()
        if kfold:
            kfold_log = "------------KFold %d------------" % (kfold)
            print(kfold_log)
        if kfold or is_eval:
            log = 'epoch:{}, train_loss:{:.3f}, train_f1:{:.3f}, valid_loss:{:.3f}, valid_f1:{:.3f}'.format(
                            real_epoch, train_loss, train_f1, eval_loss, eval_f1)
        else:
            log = 'epoch:{}, train_loss:{:.3f}, train_f1:{:.3f}'.format(real_epoch, train_loss, train_f1)
        each_epoch_time = "------------The {} epoch spend {}m-{:.3f}s------------".format(real_epoch, int((end_time-start_time)/60), (end_time-start_time)%60)
        print(log)
        print(each_epoch_time)

        with open(args.log_file, 'a') as pickle_file:
            if kfold:
                pickle_file.write(kfold_log+'\n')
            pickle_file.write(log+'\n')
            pickle_file.write(each_epoch_time+'\n')

        real_epochs = real_epochs + 1
        if kfold or is_eval:
            plot_loss.append([real_epoch, train_loss, eval_loss])
            plot_f1.append([real_epoch, train_f1, eval_f1])

            if epoch != 0:
                if abs(last_valid_loss - eval_loss)/last_valid_loss <= args.early_stop_scale or eval_loss > last_valid_loss: 
                    early_stop = early_stop+1
                else:
                    early_stop = 0
            if early_stop == args.early_stop_num:
                break
            last_valid_loss = eval_loss
        else:
            plot_loss.append([real_epoch, train_loss])
            plot_f1.append([real_epoch, train_f1])
    
    if kfold:
        dir_save = os.path.join(args.output_dir, args.model_specific + '_' + '{}fold'.format(args.kfold), 'kfold_'+str(kfold))
    elif is_eval:
        dir_save = os.path.join(args.output_dir, args.model_specific + "-{:.3f}-{}-{}".format(best_param["valid_f1"], best_param['epoch'], real_epochs))
    else:
        dir_save = os.path.join(args.output_dir, args.model_specific + "-{}".format(real_epochs))    
        best_model = copy.deepcopy(model)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
        print(dir_save+" create successful!")
    else:
        print(dir_save+" already exists.")
    
    model_to_save = best_model.module if hasattr(best_model, "module") else best_model # Take care of distributed/parallel training
    if kfold or is_eval:
        torch.save(model_to_save.state_dict(), dir_save+'/best-model-{:.3f}-{}-{}.pth'.format(best_param["valid_f1"], best_param['epoch'], real_epochs))
    else:
        torch.save(model_to_save.state_dict(), dir_save+'/best-model-{}.pth'.format(real_epochs))

    os.system("cp " + args.log_file + " " + dir_save)
    os.remove(args.log_file)
    os.system("cp " + __file__ + " " + dir_save)
    with open(os.path.join(dir_save, 'args.json'), 'w') as f:
        args_dict = args.__dict__
        args_dict_new = {}
        for k, v in args_dict.items():
            try:
                json.dumps({k:v})
                args_dict_new[k] = v
            except:
                continue
        json.dump(args_dict_new, f, indent=2)

    with open(os.path.join(dir_save, 'model_parameters.txt'), 'w') as f:
        for param_name, param_value in model_to_save.named_parameters():
            print(param_name, ":", param_value.size(), file=f)
        print('Model parameters:', sum(param.numel() for param in model.parameters()), file=f)

    return plot_loss, plot_f1, best_param, best_model, real_epochs, dir_save

def evaluate(args, model, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    preds = None
    nb_eval_steps = 0
    eval_loss = 0
    for _, batch in enumerate(eval_dataloader):
        model.eval()
        input_ids, attention_mask, token_type_ids, labels = batch
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        labels = labels.to(args.device)

        if args.model_type != "distilbert":
            token_type_ids = (
                token_type_ids if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    f1 = f1_score(out_label_ids, preds)
    return eval_loss, f1, preds, out_label_ids

def predict(args, model, test_dataset):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    preds = None
    for _, batch in enumerate(test_dataloader):
        model.eval()
        input_ids, attention_mask, token_type_ids, labels = batch
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        labels = labels.to(args.device)
        
        with torch.no_grad():
            if args.model_type != "distilbert":
                token_type_ids = (
                    token_type_ids if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            logits = outputs[1]

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    if args.output_mode == "classification":
        preds_label = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds_label = np.squeeze(preds)
    return preds_label, preds