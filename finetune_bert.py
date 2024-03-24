""" Код для обучения BERT с небольшими изменениями взят отсюда: https://github.com/sazzy4o/ualberta-lscdiscovery/blob/main/graded_change/models/context_dependent.py """


from models.utils.bert_utils import CategoricalProcessor, convert_examples_to_features
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from scipy.spatial.distance import cdist
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset

import pandas as pd
import numpy as np
import torch
import os
import io

from transformers import MT5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForMaskedLM

def load_pretrained_bert(bert_name, device):
    """ Loads a pretrained BERT model from cloud storage for binary classification finetuning. """

    config = BertConfig.from_pretrained(bert_name, num_labels=3) #problem_type="multi_label_classification"
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    model = BertForSequenceClassification.from_pretrained(bert_name, config=config)

    model.to(device)

    return tokenizer, model


def load_dataset(tokenizer, bert_name, filepath, max_sql=128):
    """ Loads and prepares a local dataset for a BERT model. """

    set_type = filepath.split("/")[-1][:-4]

    processor = CategoricalProcessor()

    label_list = processor.get_labels()
    examples = processor.get_examples(filepath, set_type)

    features = convert_examples_to_features(examples, label_list, max_sql,
        tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset

def load_local_bert(bert_dir, device, output_hidden_states=False):
    """ Loads a finetuned BERT model from local storage. """

    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    model = BertForSequenceClassification.from_pretrained(bert_dir, output_hidden_states=output_hidden_states)

    model.to(device)

    return tokenizer, model


def finetune_bert(experiment_dir, device="cpu", bert_name="bert-base-multilingual-cased", masked=True, **params):
    """ Finetunes a pretrained BERT model on a sentence time classification objective. """

    bert_dir = experiment_dir + "bert/"
    os.makedirs(bert_dir, exist_ok=True)

    device = torch.device(device)

    print("Loading pretrained BERT model ...")
    tokenizer, model = load_pretrained_bert(bert_name, device)

    if masked:
        train_fp = experiment_dir + "preprocessed_texts/train_masked.tsv"
    else:
        train_fp = experiment_dir + "preprocessed_texts/train.tsv"

    # train bert model
    train_dataset = load_dataset(tokenizer, bert_name, train_fp)
    train_bert(train_dataset, model, tokenizer, device, **params)

    # save bert model
    model.save_pretrained(bert_dir)
    tokenizer.save_pretrained(bert_dir)

    # reload bert model and test dataset
    tokenizer, model = load_local_bert(bert_dir, device)
    test_dataset = load_dataset(tokenizer, bert_name, experiment_dir + "preprocessed_texts/train.tsv")

    # evaluate bert model and save results
    acc = test_bert(test_dataset, model, tokenizer, device, **params)
    np.save(bert_dir + "classification_accuracy.npy", np.round(acc, decimals=2))


def train_bert(train_dataset, model, tokenizer, device, n_epochs=1, batch_size=10, learning_rate=4e-5, warmup_ratio=0.05, **kwargs):
    """ Trains a BERT model on a train dataset. """

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    t_total = len(train_dataloader) * n_epochs

    optimizer_params = [{"params": [p for n, p in model.named_parameters()], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_params, lr=learning_rate, eps=1e-8)

    warmup_steps = int(warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total) #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    model.zero_grad()
    model.train()

    for _ in tqdm(range(n_epochs), desc="BERT Training"):

        for step, batch in enumerate(train_dataloader): #enumerate(tqdm(train_dataloader, desc="Current Epoch")):

            batch = tuple(t.to(device) for t in batch)

            inputs = {"input_ids":      batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels":         batch[3]}

            outputs = model(**inputs)

            loss = outputs[0]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()


def test_bert(test_dataset, model, tokenizer, device, batch_size=10, **kwargs):
    """ Evaluates a BERT model on a test dataset. """

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    preds = None
    model.eval()

    for batch in tqdm(test_dataloader, desc="BERT Testing"):

        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():

            inputs = {"input_ids":      batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels":         batch[3]}

            logits = model(**inputs)[1]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    accuracy = (out_label_ids == preds).sum() / preds.shape[0]

    return accuracy
  
