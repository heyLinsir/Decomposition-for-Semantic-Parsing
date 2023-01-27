# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import json, csv
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    # AutoConfig,
    # AutoModelWithLMHead,
    # AutoTokenizer,
    MODEL_WITH_LM_HEAD_MAPPING,
    BertPreTrainedModel,
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerNorm,
    T5Stack,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from nltk.translate.bleu_score import sentence_bleu

logger = logging.getLogger(__name__)


def normalize(x):
    length = torch.sqrt((x * x).sum(dim=-1, keepdim=True) + 1e-15)
    return x / length


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "t5": (T5Config, T5ForConditionalGeneration, T5Tokenizer),
}

ORIGIN_VOCAB_SIZE = 32100
UNK_ID = 2

class BaseDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, num_examples=-1, mode='train'):
        self.tokenizer = tokenizer
        self.mode = mode
        assert self.mode in ['train', 'dev', 'test']

        assert os.path.isfile(file_path)

        data = pickle.load(open(file_path, "rb"))
        self.examples = []
        self.data = data
        self.file_path = file_path
        for item in tqdm(data):
            ###################
            # if 'synthesis_question_paraphrases_sub_sentences' not in item:
            #     continue
            # for paraphrase_sub_sentences in item['synthesis_question_paraphrases_sub_sentences']:
            #     paraphrase = paraphrase_sub_sentences['paraphrase']
            #     sub_sentences = paraphrase_sub_sentences['sub_sentences']
            ###################
            key = 'synthesis_question'
            if '%s_sub_sentences' % (key) not in item:
                continue
            paraphrases_sub_sentences = [(item[key], item['%s_sub_sentences' % (key)])]
            for paraphrase, sub_sentences in paraphrases_sub_sentences:
            ###################
                if sub_sentences == []:
                    continue

                input_text = paraphrase.strip()
                output_text = ' <FACT-SEP> '.join(sub_sentences).replace('\\', '').strip()

                input_ids = tokenizer(input_text).input_ids
                output_ids = tokenizer(output_text).input_ids

                if len(self.examples) < 5:
                    print('=' * 20)
                    print('input tokens:', input_text)
                    print('output tokens:', output_text)
                    print('input ids:', input_ids)
                    print('output ids:', output_ids)

                self.examples.append({
                        'input': input_ids,
                        'output': output_ids,
                    })

        if len(self.examples) > 0:
            print('Loaded data size:', len(self.examples))
            print('Original max input length:', max([len(example['input']) for example in self.examples]))
            print('Original max output length:', max([len(example['output']) for example in self.examples]))
            self.examples = [example for example in self.examples if len(example['output']) < 300 and len(example['input']) < 100]
            if num_examples > 0:
                self.examples = self.examples[:num_examples]
            print('Used data size:', len(self.examples))
            self.max_output_length = max([len(example['output']) for example in self.examples])
            print('Max input length:', max([len(example['input']) for example in self.examples]))
            print('Max output length:', self.max_output_length)
        else:
            print('No examples loaded from %s' % (file_path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        item = self.examples[item]

        input_ids = item['input']
        labels = item['output']

        return (input_ids, labels, item)

def load_and_cache_examples(args, tokenizer, mode='train', num_examples=-1):
    if mode == 'train':
        # file_path = './data/TACL/ComplexWebQuestions/composition/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-ComplexWebQuestions-7syn_temp0_topP1.5000'
        # file_path = './data/TACL/KQA/decomposition/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-7para_temp0_topP1.4600'
        # file_path = './data/TACL/GrailQA/decomposition/grailqa_v1.0_train.json.addCanonical.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-GrailQA-11para_temp0_topP1.7200'
        
        ## BREAK
        # file_path = './data/TACL/BREAK/QDMR/train.csv.pkl'

        ## text prompt
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-10T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-5T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14nature-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14nature-textPrompt_temp0_topP1'
        file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14syn-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_OPT-66b-KQA-14T5Para-textPrompt_greedy'
        # file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-CWQ-7syn-textPrompt_temp0_topP1'
        
        ## text prompt (codex paraphrase)
        # file_path = './data/TACL/KQA/Codex-paraphrase/decomposition/train.json.pkl.paraphrase_codex-paraphrase_temp1_topP0.9.decom_codex-KQA-14codexPara-textPrompt_temp0_topP1'
    elif mode == 'dev':
        # file_path = './data/TACL/ComplexWebQuestions/composition/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-ComplexWebQuestions-7syn_temp0_topP1.5000'
        # file_path = './data/TACL/KQA/decomposition/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-7para_temp0_topP1.4600'
        # file_path = './data/TACL/GrailQA/decomposition/grailqa_v1.0_train.json.addCanonical.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-GrailQA-11para_temp0_topP1.7200'
    
        ## BREAK
        # file_path = './data/TACL/BREAK/QDMR/dev.csv.pkl'

        ## text prompt
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-10T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-5T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14nature-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14nature-textPrompt_temp0_topP1'
        file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-14syn-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_OPT-66b-KQA-14T5Para-textPrompt_greedy'
        # file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-CWQ-7syn-textPrompt_temp0_topP1'
        
        ## text prompt (codex paraphrase)
        # file_path = './data/TACL/KQA/Codex-paraphrase/decomposition/train.json.pkl.paraphrase_codex-paraphrase_temp1_topP0.9.decom_codex-KQA-14codexPara-textPrompt_temp0_topP1'
    dataset = BaseDataset(tokenizer, args, file_path=file_path, num_examples=num_examples, mode=mode)
    # if mode == 'train':
    #     dataset.examples = dataset.examples[300:]
    # elif mode == 'dev':
    #     dataset.examples = dataset.examples[:300]
    # if mode == 'dev':
    #     dataset.examples = dataset.examples[::20]
    if mode == 'train':
        dataset.examples = dataset.examples[:5000]
    elif mode == 'dev':
        dataset.examples = dataset.examples[5000:5300]
    # if mode == 'train':
    #     dataset.examples = dataset.examples[:2700]
    # elif mode == 'dev':
    #     dataset.examples = dataset.examples[2700:3000]
    return dataset


class DecomposeDataset(Dataset):
    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)

        self.file_path = file_path
        self.load()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def load(self):
        if os.path.isfile('%s.T5-large-decom' % (self.file_path)):
            self.examples = pickle.load(open('%s.T5-large-decom' % (self.file_path), 'rb'))
        else:
            self.examples = pickle.load(open(self.file_path, 'rb'))

    def save(self):
        pickle.dump(self.examples, open('%s.T5-large-decom' % (self.file_path), 'wb'))

def load_and_cache_examples_decompose():
    # file_path = './data/TACL/ComplexWebQuestions/composition/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w'
    # file_path = './data/TACL/KQA/decomposition/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w'
    # file_path = './data/TACL/GrailQA/decomposition/grailqa_v1.0_train.json.addCanonical.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w'
    
    ## BREAK
    # file_path = './data/TACL/KQA/decomposition/BREAK/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.0-18876'
    # file_path = './data/TACL/KQA/decomposition/BREAK/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.18876-37752'
    # file_path = './data/TACL/KQA/decomposition/BREAK/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.37752-56628'
    # file_path = './data/TACL/KQA/decomposition/BREAK/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.56628-75504'
    # file_path = './data/TACL/KQA/decomposition/BREAK/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.75504-94380'
    # file_path = './data/TACL/KQA/decomposition/BREAK/val.json.dev.pkl'
    # file_path = './data/TACL/KQA/decomposition/BREAK/val.json.test.pkl'
    # file_path = './data/TACL/ComplexWebQuestions/composition/BREAK/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w'
    # file_path = './data/TACL/ComplexWebQuestions/composition/BREAK/ComplexWebQuestions_dev.json.pkl'
    # file_path = './data/TACL/ComplexWebQuestions/composition/BREAK/ComplexWebQuestions_test.json.pkl'

    ## text prompt
    # file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.0-18876'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.18876-37752'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.37752-56628'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.56628-75504'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.75504-94380'
    file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/val.json.test.pkl'
    
    ## text prompt (random prompt)
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.0-18876'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.18876-37752'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.37752-56628'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.56628-75504'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.75504-94380'

    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.0-18876'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.18876-37752'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.37752-56628'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.56628-75504'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.75504-94380'

    ## text prompt (10 prompt)
    # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.0-18876'
    # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.18876-37752'
    # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.37752-56628'
    # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.56628-75504'
    # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.75504-94380'

    ## text prompt (5 prompt)
    # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.0-18876'
    # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.18876-37752'
    # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.37752-56628'
    # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.56628-75504'
    # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.75504-94380'

    ## text prompt (14 nature prompt)
    # file_path = './data/TACL/KQA/decomposition-14nature-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w'

    ## text prompt (14 syn prompt)
    # file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.0-23594'
    # file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.23594-47188'
    # file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.47188-70782'
    # file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.70782-94376'

    ## text prompt (OPT prompt)
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.0-18876'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.18876-37752'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.37752-56628'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.56628-75504'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.75504-94380'
    
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/val.json.dev.pkl'
    # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/val.json.test.pkl'

    ## text prompt (codex paraphrase)
    # file_path = './data/TACL/KQA/Codex-paraphrase/decomposition/train.json.pkl.paraphrase_codex-paraphrase_temp1_topP0.9.0-18876'
    # file_path = './data/TACL/KQA/Codex-paraphrase/decomposition/train.json.pkl.paraphrase_codex-paraphrase_temp1_topP0.9.18876-37752'
    # file_path = './data/TACL/KQA/Codex-paraphrase/decomposition/train.json.pkl.paraphrase_codex-paraphrase_temp1_topP0.9.37752-56628'
    # file_path = './data/TACL/KQA/Codex-paraphrase/decomposition/train.json.pkl.paraphrase_codex-paraphrase_temp1_topP0.9.56628-75504'
    # file_path = './data/TACL/KQA/Codex-paraphrase/decomposition/train.json.pkl.paraphrase_codex-paraphrase_temp1_topP0.9.75504-94380'
    
    dataset = DecomposeDataset(file_path=file_path)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def collate(examples):
    input_ids = []
    labels = []
    items = []

    for example in examples:
        input_ids.append(torch.tensor(example[0]).long())
        labels.append(torch.tensor(example[1]).long())
        items.append(example[2])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return (input_ids.long(), labels.long(), items)

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir + '-log')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    # if (
    #     args.model_name_or_path
    #     and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
    #     and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    # if args.model_name_or_path and os.path.exists(args.model_name_or_path):
    #     try:
    #         # set global_step to gobal_step of last saved checkpoint from model path
    #         checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
    #         global_step = int(checkpoint_suffix)
    #         epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #         steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info("  Continuing training from epoch %d", epochs_trained)
    #         logger.info("  Continuing training from global step %d", global_step)
    #         logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    #     except ValueError:
    #         logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    best_bleu4, early_stop_indicator = -1, 0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, (input_ids, labels, items) in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            model.train()

            loss = model(input_ids=input_ids, labels=labels).loss
            tr_loss += loss.item() / args.gradient_accumulation_steps

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training and global_step % 500 == 0
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, mode='dev')
                        if 'KQA' in train_dataset.file_path:
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_KQA.decom.dev_%s" % (key), value, global_step)
                        elif 'ComplexWebQuestions' in train_dataset.file_path:
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_ComplexWebQuestions.decom.dev_%s" % (key), value, global_step)
                        elif 'GrailQA' in train_dataset.file_path:
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_GrailQA.decom.dev_%s" % (key), value, global_step)
                        elif 'BREAK' in train_dataset.file_path:
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_BREAK.decom.dev_%s" % (key), value, global_step)
                        else:
                            raise
                        bleu4 = results['bleu4']

                        if bleu4 > best_bleu4:
                            best_bleu4 = bleu4
                            early_stop_indicator = 0

                            output_dir = os.path.join(args.output_dir, "best_bleu4")
                            os.makedirs(output_dir, exist_ok=True)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Update best model (bleu4: %f) checkpoint to %s" % (best_bleu4, output_dir))
                        else:
                            early_stop_indicator += 1
                            if early_stop_indicator >= 10:
                                return global_step, tr_loss / global_step
                        
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss


                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="", dataset='unified', mode='None', do_generate=True) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, mode=mode)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = []
    eval_f1 = []
    eval_em = []
    eval_bleu1, eval_bleu2, eval_bleu3, eval_bleu4 = [], [], [], []
    model.eval()

    def decode(ids, tokenizer):
        sentence = tokenizer.decode(ids, skip_special_tokens=False)
        sentence = sentence.replace('</s>', ' ').replace('<pad>', ' ')
        sentence = ' '.join([word for word in sentence.split(' ') if word != ''])
        return sentence.strip()

    def calculate_bleu(candidate, references):
        references = [reference.split(' ') for reference in references]
        candidate = candidate.split(' ')
        bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu(references, candidate, weights=(0, 1, 0, 0))
        bleu3 = sentence_bleu(references, candidate, weights=(0, 0, 1, 0))
        bleu4 = sentence_bleu(references, candidate, weights=(0, 0, 0, 1))

        return bleu1, bleu2, bleu3, bleu4

    def normalize_text(s):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_exact_match(prediction, truth):
        return int(normalize_text(prediction) == normalize_text(truth))

    def compute_f1(prediction, truth):
        pred_tokens = normalize_text(prediction).split()
        truth_tokens = normalize_text(truth).split()
        
        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        
        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0
        
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        
        return 2 * (prec * rec) / (prec + rec)

    record = []
    for example_id, (input_ids, labels, item) in tqdm(enumerate(eval_dataloader)):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            model_outputs = model(input_ids=input_ids, labels=labels)
            loss = model_outputs.loss
            eval_loss.append(loss.item())

            if do_generate:
                if True:
                    num_return_sequences = 1
                    outputs = model.generate(
                                input_ids=input_ids,
                                max_length=eval_dataset.max_output_length + 64,
                                num_beams=5,
                                eos_token_id=1,
                                num_return_sequences=num_return_sequences,
                            )
                else:
                    num_return_sequences = 20
                    outputs = model.generate(
                                input_ids=input_ids,
                                max_length=eval_dataset.max_output_length + 64,
                                do_sample=True,
                                top_p=0.9,
                                eos_token_id=1,
                                num_return_sequences=num_return_sequences
                            )
                outputs = outputs.view(input_ids.size(0), num_return_sequences, outputs.size(1))
                for _input_ids, _outputs, _labels in zip(input_ids, outputs, labels):
                    predictions = [decode(output, tokenizer) for output in _outputs]
                    _labels = _labels.tolist()
                    if -100 in _labels:
                        _labels = _labels[:_labels.index(-100)]
                    ground_truth = decode(_labels, tokenizer)

                    metrics = [(compute_f1(prediction, ground_truth), compute_exact_match(prediction, ground_truth), calculate_bleu(prediction, [ground_truth])) for prediction in predictions]
                    metrics = sorted(list(zip(metrics, predictions)), key=lambda x: x[0][0], reverse=True)
                    predictions = [metric[1] for metric in metrics]
                    metric = metrics[0][0]

                    print('=' * 20)
                    try:
                        print('Question    :', decode(_input_ids, tokenizer))
                        print('Prediction  :', predictions[0])
                        print('Ground Truth:', ground_truth)
                    except:
                        pass
                    if metric[1] == 0:
                        print('False')
                    else:
                        print('True')

                    eval_f1.append(metric[0])
                    eval_em.append(metric[1])
                    eval_bleu1.append(metric[2][0])
                    eval_bleu2.append(metric[2][1])
                    eval_bleu3.append(metric[2][2])
                    eval_bleu4.append(metric[2][3])

                    record.append({
                                    'question': decode(_input_ids, tokenizer),
                                    'prediction': predictions[0],
                                    'answer': ground_truth,
                                })

                    # pickle.dump(record, open(os.path.join(args.output_dir, '%s.inference' % (mode)), 'wb'))
                    # print('save inference results to', os.path.join(args.output_dir, '%s.inference' % (mode)))

    if do_generate:
        result = {
                    "loss": np.mean(eval_loss),
                    "f1": np.mean(eval_f1),
                    "EM": np.mean(eval_em),
                    "bleu1": np.mean(eval_bleu1),
                    "bleu2": np.mean(eval_bleu2),
                    "bleu3": np.mean(eval_bleu3),
                    "bleu4": np.mean(eval_bleu4),
                }
    else:
        result = {
                "loss": np.mean(eval_loss),
            }

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return result


def collate_generate(examples):
    return [example for example in examples if 'synthesis_question_paraphrases_sub_sentences' not in example]

def generate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples_decompose()

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_generate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Generation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    def decode(ids, tokenizer):
        sentence = tokenizer.decode(ids, skip_special_tokens=False)
        sentence = sentence.replace('</s>', ' ').replace('<pad>', ' ')
        sentence = ' '.join([word for word in sentence.split(' ') if word != ''])
        return sentence

    for example_id, items in tqdm(enumerate(eval_dataloader)):
        if items == []:
            continue
        paraphrases = []
        for item in items:
            paraphrases.extend(item['synthesis_question_paraphrases'][:5])
        input_ids = [torch.tensor(tokenizer(paraphrase).input_ids).long() for paraphrase in paraphrases]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).long().to(args.device)
        
        with torch.no_grad():
            outputs = model.generate(
                                input_ids=input_ids,
                                max_length=300,
                                num_beams=5,
                                eos_token_id=1,
                                num_return_sequences=1,
                            )

            sub_sentences_list = [decode(output, tokenizer) for output in outputs]
            sub_sentences_id = 0
            for item in items:
                item['synthesis_question_paraphrases_sub_sentences'] = []
                for paraphrase in item['synthesis_question_paraphrases'][:5]:
                    item['synthesis_question_paraphrases_sub_sentences'].append({'paraphrase': paraphrase, 'sub_sentences': sub_sentences_list[sub_sentences_id]})
                    print('=' * 20)
                    print('question : %s' % (paraphrase))
                    print('decompose: %s' % (sub_sentences_list[sub_sentences_id]))
                    sub_sentences_id += 1
            assert sub_sentences_id == len(paraphrases)
            
            if example_id % 100 == 0:
                eval_dataset.save()

    eval_dataset.save()

def collate_generate_for_origin_question(examples):
    return examples

def generate_for_origin_question(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples_decompose()

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_generate_for_origin_question
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Generation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    def decode(ids, tokenizer):
        sentence = tokenizer.decode(ids, skip_special_tokens=False)
        sentence = sentence.replace('</s>', ' ').replace('<pad>', ' ')
        sentence = ' '.join([word for word in sentence.split(' ') if word != ''])
        return sentence

    # key = 'synthesis_question'
    key = 'nature_question'
    for example_id, items in tqdm(enumerate(eval_dataloader)):
        if items == []:
            continue
        paraphrases = []
        for item in items:
            paraphrases.append(item[key])
        input_ids = [torch.tensor(tokenizer(paraphrase).input_ids).long() for paraphrase in paraphrases]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).long().to(args.device)
        
        with torch.no_grad():
            outputs = model.generate(
                                input_ids=input_ids,
                                max_length=300,
                                num_beams=5,
                                eos_token_id=1,
                                num_return_sequences=1,
                            )

            sub_sentences_list = [decode(output, tokenizer) for output in outputs]
            for item, sub_sentences in zip(items, sub_sentences_list):
                item['%s_sub_sentences' % (key)] = sub_sentences.split(' <FACT-SEP> ')
                print('=' * 20)
                print('question : %s' % (item[key]))
                print('decompose:', item['%s_sub_sentences' % (key)])
            assert len(sub_sentences_list) == len(items)
            
            if example_id % 100 == 0:
                eval_dataset.save()

    eval_dataset.save()

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--domain_name", default=None, type=str, required=True, help="The domain name (e.g. books, music, ...)."
    )
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_dataset",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--train_from_scratch", action="store_true", help="Whether to train from scratch.")
    parser.add_argument("--constrain_generation", action="store_true", help="Whether to add constrain to generation process.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--contrastive_temperature", type=float, default=1., help="The temperature of contrastive paraphrase loss.")
    parser.add_argument("--paraphrase_weight", type=float, default=0., help="The weight of paraphrase loss.")
    parser.add_argument("--num_examples", type=int, default=-1, help="The number of used training examples.")
    parser.add_argument("--factor_previous_examples", default=0.2, type=float, help="The factor of examples from pretraining tasks.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
    #     raise ValueError(
    #         "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
    #         "flag (masked language modeling)."
    #     )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    AutoConfig, AutoModelWithLMHead, AutoTokenizer = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    print(len(tokenizer))
    addition_tokens = ['<FACT-SEP>']
    print(addition_tokens)
    if len(tokenizer) == ORIGIN_VOCAB_SIZE:
        for addition_token in addition_tokens:
            tokenizer.add_tokens(addition_token)
    print(len(tokenizer))
    assert len(tokenizer) == ORIGIN_VOCAB_SIZE + len(addition_tokens)
    

    if args.model_name_or_path and not args.train_from_scratch:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        # model = AutoModelWithLMHead.from_config(config)
        model = AutoModelWithLMHead(config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        # train_dataset = eval('load_and_cache_examples_Schema2QA_%s' % (args.domain_name))(args, tokenizer, mode='train', num_examples=args.num_examples)
        # train_dataset = load_and_cache_examples_ParaBank2(args, tokenizer, mode='train', num_examples=args.num_examples)
        # train_dataset = load_and_cache_examples_KQA(args, tokenizer, mode='train', num_examples=args.num_examples)
        # train_dataset = load_and_cache_examples_Overnight_socialnetwork(args, tokenizer, mode='train', num_examples=args.num_examples)
        train_dataset = load_and_cache_examples(args, tokenizer, mode='train', num_examples=args.num_examples)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            # result = evaluate(args, model, tokenizer, dataset='KQA', mode='dev')
            # result = console(args, model, tokenizer, dataset='KQA', mode='dev')
            # result = evaluate_sts(args, model, tokenizer, mode='dev')
            # result = evaluate_sts1(args, model, tokenizer, mode='dev')
            # generate(args, model, tokenizer)
            generate_for_origin_question(args, model, tokenizer)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
