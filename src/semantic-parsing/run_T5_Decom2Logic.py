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
    AutoConfig,
    AutoModelWithLMHead,
    AutoModel,
    AutoTokenizer,
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

from opendelta import SoftPromptModel, LoraModel

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

def shuffle_entities(sentence):
    max_id = 1
    for i in list(range(1, 31, 1))[::-1]:
        if 'ENTITY%d' % (i) in sentence:
            max_id = i
    original_ids = list(range(1, max_id + 1, 1))
    shuffled_ids = list(range(1, max_id + 1, 1))
    random.shuffle(shuffled_ids)
    for original_id, shuffled_id in zip(original_ids[::-1], shuffled_ids[::-1]):
        sentence = sentence.replace('ENTITY%d' % (original_id), 'TEMP%d' % (shuffled_id))
    for shuffled_id in shuffled_ids[::-1]:
        sentence = sentence.replace('TEMP%d' % (shuffled_id), 'ENTITY%d' % (shuffled_id))
    return sentence

def shuffle_facts(sentence):
    facts = sentence.split(' <FACT-SEP> ')
    question = facts[-1]
    facts = facts[:-1]

    shuffled_ids = list(range(1, len(facts) + 1, 1))
    random.shuffle(shuffled_ids)
    new_facts = [None] * len(facts)
    for original_id, shuffled_id in enumerate(shuffled_ids):
        new_facts[shuffled_id - 1] = facts[original_id]
    sentence = ' <FACT-SEP> '.join(new_facts + [question])

    original_ids = list(range(1, len(facts) + 1, 1))
    for original_id, shuffled_id in zip(original_ids[::-1], shuffled_ids[::-1]):
        sentence = sentence.replace('FACT%d' % (original_id), 'TEMP%d' % (shuffled_id))
    for shuffled_id in shuffled_ids[::-1]:
        sentence = sentence.replace('TEMP%d' % (shuffled_id), 'FACT%d' % (shuffled_id))
    return sentence

class DecomDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, num_examples=-1, mode='train'):
        self.tokenizer = tokenizer
        assert mode in ['train', 'dev', 'test']

        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached-Decom2Logic_" + filename
        )

        if False and os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            data = pickle.load(open(file_path, "rb"))
            self.data = data

            self.examples = []
            num_nature_training_examples = 0
            for item in tqdm(data):
                program = item['program'].strip()
                program_ids = tokenizer(program).input_ids
                questions = []
                if mode == 'train':
                    # for paraphrase_item in item['synthesis_question_paraphrases_sub_sentences'][:5]:
                    #     sub_sentences = paraphrase_item['sub_sentences']
                    #     if type(sub_sentences) == list:
                    #         sub_sentences = ' <FACT-SEP> '.join(sub_sentences).replace('\\', '').strip()
                    #     else:
                    #         sub_sentences = sub_sentences.strip()
                    #     questions.append(sub_sentences)
                        # questions[-1] = ' <FACT-SEP> '.join([paraphrase_item['paraphrase'], questions[-1]])
                    if 'nature_question_sub_sentences' in item and num_nature_training_examples < 50000:
                        sub_sentences = item['nature_question_sub_sentences']
                        if type(sub_sentences) == list:
                            sub_sentences = ' <FACT-SEP> '.join(sub_sentences).replace('\\', '').strip()
                        else:
                            sub_sentences = sub_sentences.strip()
                        questions.append(sub_sentences)
                        num_nature_training_examples += 1
                elif mode == 'dev':
                    if 'nature_question_sub_sentences' in item:
                        sub_sentences = item['nature_question_sub_sentences']
                        if type(sub_sentences) == list:
                            sub_sentences = ' <FACT-SEP> '.join(sub_sentences).replace('\\', '').strip()
                        else:
                            sub_sentences = sub_sentences.strip()
                        questions.append(sub_sentences)
                        # questions[-1] = ' <FACT-SEP> '.join([item['synthesis_question'], questions[-1]])
                    # if 'nature_question_sub_sentences' in item:
                    #     sub_sentences = ' <FACT-SEP> '.join(item['nature_question_sub_sentences']).replace('\\', '').strip()
                    #     questions.append(sub_sentences)
                elif mode == 'test':
                    if 'nature_question_sub_sentences' in item:
                        sub_sentences = item['nature_question_sub_sentences']
                        if type(sub_sentences) == list:
                            sub_sentences = ' <FACT-SEP> '.join(sub_sentences).replace('\\', '').strip()
                        else:
                            sub_sentences = sub_sentences.strip()
                        questions.append(sub_sentences)
                        # questions[-1] = ' <FACT-SEP> '.join([item['nature_question'], questions[-1]])
                else:
                    raise

                for question in questions:
                    if False and mode == 'train':
                        shuffled_question = shuffle_facts(shuffle_entities(question))
                        question_ids = tokenizer(shuffled_question).input_ids
                    else:
                        question_ids = tokenizer(question).input_ids
                    if len(self.examples) < 5:
                        print('=' * 20)
                        print('question tokens:')
                        for _question in question.split(' <FACT-SEP> '):
                            print(_question)
                        if False and mode == 'train':
                            print('shuffled tokens:')
                            for _shuffled_question in shuffled_question.split(' <FACT-SEP> '):
                                print(_shuffled_question)
                        print('program tokens:', program)
                        print('question ids:', question_ids)
                        print('program ids:', program_ids)

                    self.examples.append({
                            'question': question_ids,
                            'program': program_ids,
                        })

            # logger.info("Saving features into cached file %s", cached_features_file)
            # with open(cached_features_file, "wb") as handle:
            #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Loaded data size:', len(self.examples))
        print('Original max question length:', max([len(example['question']) for example in self.examples]))
        print('Original max program length:', max([len(example['program']) for example in self.examples]))
        # self.examples = [example for example in self.examples if len(example['program']) < 300]
        if num_examples > 0:
            self.examples = self.examples[:num_examples]
        print('Used data size:', len(self.examples))
        self.max_program_length = max([len(example['program']) for example in self.examples])
        print('Max question length:', max([len(example['question']) for example in self.examples]))
        print('Max program length:', self.max_program_length)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        item = self.examples[item]

        input_ids = item['question']
        labels = item['program']

        return (input_ids, labels, item)

class QuestionDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, num_examples=-1, mode='train'):
        self.tokenizer = tokenizer
        assert mode in ['train', 'dev', 'test']

        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            # directory, args.model_type + "_cached-Question2Logic_" + filename
            directory, args.model_type + "_cached-NatureQuestion2Logic_" + filename
        )

        if False and os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            data = pickle.load(open(file_path, "rb"))
            self.data = data

            self.examples = []
            num_nature_training_examples = 0
            for item in tqdm(data):
                program = item['program'].strip()
                program_ids = tokenizer(program).input_ids
                questions = []
                if mode == 'train':
                    # for paraphrase_item in item['synthesis_question_paraphrases_sub_sentences'][:5]:
                    #     paraphrase = paraphrase_item['paraphrase'].strip()
                    #     questions.append(paraphrase)
                    # questions.append(item['nature_question'].strip())
                    # for paraphrase in item['synthesis_question_paraphrases'][:5]:
                    #     questions.append(paraphrase.strip())
                    # questions.append(item['synthesis_question'].strip())
                    if 'nature_question_sub_sentences' in item and num_nature_training_examples < 50000:
                        questions.append(item['nature_question'].strip())
                        num_nature_training_examples += 1
                elif mode == 'dev':
                    # if 'synthesis_question_sub_sentences' in item:
                    #     questions.append(item['synthesis_question'].strip())
                        # questions.append(item['nature_question'].strip())
                    questions.append(item['nature_question'].strip())
                elif mode == 'test':
                    if 'nature_question_sub_sentences' in item:
                        questions.append(item['nature_question'].strip())
                    # questions.append(item['synthesis_question'].strip())
                else:
                    raise

                for question in questions:
                    question_ids = tokenizer(question).input_ids
                    if len(self.examples) < 5:
                        print('=' * 20)
                        print('question tokens:', question)
                        print('program tokens:', program)
                        print('question ids:', question_ids)
                        print('program ids:', program_ids)

                    self.examples.append({
                            'question': question_ids,
                            'program': program_ids,
                        })

            # logger.info("Saving features into cached file %s", cached_features_file)
            # with open(cached_features_file, "wb") as handle:
            #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Loaded data size:', len(self.examples))
        print('Original max question length:', max([len(example['question']) for example in self.examples]))
        print('Original max program length:', max([len(example['program']) for example in self.examples]))
        # self.examples = [example for example in self.examples if len(example['program']) < 300]
        if num_examples > 0:
            self.examples = self.examples[:num_examples]
        print('Used data size:', len(self.examples))
        self.max_program_length = max([len(example['program']) for example in self.examples])
        print('Max question length:', max([len(example['question']) for example in self.examples]))
        print('Max program length:', self.max_program_length)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        item = self.examples[item]

        input_ids = item['question']
        labels = item['program']

        return (input_ids, labels, item)

def load_and_cache_examples_ComplexWebQuestions(args, tokenizer, mode='train', num_examples=-1):
    if mode == 'train':
        # file_path = './data/TACL/ComplexWebQuestions/composition/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
        
        ## BREAK
        # file_path = './data/TACL/ComplexWebQuestions/composition/BREAK/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
    
        ## prompt
        # file_path = './data/TACL/ComplexWebQuestions/composition/prompt/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom.100forPrompt'
        
        ## text prompt
        file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/ComplexWebQuestions_train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
    elif mode == 'dev':
        # file_path = './data/TACL/ComplexWebQuestions/composition/ComplexWebQuestions_dev.json.pkl.decom_codex-ComplexWebQuestions-7syn_temp0_topP1'
        
        ## BREAK
        # file_path = './data/TACL/ComplexWebQuestions/composition/BREAK/ComplexWebQuestions_dev.json.pkl.T5-large-decom'
        
        ## text prompt
        file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/ComplexWebQuestions_dev.json.pkl.decom_codex-CWQ-7syn-textPrompt_temp0_topP1'
        
        num_examples = 1000
    elif mode == 'test':
        # file_path = './data/TACL/ComplexWebQuestions/composition/ComplexWebQuestions_test.json.pkl.decom_codex-ComplexWebQuestions-7syn_temp0_topP1'
        
        # file_path = './data/TACL/ComplexWebQuestions/composition/length/ComplexWebQuestions_test.json.pkl.decom_codex-ComplexWebQuestions-7syn_temp0_topP1.1000.length<=4'
        # file_path = './data/TACL/ComplexWebQuestions/composition/length/ComplexWebQuestions_test.json.pkl.decom_codex-ComplexWebQuestions-7syn_temp0_topP1.1000.length>4'
        
        ## BREAK
        # file_path = './data/TACL/ComplexWebQuestions/composition/BREAK/ComplexWebQuestions_test.json.pkl.T5-large-decom'
        
        # file_path = './data/TACL/ComplexWebQuestions/composition/BREAK/length/ComplexWebQuestions_test.json.pkl.T5-large-decom.1000.length<=4'
        # file_path = './data/TACL/ComplexWebQuestions/composition/BREAK/length/ComplexWebQuestions_test.json.pkl.T5-large-decom.1000.length>4'
        
        ## text prompt
        # file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/ComplexWebQuestions_test.json.pkl.decom_codex-CWQ-7syn-textPrompt_temp0_topP1'

        # file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/length/ComplexWebQuestions_test.json.pkl.decom_codex-CWQ-7syn-textPrompt_temp0_topP1.1000'
        file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/length/ComplexWebQuestions_test.json.pkl.decom_codex-CWQ-7syn-textPrompt_temp0_topP1.1000.length<=4'
        # file_path = './data/TACL/ComplexWebQuestions/decomposition-7syn-textPrompt/length/ComplexWebQuestions_test.json.pkl.decom_codex-CWQ-7syn-textPrompt_temp0_topP1.1000.length>4'

        # num_examples = 300
    # dataset = DecomDataset(tokenizer, args, file_path=file_path, num_examples=num_examples, mode=mode)
    dataset = QuestionDataset(tokenizer, args, file_path=file_path, num_examples=num_examples, mode=mode)
    return dataset

def load_and_cache_examples_GrailQA(args, tokenizer, mode='train', num_examples=-1):
    if mode == 'train':
        file_path = './data/TACL/GrailQA/decomposition/grailqa_v1.0_train.json.addCanonical.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
    elif mode == 'dev':
        file_path = './data/TACL/GrailQA/decomposition/grailqa_v1.0_dev.json.addCanonical.dev.pkl.decom_codex-GrailQA-11para_temp0_topP1'
        num_examples = 300
    elif mode == 'test':
        file_path = './data/TACL/GrailQA/decomposition/grailqa_v1.0_dev.json.addCanonical.test.pkl.decom_codex-GrailQA-11para_temp0_topP1'
        num_examples = 300
    # dataset = DecomDataset(tokenizer, args, file_path=file_path, num_examples=num_examples, mode=mode)
    dataset = QuestionDataset(tokenizer, args, file_path=file_path, num_examples=num_examples, mode=mode)
    return dataset

def load_and_cache_examples_KQA(args, tokenizer, mode='train', num_examples=-1):
    if mode == 'train':
        # file_path = './data/TACL/KQA/decomposition/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
    
        ## BREAK
        # file_path = './data/TACL/KQA/decomposition/BREAK/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
    
        ## nature-to-synthesis generalization
        # file_path = './data/TACL/KQA/decomposition/edit-distance/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom.diverse_thres11'
    
        ## codex-paraphrase
        # file_path = './data/TACL/KQA/Codex-paraphrase/train.json.pkl.paraphrase_codex-paraphrase_temp1_topP0.9.complement'
    
        ## prompt
        # file_path = './data/TACL/KQA/decomposition/prompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom.100forPrompt'

        ## few-shot
        # file_path = './data/TACL/KQA/decomposition/few-shot/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-7para_temp0_topP1.10forFewShot'
        # file_path = './data/TACL/KQA/decomposition/few-shot/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-7para_temp0_topP1.100forFewShot'
        # file_path = './data/TACL/KQA/decomposition/few-shot/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-7para_temp0_topP1.1000forFewShot'
        # file_path = './data/TACL/KQA/decomposition/few-shot/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.decom_codex-KQA-7para_temp0_topP1.5000forFewShot'
    
        ## text prompt
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        
        ## text prompt (random prompt)
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
    
        ## text prompt (10 prompt)
        # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
        
        ## text prompt (5 prompt)
        # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'

        ## text prompt (14 nature prompt)
        file_path = './data/TACL/KQA/decomposition-14nature-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'
    
        ## text prompt (14 syn prompt)
        # file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'

        ## text prompt (OPT prompt)
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'

    elif mode == 'dev':
        # file_path = './data/TACL/KQA/decomposition/val.json.dev.pkl.decom_codex-KQA-7para_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition/train.json.pkl.generate10Paraphrases.temperature1.20.T5large_ParaBank2_500w_ckp20w.T5-large-decom'

        ## BREAK
        # file_path = './data/TACL/KQA/decomposition/BREAK/val.json.dev.pkl.T5-large-decom'
        
        ## nature-to-synthesis generalization
        # file_path = './data/TACL/KQA/decomposition/edit-distance/val.json.dev.pkl.decom_codex-KQA-7para_temp0_topP1.diverse_thres11'

        ## text prompt
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/val.json.dev.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        
        ## text prompt (random prompt)
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/val.json.dev.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/val.json.dev.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        
        ## text prompt (10 prompt)
        # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/val.json.dev.pkl.decom_codex-KQA-10T5Para-textPrompt_temp0_topP1'

        ## text prompt (5 prompt)
        # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/val.json.dev.pkl.decom_codex-KQA-5T5Para-textPrompt_temp0_topP1'

        ## text prompt (14 nature prompt)
        file_path = './data/TACL/KQA/decomposition-14nature-textPrompt/val.json.dev.pkl.decom_codex-KQA-14nature-textPrompt_temp0_topP1'

        ## text prompt (14 syn prompt)
        # file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/val.json.dev.pkl.decom_codex-KQA-14syn-textPrompt_temp0_topP1'

        ## text prompt (OPT prompt)
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/val.json.dev.pkl.T5-large-decom'

        # num_examples = 300
    elif mode == 'test':
        # file_path = './data/TACL/KQA/decomposition/val.json.test.pkl.decom_codex-KQA-7para_temp0_topP1'
        
        ## BREAK
        # file_path = './data/TACL/KQA/decomposition/BREAK/val.json.test.pkl.T5-large-decom'

        # file_path = './data/TACL/KQA/decomposition/BREAK/length/val.json.test.pkl.T5-large-decom.1000.length0-3'
        # file_path = './data/TACL/KQA/decomposition/BREAK/length/val.json.test.pkl.T5-large-decom.1000.length4-4'
        # file_path = './data/TACL/KQA/decomposition/BREAK/length/val.json.test.pkl.T5-large-decom.1000.length5-5'
        # file_path = './data/TACL/KQA/decomposition/BREAK/length/val.json.test.pkl.T5-large-decom.1000.length6-100'
        
        ## nature-to-synthesis generalization
        # file_path = './data/TACL/KQA/decomposition/edit-distance/val.json.test.pkl.decom_codex-KQA-7para_temp0_topP1'

        ## length
        # file_path = './data/TACL/KQA/decomposition/length/val.json.test.pkl.decom_codex-KQA-7para_temp0_topP1.1000.length0-3'
        # file_path = './data/TACL/KQA/decomposition/length/val.json.test.pkl.decom_codex-KQA-7para_temp0_topP1.1000.length4-4'
        # file_path = './data/TACL/KQA/decomposition/length/val.json.test.pkl.decom_codex-KQA-7para_temp0_topP1.1000.length5-5'
        # file_path = './data/TACL/KQA/decomposition/length/val.json.test.pkl.decom_codex-KQA-7para_temp0_topP1.1000.length6-100'

        ## novel structures
        # file_path = './data/TACL/KQA/decomposition/quality_annotation/val.json.test.pkl.decom_codex-KQA-7para_temp0_topP1.1000.quality_annotation.question_similar_structure'
        # file_path = './data/TACL/KQA/decomposition/quality_annotation/val.json.test.pkl.decom_codex-KQA-7para_temp0_topP1.1000.quality_annotation.question_diverse_structure'

        ## text prompt
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/length/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1.1000'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/length/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1.1000.length0-3'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/length/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1.1000.length4-4'
        file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/length/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1.1000.length5-5'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt/length/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1.1000.length6-100'

        ## text prompt (random prompt)
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1'

        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt1/length/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1.1000'
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-ranPrompt2/length/val.json.test.pkl.decom_codex-KQA-14T5Para-textPrompt_temp0_topP1.1000'

        ## text prompt (10 prompt)
        # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/val.json.test.pkl.decom_codex-KQA-10T5Para-textPrompt_temp0_topP1'
        
        # file_path = './data/TACL/KQA/decomposition-10T5Para-textPrompt/length/val.json.test.pkl.decom_codex-KQA-10T5Para-textPrompt_temp0_topP1.1000'
        
        ## text prompt (5 prompt)
        # file_path = './data/TACL/KQA/decomposition-5T5Para-textPrompt/val.json.test.pkl.decom_codex-KQA-5T5Para-textPrompt_temp0_topP1'

        ## text prompt (14 nature prompt)
        # file_path = './data/TACL/KQA/decomposition-14nature-textPrompt/val.json.test.pkl.decom_codex-KQA-14nature-textPrompt_temp0_topP1'

        ## text prompt (14 syn prompt)
        # file_path = './data/TACL/KQA/decomposition-14syn-textPrompt/val.json.test.pkl.decom_codex-KQA-14syn-textPrompt_temp0_topP1'

        ## text prompt (OPT prompt)
        # file_path = './data/TACL/KQA/decomposition-14T5Para-textPrompt-OPT/val.json.test.pkl.T5-large-decom'

        # num_examples = 300
    # dataset = DecomDataset(tokenizer, args, file_path=file_path, num_examples=num_examples, mode=mode)
    dataset = QuestionDataset(tokenizer, args, file_path=file_path, num_examples=num_examples, mode=mode)
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
    best_EM, early_stop_indicator = -1, 0

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
                        # results = evaluate(args, model, tokenizer, dataset=args.eval_dataset, mode='test')
                        # for key, value in results.items():
                        #     tb_writer.add_scalar("eval_%s.test_%s" % (args.eval_dataset, key), value, global_step)
                        if global_step >= 30000:
                            results = evaluate(args, model, tokenizer, dataset=args.eval_dataset, mode='dev')
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_%s.dev_%s" % (args.eval_dataset, key), value, global_step)
                            EM = results['EM']

                            if global_step % 5000 == 0:
                                best_EM = -1
                            if EM > best_EM:
                                best_EM = EM
                                early_stop_indicator = 0

                                output_dir = os.path.join(args.output_dir, "best_EM_%d" % (global_step))
                                os.makedirs(output_dir, exist_ok=True)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)

                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                logger.info("Update best model (EM: %f) checkpoint to %s" % (best_EM, output_dir))
                            else:
                                early_stop_indicator += 1
                                # if early_stop_indicator >= 10:
                                #     return global_step, tr_loss / global_step
                        
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

    ## save the training end
    output_dir = os.path.join(args.output_dir, "TrainingEnd_%d" % (global_step))
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Save the training-end checkpoint to %s" % (output_dir))
    ##

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="", dataset='unified', mode='None', do_generate=True) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = eval('load_and_cache_examples_%s' % (dataset))(args, tokenizer, mode=mode)

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
    eval_precise = []
    eval_recall = []
    eval_em = []
    eval_bleu1, eval_bleu2, eval_bleu3, eval_bleu4 = [], [], [], []
    model.eval()

    def decode(ids, tokenizer):
        sentence = tokenizer.decode(ids, skip_special_tokens=False)
        sentence = sentence.replace('</s>', ' ').replace('<pad>', ' ')
        sentence = ' '.join([word for word in sentence.split(' ') if word != ''])
        return sentence

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
    for example_id, (input_ids, labels, items) in tqdm(enumerate(eval_dataloader)):
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
                                max_length=eval_dataset.max_program_length + 64,
                                num_beams=5,
                                eos_token_id=1,
                                num_return_sequences=num_return_sequences,
                            )
                else:
                    num_return_sequences = 20
                    outputs = model.generate(
                                input_ids=input_ids,
                                max_length=eval_dataset.max_program_length + 64,
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

                    # pickle.dump(record, open(os.path.join(args.output_dir, '%s.inference.pkl' % (mode)), 'wb'))
                    # print('save inference results to', os.path.join(args.output_dir, '%s.inference.pkl' % (mode)))

    # with open(os.path.join(args.output_dir, '%s.inference.txt' % (mode)), 'w') as w:
    #     for _record, item in zip(record, eval_dataset.data):
    #         _record['item'] = item
    #         w.write('=' * 20 + '\n')
    #         w.write('synthetic :%s\n' % (item['synthesis_question']))
    #         w.write('natural   :%s\n' % (item['nature_question']))
    #         w.write('question  :%s\n' % (_record['question']))
    #         w.write('prediction:%s\n' % (_record['prediction']))
    #         w.write('gold      :%s\n' % (_record['answer']))
    #         w.write('%s\n' % (str(_record['prediction'] == _record['answer'])))
    # pickle.dump(record, open(os.path.join(args.output_dir, '%s.inference.pkl' % (mode)), 'wb'))
    # print('save inference results to', os.path.join(args.output_dir, '%s.inference.pkl' % (mode)))

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

def evaluate_semantic_match(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="", dataset='unified', mode='None', do_generate=True) -> Dict:
    if dataset == 'ComplexWebQuestions':
        logic2question_model = T5ForConditionalGeneration.from_pretrained('/data/niuyilin/Foundation-Skill-Semantic-Parser/runs/DecomParaCom/T5-large-Logic2Question-ComplexWebQuestions_18609Syn_evalONdev500Syn-lr1e-4-bs16-epoch10/best_bleu4_2000')
    elif dataset == 'KQA':
        logic2question_model = T5ForConditionalGeneration.from_pretrained('/data/niuyilin/Foundation-Skill-Semantic-Parser/runs/DecomParaCom/T5-large-Logic2Question-KQA_94376Syn_evalONdev500Syn-lr1e-4-bs16-epoch10/best_bleu4_5000')
    logic2question_model.to(args.device)
    logic2question_model.eval()

    simcse_file = "/data/niuyilin/pre-trained-models/sup-simcse-roberta-large"
    simcse_tokenizer = AutoTokenizer.from_pretrained(simcse_file)
    simcse_model = AutoModel.from_pretrained(simcse_file)
    simcse_model.to(args.device)
    simcse_model.eval()
    def encode(texts):
        inputs = simcse_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        for key, value in inputs.items():
            inputs[key] = value.to(args.device)

        # Get the embeddings
        with torch.no_grad():
            embeddings = simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        return embeddings

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = eval('load_and_cache_examples_%s' % (dataset))(args, tokenizer, mode=mode)

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
    eval_precise = []
    eval_recall = []
    eval_em = []
    eval_bleu1, eval_bleu2, eval_bleu3, eval_bleu4 = [], [], [], []
    model.eval()

    def decode(ids, tokenizer):
        sentence = tokenizer.decode(ids, skip_special_tokens=False)
        sentence = sentence.replace('</s>', ' ').replace('<pad>', ' ')
        sentence = ' '.join([word for word in sentence.split(' ') if word != ''])
        return sentence

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
    for example_id, (input_ids, labels, items) in tqdm(enumerate(eval_dataloader)):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            model_outputs = model(input_ids=input_ids, labels=labels)
            loss = model_outputs.loss
            eval_loss.append(loss.item())

            if do_generate:
                if True:
                    num_return_sequences = 5
                    outputs = model.generate(
                                input_ids=input_ids,
                                max_length=eval_dataset.max_program_length + 64,
                                num_beams=5,
                                eos_token_id=1,
                                num_return_sequences=num_return_sequences,
                            )
                else:
                    num_return_sequences = 20
                    outputs = model.generate(
                                input_ids=input_ids,
                                max_length=eval_dataset.max_program_length + 64,
                                do_sample=True,
                                top_p=0.9,
                                eos_token_id=1,
                                num_return_sequences=num_return_sequences
                            )
                outputs = outputs.view(input_ids.size(0), num_return_sequences, outputs.size(1))
                for _input_ids, _outputs, _labels in zip(input_ids, outputs, labels):
                    print('=' * 20)
                    predictions = [decode(output, tokenizer) for output in _outputs]
                    predictions_ids = [torch.tensor(tokenizer(prediction).input_ids).long() for prediction in predictions]
                    predictions_ids = pad_sequence(predictions_ids, batch_first=True, padding_value=0).to(args.device)
                    predictions_question_ids = logic2question_model.generate(
                                input_ids=predictions_ids,
                                max_length=200,
                                num_beams=5,
                                eos_token_id=1,
                                num_return_sequences=1,
                            )
                    predictions_question = [decode(prediction_question_ids, tokenizer) for prediction_question_ids in predictions_question_ids]

                    origin_question = decode(_input_ids, tokenizer)

                    embeddings = normalize(encode([origin_question] + predictions_question))
                    prediction_similarity_list = [(prediction, prediction_question, (embedding * embeddings[0]).sum().item()) for prediction, prediction_question, embedding in zip(predictions, predictions_question, embeddings[1:])]
                    reranked_predictions = sorted(prediction_similarity_list, key=lambda x: x[2], reverse=True)
                    for prediction, prediction_question, similarity in reranked_predictions:
                        print('%.3f\t%s' % (similarity, prediction_question))
                    reranked_prediction = reranked_predictions[0][0]

                    _labels = _labels.tolist()
                    if -100 in _labels:
                        _labels = _labels[:_labels.index(-100)]
                    ground_truth = decode(_labels, tokenizer)

                    metric = (compute_f1(reranked_prediction, ground_truth), compute_exact_match(reranked_prediction, ground_truth), calculate_bleu(reranked_prediction, [ground_truth]))

                    print('-' * 20)
                    try:
                        print('Question    :', origin_question)
                        print('Prediction  :', reranked_prediction)
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
                                    'question': origin_question,
                                    'prediction': reranked_prediction,
                                    'answer': ground_truth,
                                })

                    # pickle.dump(record, open(os.path.join(args.output_dir, '%s.inference.pkl' % (mode)), 'wb'))
                    # print('save inference results to', os.path.join(args.output_dir, '%s.inference.pkl' % (mode)))

    # with open(os.path.join(args.output_dir, '%s.inference.txt' % (mode)), 'w') as w:
    #     for _record, item in zip(record, eval_dataset.data):
    #         _record['item'] = item
    #         w.write('=' * 20 + '\n')
    #         w.write('synthetic :%s\n' % (item['synthesis_question']))
    #         w.write('natural   :%s\n' % (item['nature_question']))
    #         w.write('question  :%s\n' % (_record['question']))
    #         w.write('prediction:%s\n' % (_record['prediction']))
    #         w.write('gold      :%s\n' % (_record['answer']))
    #         w.write('%s\n' % (str(_record['prediction'] == _record['answer'])))
    # pickle.dump(record, open(os.path.join(args.output_dir, '%s.inference.pkl' % (mode)), 'wb'))
    # print('save inference results to', os.path.join(args.output_dir, '%s.inference.pkl' % (mode)))

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
    parser.add_argument("--do_prompt_tuning", action="store_true", help="Whether to use prompt-tuning.")
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

    ParserConfig, ParserModelWithLMHead, ParserTokenizer = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = ParserConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = ParserConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        tokenizer = ParserTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = ParserTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
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
        model = ParserModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        # model = ParserModelWithLMHead.from_config(config)
        model = ParserModelWithLMHead(config=config)

    if args.do_prompt_tuning:
        logger.info("Initialize the model for prompt-tuning.")
        LoraModel(model)
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
        train_dataset = eval('load_and_cache_examples_%s' % (args.eval_dataset))(args, tokenizer, mode='train', num_examples=args.num_examples)
        # train_dataset = load_and_cache_examples_Overnight_socialnetwork(args, tokenizer, mode='train', num_examples=args.num_examples)

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
        model = ParserModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = ParserTokenizer.from_pretrained(args.output_dir)
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

            model = ParserModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            # result = evaluate(args, model, tokenizer, dataset=args.eval_dataset, mode='dev')
            # result = evaluate(args, model, tokenizer, dataset=args.eval_dataset, mode='test')
            result = evaluate_semantic_match(args, model, tokenizer, dataset=args.eval_dataset, mode='test')
            # result = evaluate(args, model, tokenizer, dataset='KQA', mode='dev')
            # result = evaluate(args, model, tokenizer, dataset='KQA', mode='test')
            # result = console(args, model, tokenizer, dataset='KQA', mode='dev')
            # result = evaluate_sts(args, model, tokenizer, mode='dev')
            # result = evaluate_sts1(args, model, tokenizer, mode='dev')
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


# import time
# print('begin sleep 8000 seconds...')
# time.sleep(8000)

if __name__ == "__main__":
    main()
