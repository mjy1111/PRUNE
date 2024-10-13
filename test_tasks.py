import os
import sys
import csv
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from easyeditor.editors.bi_editor_loadw import BiEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams

import os,json
from pathlib import Path
from dsets import CounterFactDataset, MultiCounterFactDataset, BiCounterFactDataset
from typing import Tuple, Union
from time import time
from transformers import GPT2Tokenizer

from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import numpy as np
import regex
import string
import jsonlines

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def OpenDomainQA(mode,method, sample_total, model_name, tokenizer, edited_model, preserve):
    flag = False
    with open("./data/task-data/test-OpenDomainQA.jsonl", "r+", encoding="utf8") as f:
        exact_match_count = 0
        answer_lengths = []

        if not os.path.exists(f"./test-result/{model_name}/test-OpenDomainQA/{preserve}"): 
            os.makedirs(f"./test-result/{model_name}/test-OpenDomainQA/{preserve}") 

        num=0
        for data in jsonlines.Reader(f):
            num+=1
            if num==262:
                break
            if mode =="Batch-Sequential":
                edit_time = int(sample_total) // int(batch_size)
                result = open(f"./test-result/{model_name}/test-OpenDomainQA/{preserve}/result-OpenDomainQA-{mode}-{model_name}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
            else:
                result = open(f"./test-result/{model_name}/test-OpenDomainQA/{preserve}/result-OpenDomainQA-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
            question = data['question']
            document = data['output']
            answer = data['answer']
            generation_prompts = [f"Refer to the passage below and answer the following question. Passage: {document[0]} Question: {question}. Answer:"]
            #print(generation_prompts)
            
            if model_name=="gpt-j-6B":
                batch = tokenizer(generation_prompts, return_tensors='pt')
            else:
                batch = tokenizer(generation_prompts, return_tensors='pt',padding="max_length")
    
            post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=20)
    
            Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
            
            #print(Outputs)
            answer = data['answer']
            predict = Outputs[-1].split("Answer:")[-1].replace("\n", " ")
            predict = normalize_answer(predict)
            for per_answer in answer:
                result.write(str(normalize_answer(per_answer)) + " ")
            result.write("\t")
            result.write(f'The model predict is: {str(predict)}' + "\n")
            words = predict.split(" ")
            if ems(words[0], answer) or ems(words[-1], answer): 
                exact_match_count += 1
                continue
            for i in range(len(words)-1):
                output = words[i]
                if ems(output, answer): 
                    exact_match_count += 1
                    continue
                for j in range(i+1, len(words)):
                    output = output + " " + words[j]
                    if ems(output, answer): 
                        exact_match_count += 1
                        flag = True
                        break
                if flag:
                    break
            answer_lengths.append(len(predict.split()))
            result.close()
        
        em = round(exact_match_count/261, 4)
        lens = round(np.mean(answer_lengths), 4)
        
        if mode =="Batch-Sequential":
            edit_time = int(sample_total) // int(batch_size)
            result = open(f"./test-result/{model_name}/test-OpenDomainQA/{preserve}/result-OpenDomainQA-{mode}-{model_name}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
        else:
            result = open(f"./test-result/{model_name}/test-OpenDomainQA/{preserve}/result-OpenDomainQA-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
        result.write(str(em) + "\t")
        result.write(str(lens) + "\n\n\n")




def NLI(mode,method, sample_total, model_name, tokenizer, edited_model, preserve):

    if not os.path.exists(f"./test-result/{model_name}/test-NLI/{preserve}"): 
        os.makedirs(f"./test-result/{model_name}/test-NLI/{preserve}") 


    with open('./data/task-data/test-NLI.tsv') as f:
        index = []
        sentence_1 = []
        sentence_2 = []
        label = []
        generation_prompts_list = []
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        others = 0
        tsvreader = csv.reader(f, delimiter='\t')
        num=0
        for line in tsvreader:
            index.append(line[0])
            sentence_1.append(line[1])
            sentence_2.append(line[2])
            label.append(line[3])
            num+=1
            if num==300:
                break
        for i in range(1,len(index)):
            generation_prompts = [f"{sentence_1[i]} entails the {sentence_2[i]}. True or False? answer:"]
            generation_prompts_list.append(generation_prompts)
        for j in range(len(generation_prompts_list)):
            if mode =="Batch-Sequential":
                edit_time = int(sample_total) // int(batch_size)
                result = open(f"./test-result/{model_name}/test-NLI/{preserve}/result-NLI-{mode}-{model_name}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
            else:
                result = open(f"./test-result/{model_name}/test-NLI/{preserve}/result-NLI-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
            batch = tokenizer(generation_prompts_list[j], return_tensors='pt', padding="max_length")
            post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=1)
    
            Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
            predict = Outputs[-1].split("answer:")[-1]
            result.write(str(label[j+1]) + '\t')
            result.write(str(predict) + '\n')
            if ('true' in predict.lower()) or ('false' in predict.lower()):
                if 'not_entailment' in label[j+1].lower():
                    if 'true' in predict.lower():
                        FP = FP + 1
                    if 'false' in predict.lower():
                        FN = FN + 1
                else:
                    if 'true' in predict.lower():
                        TP = TP + 1
                    if 'false' in predict.lower():
                        TN = TN + 1
            else:
                others = others + 1
            result.close()
    
    if mode =="Batch-Sequential":
        edit_time = int(sample_total) // int(batch_size)
        result = open(f"./test-result/{model_name}/test-NLI/{preserve}/result-NLI-{mode}-{model_name}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
    else:
        result = open(f"./test-result/{model_name}/test-NLI/{preserve}/result-NLI-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
    if others == 299:
        result.write("error" + '\n\n\n')
    else:
        accuracy = (TP + FN)/(TP + FN + TN + FP)
        total_accuracy = (TP + FN)/(TP + FN + TN + FP + others)
        result.write(str(TP) + '\t')
        result.write(str(FN) + '\t')
        result.write(str(TN) + '\t')
        result.write(str(FP) + '\t')
        result.write(str(others) + '\n')
        result.write(str(accuracy) + '\t')
        result.write(str(total_accuracy) + '\n\n\n')
    result.close()
    
    


def Dialogue(mode,method, sample_total, model_name, tokenizer, edited_model, preserve):


    if not os.path.exists(f"./test-result/{model_name}/test-dialogue/{preserve}"): 
        os.makedirs(f"./test-result/{model_name}/test-dialogue/{preserve}") 

    correct = 0
    other = 0
    for i in range(1,357):
        with open(f"./data/task-data/test-dialogue/dev_{i}.txt") as f:
            line = eval(f.read())
            answers = line["answers"]
            options = line["options"]
            article = line["article"]
            generation_prompts = [f"Q: {article} Which choice is correct? Answer Chioces: (A){options[0]}(B){options[1]}(C){options[2]}(D){options[3]} A: Among A through D, the answer is"]
            
            result = open(f"./test-result/{model_name}/test-dialogue/{preserve}/result-dialogue-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
            model = edited_model
            batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
            pre_edit_outputs = model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=1)
                
            Outputs = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
            predict = Outputs[-1].split("the answer is")[-1]
            result.write(str(answers) + '\t')
            result.write(str(predict) + '\n')
            if ('A' in predict) or ('B' in predict) or ('C' in predict) or ('D' in predict):
                if (answers in predict):
                    correct = correct + 1
            else:
                other = other + 1
            result.close()
    
        f.close()
    
    result = open(f"./test-result/{model_name}/test-dialogue/{preserve}/result-dialogue-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
    if other == 356:
        result.write("error" + '\n\n\n')
    else:
        accuracy = correct / 356
        result.write(str(correct) + '\t')
        result.write(str(other) + '\t')
        result.write(str(accuracy) + '\n\n\n')
    result.close()


def Sentiment(mode,method, sample_total, model_name, tokenizer, edited_model, preserve):
  
    if not os.path.exists(f"./test-result/{model_name}/test-SentimentAnalysis/{preserve}"): 
        os.makedirs(f"./test-result/{model_name}/test-SentimentAnalysis/{preserve}") 
    
    with open('./data/task-data/test-SentimentAnalysis.tsv') as f:
        text = []
        label = []
        generation_prompts_list = []
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        others = 0
        tsvreader = csv.reader(f, delimiter='\t')
        
        num=0
        for line in tsvreader:
            text.append(line[0])
            label.append(line[1])
            num+=1
            if num==300:
                break
            
        for i in range(1,len(text)):
            generation_prompts = [f"For each snippet of text,label the sentiment of the text as positive or negative.The answer should be exact 'positive' or 'negative'. text: {text[i]} answer:"]
            generation_prompts_list.append(generation_prompts)
        for j in range(len(generation_prompts_list)):
            if mode =="Batch-Sequential":
                edit_time = int(sample_total) // int(batch_size)
                result = open(f"./test-result/{model_name}/test-SentimentAnalysis/{preserve}/result-SentimentAnalysis-{mode}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
            else:
                 result = open(f"./test-result/{model_name}/test-SentimentAnalysis/{preserve}/result-SentimentAnalysis-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
            batch = tokenizer(generation_prompts_list[j], return_tensors='pt', padding="max_length")
    
            post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=1)
    
            Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
            predict = Outputs[-1].split("answer:")[-1]
            result.write(str(label[j+1]) + '\t')
            result.write(str(predict) + '\n')
            if ('positive' in predict.lower()) or ('negative' in predict.lower()):
                if ('positive' in predict.lower()) and (int(label[j+1]) == 1):
                    TP = TP + 1
                elif ('negative' in predict.lower()) and (int(label[j+1]) == 0):
                    FN = FN + 1
                elif ('negative' in predict.lower()) and (int(label[j+1]) == 1):
                    TN = TN + 1
                elif ('positive' in predict.lower()) and (int(label[j+1]) == 0):
                    FP = FP + 1
            else:
                others = others + 1
            result.close()
    print(TP,FN,TN,FP, others)
    
    if mode =="Batch-Sequential":
        edit_time = int(sample_total) // int(batch_size)
        result = open(f"./test-result/{model_name}/test-SentimentAnalysis/{preserve}/result-SentimentAnalysis-{mode}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
    else:
        result = open(f"./test-result/{model_name}/test-SentimentAnalysis/{preserve}/result-SentimentAnalysis-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
    if others == 299:
        result.write("error" +" 0" '\n\n')
    else:
        accuracy = (TP + FN)/(TP + FN + TN + FP)
        total_accuracy = (TP + FN)/(TP + FN + TN + FP + others)
        result.write(str(TP) + '\t')
        result.write(str(FN) + '\t')
        result.write(str(TN) + '\t')
        result.write(str(FP) + '\t')
        result.write(str(others) + '\n')
        result.write(str(accuracy) + '\t')
        result.write(str(total_accuracy) + '\n\n')
    result.close()
    
    
    
    
def Reasoning(mode,method, sample_total, model_name, tokenizer, edited_model, preserve):
  
    if not os.path.exists(f"./test-result/{model_name}/test-reasoning/{preserve}"): 
        os.makedirs(f"./test-result/{model_name}/test-reasoning/{preserve}") 
    
    correct = 0
    
    with open("./data/task-data/test-reasoning.jsonl", "r+", encoding="utf8") as f:
        num=0
        for data in jsonlines.Reader(f):
            num+=1
            if num==359:
                break
            if mode =="Batch-Sequential":
                edit_time = int(sample_total) // int(batch_size)
                result = open(f"./test-result/{model_name}/test-reasoning/{preserve}/result-reasoning-{mode}-{model_name}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
            else:
                result = open(f"./test-result/{model_name}/test-reasoning/{preserve}/result-reasoning-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
            question = data['question']
            answers = data['answer']
            hint = answers.split("#### ")[0]
            generation_prompts = [f"Q: {question} A:Let's think step by step. {hint} Therefore, the answer (arabic numerals) is:"]
            answers = answers.split("#### ")[1]
            batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
    
            post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=5)
    
            Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
            predict = Outputs[-1].split("Therefore, the answer (arabic numerals) is:")[1]
            result.write(f"The answers is:{str(answers)}" + "\n")
            result.write(f'The model predict is:{str(predict)}'+ "\n")
            if answers in predict:
                correct = correct + 1
            result.close()
    
        acc = correct / 358
        if mode =="Batch-Sequential":
            edit_time = int(sample_total) // int(batch_size)
            result = open(f"./test-result/{model_name}/test-reasoning/{preserve}/result-reasoning-{mode}-{model_name}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
        else:
            result = open(f"./test-result/{model_name}/test-reasoning/{preserve}/result-reasoning-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
        result.write(str(acc) + "\n\n\n")
        result.close()
  


def Summarization(mode,method, sample_total, model_name, tokenizer, edited_model, preserve):

  
    if not os.path.exists(f"./test-result/{model_name}/test-summarization/{preserve}"):
        os.makedirs(f"./test-result/{model_name}/test-summarization/{preserve}")

    f = open('./data/task-data/test-summarization.json', 'r')
    content = f.read()
    corpus = json.loads(content)
    
    summary = []
    dialogue = []
    for i in range(149):
        summary.append(corpus[i]['summary'])
    for i in range(149):
        dialogue.append(corpus[i]['dialogue'])
    

    bleu_score_total = 0
    rouge_score_total = 0
    for i in range(len(dialogue)):
        if mode =="Batch-Sequential":
            edit_time = int(sample_total) // int(batch_size)
            result = open(f"./test-result/{model_name}/test-summarization/{preserve}/result-summarization-{mode}-{model_name}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
        else:
            result = open(f"./test-result/{model_name}/test-summarization/{preserve}/result-summarization-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
        generation_prompts = [f"{dialogue[i]}\nTL;DR:"]
        batch = tokenizer(generation_prompts, return_tensors='pt', padding="max_length")
    
        post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=25)
            
        Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
        predict = Outputs[-1].split("DR:")[-1]
        predict = predict[0:-13]
        result.write(str(summary[i]) + "\t")
        result.write(str(predict) + "\t")
    
        if len(predict) <= 1:
            bleu_score = 0
            result.write(str(bleu_score) + "\t")
            bleu_score_total = bleu_score_total + bleu_score
            rouge_score = 0
            result.write(str(rouge_score) + "\n")
            rouge_score_total = rouge_score_total + rouge_score
            continue
        else:
            reference = []
            reference.append(summary[i].split())
            candidate = predict.split()
            bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
            result.write(str(bleu_score) + "\t")
            bleu_score_total = bleu_score_total + bleu_score
            rouge = Rouge()
            score = rouge.get_scores(predict, summary[i])
            rouge_score = (score[0]['rouge-1']['f'] + score[0]['rouge-2']['f'] + score[0]['rouge-l']['f']) / 3
            result.write(str(rouge_score) + "\n")
            rouge_score_total = rouge_score_total + rouge_score
        result.close()
    
    if mode =="Batch-Sequential":
        edit_time = int(sample_total) // int(batch_size)
        result = open(f"./test-result/{model_name}/test-summarization/{preserve}/result-summarization-{mode}-{model_name}-{method}-{batch_size}*{str(edit_time)[-1]}.txt", "a", encoding="utf8")
    else:
        result = open(f"./test-result/{model_name}/test-summarization/{preserve}/result-summarization-{mode}-{model_name}-{method}{sample_total}.txt", "a", encoding="utf8")
    result.write(str(bleu_score_total / 149) + "\t")
    result.write(str(rouge_score_total / 149) + "\n\n\n\n\n\n")
    result.close()