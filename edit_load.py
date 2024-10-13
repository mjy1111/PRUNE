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

from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
import torch
import numpy as np
import regex
import string
import jsonlines

from test_tasks import OpenDomainQA, NLI, Dialogue, Sentiment, Reasoning, Summarization

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

'''
from dsets import (
    CounterFactDataset,
    MultiCounterFactDataset,
)
'''

'''
from easyeditor.dataset import (
    CounterFactDataset,
    MultiCounterFactDataset,
)

from eval_bi.eval_utils_counterfact import compute_rewrite_quality_counterfact
'''
DS_DICT = {
    "mcf": (MultiCounterFactDataset),
    "cf": (BiCounterFactDataset),
    "bi_cf_qa": (BiCounterFactDataset),
    "bi_cf_judge": (BiCounterFactDataset),
}



def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]
#print(type(edited_model))

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    ds_name: str,
    dataset_size_limit: int,
    dir_name: str,
    num_edits: int,
    cuda: int,
    aerfa: float,
    beta: float,
    svd: int,
    seq_num: int,
    Reduce: str):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    

    RESULTS_DIR="results/{}".format(ds_name)
    if ds_name=="cf":
        DATA_DIR="data/counterfact.json"
    else:
        DATA_DIR="data/BAKE_{}.json".format(ds_name.split("_")[-1])
    continue_from_run=None


    #*****************dir name***************
    if Reduce==None:
        dir_name=dir_name+"_{}".format(str(seq_num))
    else:
        dir_name="reduce_{}/".format(Reduce)+dir_name+"_{}".format(str(seq_num))


    if continue_from_run is None:
        alg_dir = Path("{}/{}/{}/".format(RESULTS_DIR, dir_name, model_name))
        print(alg_dir)
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = Path("{}/{}/{}/".format(RESULTS_DIR,dir_name,model_name) + f"run_{str(run_id).zfill(3)}")
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    


    ds_class = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit)

    # Iterate through dataset
    
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]

        #etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

    start = time()
    

    prompts=[record['requested_rewrite']["prompt"].format(record['requested_rewrite']["subject"]) for record in ds]
    ground_truth = [record['requested_rewrite']['target_true']["str"] for record in ds]
    target_new = [record['requested_rewrite']['target_new']["str"] for record in ds]
    subject = [record['requested_rewrite']["subject"] for record in ds]
    relation = [record['requested_rewrite']["relation_id"] for record in ds]
    
    para=[record['paraphrase_prompts'] for record in ds]
    neighbor=[record['neighborhood_prompts'] for record in ds]
    if 'reverse_qa' in ds[0]:
        reverse_qa=[record['reverse_qa'] for record in ds]
    else:
        reverse_qa=None
    if "reverse_judge" in ds[0]:
        reverse_judge=[record['reverse_judge'] for record in ds]
    else:
        reverse_judge=None
    
    
    '''
    edited_model, weights_copy = apply_algo(
        model,
        tok,
        [
            {"case_id": record["case_id"], **record["requested_rewrite"]}
            for record in record_chunks
        ],
        hparams,
        copy=False,
        return_orig_weights=True,
        **args_conserve_memory,
        **etc_args,
    )
    '''
    if alg_name=="MEMIT":
        hparams=MEMITHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
    elif alg_name=="ROME":
        hparams=ROMEHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
        hparams.aerfa=0
        hparams.beta=0
    elif alg_name=="BIRD":
        hparams=ROMEHyperParams.from_hparams('hparams/{}/{}.yaml'.format("ROME", model_name))
        hparams.aerfa=args.aerfa
        hparams.beta=args.beta
    elif alg_name=="MEND":
        hparams=MENDHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
    elif alg_name=="KN":
        hparams=KNHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
    elif alg_name=="FT":
        hparams=FTHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))

    editor=BiEditor.from_hparams(hparams)
    tok=editor.tok
    #print(1)
    all_metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        rephrase_prompts=para,
        locality_inputs=neighbor,
        keep_original_weight=False,
        reverse_qa_prompts=reverse_qa,
        reverse_judge_prompts=reverse_judge,
        relation_id=relation,
        case_result_template=case_result_template,
        num_edits1=num_edits,
        svd=svd,
        seq_num=seq_num,
        Reduce=args.reduce_type,
    )
    exec_time = time() - start
    print("Execution took", exec_time)
    

    if 'gpt' in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(f'./hugging_cache/{model_name}')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side='left'
    elif 'llama' or "vicuna" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(f'./hugging_cache/{model_name}')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        #tokenizer.padding_side='left'

    print(tokenizer.pad_token_id)

    #tokenizer.padding_side='left'
    mode="Sequential"
    method=alg_name
    sample_total=seq_num
    
    preserve="original"
    if Reduce:
        preserve="reduce_"+Reduce
    

    for i in range(2):
        Summarization(mode,method, sample_total, model_name, tokenizer, edited_model, preserve)
        OpenDomainQA(mode,method, sample_total, model_name, tokenizer, edited_model, preserve)
        Reasoning(mode,method, sample_total, model_name, tokenizer, edited_model, preserve)
        #Dialogue(mode,method, sample_total, model_name, tokenizer, edited_model, preserve)
        #OpenDomainQA(mode,method, sample_total, model_name, tokenizer, edited_model, preserve)
        NLI(mode,method, sample_total, model_name, tokenizer, edited_model, preserve)










if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE", "MEMIT","SERAC", "BIRD"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["llama-7b", "gpt2-xl", "gpt-j-6B","llama-13b","vicuna-13b","vicuna-7b","llama2-7b","llama2-13b","llama3-8b"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre","bi_cf_qa","bi_cf_judge"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--dir_name",
        default="cf",
        help="the directory to save results",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="cuda name",
    )
    parser.add_argument(
        "--aerfa",
        type=float,
        default=0,
        help="cuda name",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0,
        help="cuda name",
    )
    parser.add_argument(
        "--svd",
        type=int,
        default=100,
        help="cuda name",
    )
    parser.add_argument(
        "--seq_num",
        type=int,
        default=100,
        help="cuda name",
    )
    parser.add_argument(
        "--reduce_type",
        choices=["linear", "log", "log2","log1_5","log1_2"],
        default=None,
        help="REDUCE MAX",
    )

    args = parser.parse_args()
    if args.dir_name=="cf":
        args.dir_name=args.alg_name

    main(args.alg_name, args.model_name, args.ds_name, args.dataset_size_limit, args.dir_name, args.num_edits, args.cuda, args.aerfa,args.beta, args.svd, args.seq_num, args.reduce_type)