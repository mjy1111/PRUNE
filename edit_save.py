
import os,json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from easyeditor.editors.bi_editor_save import BiEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams

import os,json
from pathlib import Path
from dsets import CounterFactDataset, MultiCounterFactDataset, BiCounterFactDataset, MENDQADataset
from typing import Tuple, Union
from time import time

import torch
import random
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
    "zsre": (MENDQADataset)
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
    beta: float,):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    

    RESULTS_DIR="results/{}".format(ds_name)
    
    if ds_name=="cf":
        DATA_DIR="data/counterfact.json"
    elif ds_name=="zsre":
        DATA_DIR="data/zsre_mend_eval.json"
    
    continue_from_run=None
    
    #*****************dir name***************
    if alg_name=="BIRD" and aerfa!=0:
        dir_name=dir_name+'_'+str(beta)+'_'+str(aerfa)

    if alg_name=="FT" and aerfa!=0:
        dir_name=dir_name+'_'+str(beta)+'_'+str(aerfa)


    if alg_name=="MEMIT" and aerfa!=0:
        dir_name=dir_name+'_'+str(aerfa)


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
    #ds = ds_class(DATA_DIR, size=dataset_size_limit)
    ds = ds_class(DATA_DIR, size=dataset_size_limit)
    
    '''
    #shuffle
    ds_list=[]
    for i in ds:
        ds_list.append(i)
    
    random.shuffle(ds_list)
    
    ds=ds_list[:dataset_size_limit]
    '''
    #print(ds)
    #**********************
    #random.shuffle(ds)

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
        hparams.aerfa=args.aerfa
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
        hparams.beta=args.beta
        hparams.aerfa=args.aerfa

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
    )
    exec_time = time() - start
    print("Execution took", exec_time)

    # Evaluate new model
    start = time()
    #gen_test_vars = [None, None]
    '''
    i=0
    for record in ds:
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue

        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": all_metrics[i]["post"],
            "pre": all_metrics[i]["pre"],
        }

        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)
        i+=1
    '''


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
    args = parser.parse_args()
    if args.dir_name=="cf":
        args.dir_name=args.alg_name

    main(args.alg_name, args.model_name, args.ds_name, args.dataset_size_limit, args.dir_name, args.num_edits, args.cuda, args.aerfa,args.beta)