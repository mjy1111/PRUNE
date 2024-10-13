import collections
import json
from pprint import pprint
from typing import List, Optional
import math
import numpy as np
from scipy.stats import hmean
from pathlib import Path
#from util.globals import *


def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    for run_dir in (Path("results/{}".format(dir_name)) if not abs_path else dir_name).iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("*case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        #print(files)
        #print(len(files))
        for case_file in files[:20]:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            if "time" in data:
                cur_sum["time"].append(data["time"])

            for prefix in ["pre","post"]:
                # Probability metrics for which new should be lower (better) than true

                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue
                    #print(data[prefix][key], math.isnan(data[prefix][key][0]["target_new"]))
                    

                    sum_key_discrete = f"{prefix}_{key.split('_prompts')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_prompts')[0]}_diff"
                    
                    
                    label=0

                    if data[prefix][key]==[]:
                        continue
                    
                    else:
                        for da in data[prefix][key]:
                            if math.isnan(da["target_new"])==True or math.isnan(da["target_true"])==True:
                                label=1
                                break

                    if label==1:
                        continue
                    
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )


                for key in ["reverse_qa_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue
                    

                    sum_key_discrete = f"{prefix}_{key.split('_prompts')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_prompts')[0]}_diff"

                    label=0

                    if data[prefix][key]==[]:
                        continue
                    
                    else:
                        for da in data[prefix][key]:
                            if math.isnan(da["target_new"])==True or math.isnan(da["target_true"])==True:
                                label=1
                                break

                    if label==1:
                        continue
                    
                    
                    ##filter

                    y=data["pre"][key][0]
                    if y["target_true"] > y["target_new"]:
                        continue


                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                
                for key in ["reverse_judge_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_prompts')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_prompts')[0]}_diff"

                    label=0

                    if data[prefix][key]==[]:
                        continue
                    
                    else:
                        for da in data[prefix][key]:
                            if math.isnan(da["target_new"])==True or math.isnan(da["target_true"])==True:
                                label=1
                                break

                    if label==1:
                        continue

                    #filter
                    
                    x=data["pre"][key][0]
                    if x["target_true"] > x["target_new"]:
                        continue
                    
                    for x in data[prefix][key]:
                        cur_sum[sum_key_discrete].append(
                            np.mean(
                                [
                                    x["target_true"] > x["target_new"]
                                ]
                            )
                        )
                        cur_sum[sum_key_cont].append(
                            np.mean(
                                [
                                    np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                ]
                            )
                        )



                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:


                    for da in data[prefix][key]:
                        if math.isnan(da["target_new"])==True or math.isnan(da["target_true"])==True:
                            label=1
                            break
                    if label==1:
                        continue


                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    )


        '''
        for i in cur_sum:
            print(len(cur_sum[i]))
        '''

        if len(cur_sum) == 0:
            continue
        #print(cur_sum)
        #print(len(cur_sum[next(iter(cur_sum.keys()))]))

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))
        

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)
        
        if "judge" in dir_name:
            for prefix in ["pre", "post"]:
                for k_efficacy, k_generalization, k_specificity,k_judge in [
                    (
                        f"{prefix}_rewrite_success",
                        f"{prefix}_paraphrase_success",
                        f"{prefix}_neighborhood_success",
                        f"{prefix}_reverse_judge_success",
                    ),
                    # (
                    #     f"{prefix}_rewrite_acc",
                    #     f"{prefix}_paraphrase_acc",
                    #     f"{prefix}_neighborhood_acc",
                    # ),
                ]:
                    if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity,k_judge]):
                        hmean_list = [
                            cur_sum[k_efficacy][0],
                            cur_sum[k_generalization][0],
                            cur_sum[k_specificity][0],
                            cur_sum[k_judge][0],
                        ]
    
                        # if f"{prefix}_ngram_entropy" in cur_sum:
                        #     hmean_list.append(2 ** (cur_sum[f"{prefix}_ngram_entropy"][0] / 100))
                        # if f"{prefix}_reference_score" in cur_sum:
                        #     hmean_list.append(cur_sum[f"{prefix}_reference_score"][0])
    
                        cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)
                        break
        if "qa" in dir_name:
            for prefix in ["pre", "post"]:
                for k_efficacy, k_generalization, k_specificity, k_qa, k_judge,in [
                    (
                        f"{prefix}_rewrite_success",
                        f"{prefix}_paraphrase_success",
                        f"{prefix}_neighborhood_success",
                        f"{prefix}_reverse_qa_success",
                        f"{prefix}_reverse_judge_success",
                    ),
                    # (
                    #     f"{prefix}_rewrite_acc",
                    #     f"{prefix}_paraphrase_acc",
                    #     f"{prefix}_neighborhood_acc",
                    # ),
                ]:
                    if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity, k_qa, k_judge]):
                        
                        reverse= (cur_sum[k_qa][0] + cur_sum[k_judge][0])/2
                        hmean_list = [
                            cur_sum[k_efficacy][0],
                            cur_sum[k_generalization][0],
                            cur_sum[k_specificity][0],
                            reverse
                        ]
    
                        # if f"{prefix}_ngram_entropy" in cur_sum:
                        #     hmean_list.append(2 ** (cur_sum[f"{prefix}_ngram_entropy"][0] / 100))
                        # if f"{prefix}_reference_score" in cur_sum:
                        #     hmean_list.append(cur_sum[f"{prefix}_reference_score"][0])
    
                        cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)
                        break

        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
    )
