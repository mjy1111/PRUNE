import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np

from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_rewrite_quality_zsre, compute_icl_edit_quality, compute_rewrite_quality_bicounterfact
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
import math


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not np.array_equal(p1.cpu().numpy(), p2.cpu().numpy()):
            return False
    return True



def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


def func_linear(original_s, delta_s):
    max_ori=original_s[0]
    for i in range(len(delta_s)):
        if delta_s[i]>=max_ori:
            delta_s[i]=1/2*delta_s[i]+1/2*max_ori
    return delta_s
    

def func_log(original_s, delta_s):
    max_ori=original_s[0]
    for i in range(len(delta_s)):
        if delta_s[i]>=max_ori:
            delta_s[i]=np.log(delta_s[i])+max_ori-np.log(max_ori)
    return delta_s

def func_logn(original_s, delta_s, n):
    max_ori=original_s[0]
    for i in range(len(delta_s)):
        if delta_s[i]>=max_ori:
            delta_s[i]=math.log(delta_s[i],n)+max_ori-math.log(max_ori,n)
    return delta_s

    


def read_weights(hparams, model, svd, seq_num, Reduce):


    '''
    if hparams.alg_name=="ROME":
        matrix = np.load("results/judge/{}/save_{}.npy".format(hparams.alg_name, seq_num-1), allow_pickle=True)
    else:
        matrix = np.load("results/qa/{}/{}/save_{}.npy".format(hparams.alg_name, hparams.model_name.split("/")[-1], seq_num-1), allow_pickle=True)
    #print(matrix)
    '''
    #matrix = np.load("results/judge/{}/save_{}.npy".format(hparams.alg_name, seq_num-1), allow_pickle=True)
    matrix = np.load("results/counter/{}/{}/save_{}.npy".format(hparams.alg_name, hparams.model_name.split("/")[-1], seq_num-1), allow_pickle=True)
    #print(matrix, "\n\n\n", matrix.item(),"\n\n")
    
    if hparams.alg_name=="MEND":
        for w_name in matrix.item():
            '''
            if w_name!="model.layers.29.mlp.down_proj.weight":
                continue
            '''
            print(w_name)
            with torch.no_grad():
                w = nethook.get_parameter(model, w_name)
                
                #print(w)
                w_origin=w.clone().detach()
    
                #print("Rank of the matrix:", rank)
                #print(matrix.item()[w_name])
                #print(matrix.item()[w_name][0])
                
                w[...] = torch.from_numpy(matrix.item()[w_name][0])
                
                #print(matrix.item()[w_name][0])
                #print(w[0][0], torch.from_numpy(matrix.item()[w_name][0]).numpy())
                
                delta=w-w_origin
                
                #rank = np.linalg.matrix_rank(delta.cpu().numpy())
                #print(delta)
    
                u,s,v=np.linalg.svd(delta.cpu().numpy(),full_matrices=1,compute_uv=1)
                
                u0,s0,v0=np.linalg.svd(w_origin.cpu().numpy(),full_matrices=1,compute_uv=1)
                
                u1,s1,v1=np.linalg.svd(w.cpu().numpy(),full_matrices=1,compute_uv=1)
                
                
                print("\n original: qiyizhi: \n",s0)
                
                print("\n delta: qiyizhi: \n",s)
                #print(s.tolist())
                
                print("\n original+delta: qiyizhi: \n",s1)
                
                #print("w_origin",w_origin,"\n\n","delta",delta,"\n\n")
    
                #print("rank is:", np.linalg.matrix_rank(delta.cpu().numpy()))
                if Reduce==None:
                    continue
                else:
                    #u,s,v=np.linalg.svd(delta.cpu().numpy(),full_matrices=1,compute_uv=1)
                    
                    rank=np.linalg.matrix_rank(delta.cpu().numpy())
                    
                    print("rank is:", rank)
    
                    #select_data=svd
                    if Reduce=="linear":
                        s2=func_linear(s0, s)
                    elif Reduce=="log":
                        s2=func_log(s0, s)
                    elif Reduce=="log2":
                        s2=func_logn(s0, s, 2)
                    elif Reduce=="log1_5":
                        s2=func_logn(s0, s, 1.5)
                    elif Reduce=="log1_2":
                        s2=func_logn(s0, s, 1.2)
                        
                    print("delta1 qiyizhi:", s2,'\n')
                    ##new delta
                    u2=u[:,:rank]
                    #s[0:20]=s[0:20]*0.5
                    s2=np.diag(s2[:rank])
                    v2=v[:rank]
                    
                    delta1=np.dot(np.dot(u2,s2),v2)
                    
                    #print("delta1:", s2,'\n')
    
                    #print(torch.from_numpy(delta1).to(f"cuda:{hparams.device}").equal(delta))
                    print(torch.from_numpy(delta1).to(f"cuda:{hparams.device}")-delta)
                    
                    print("absolute:", np.sum(np.absolute(delta.cpu().numpy())), np.sum(np.absolute(w_origin.cpu().numpy())))
                    
                    print("norm:", np.linalg.norm(x=delta.cpu().numpy()), np.linalg.norm(x=w_origin.cpu().numpy()))
                    
                    w[...] = w_origin+torch.from_numpy(delta1).to(f"cuda:{hparams.device}")

                    u3,s3,v3=np.linalg.svd(w.cpu().numpy(),full_matrices=1,compute_uv=1)
                    print("prune w qiyizhi:", s3,'\n')
                
                #print(np.linalg.svd(delta.cpu().numpy(),full_matrices=1,compute_uv=1),s.tolist(),len(s.tolist()))
                #delta=w-w_origin
                #upd_matrixs[w_name]=[w.cpu().numpy(),rank]
                #print(w.device)
                #print("Rank of the delta:", rank)
            print(f"New weights successfully inserted into {w_name}")
    
    else:
        for layer in sorted(hparams.layers, reverse=True):
            w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            print(w_name)
            with torch.no_grad():
                w = nethook.get_parameter(model, w_name)
                
                #print(w)
                w_origin=w.clone().detach()
    
                #print("Rank of the matrix:", rank)
                #print(matrix.item()[w_name])
                #print(matrix.item()[w_name][0])
                
                w[...] = torch.from_numpy(matrix.item()[w_name][0])
                
                #print(matrix.item()[w_name][0])
                #print(w[0][0], torch.from_numpy(matrix.item()[w_name][0]).numpy())
                
                delta=w-w_origin
                
                #rank = np.linalg.matrix_rank(delta.cpu().numpy())
                #print(delta)
    
                
    
                u,s,v=np.linalg.svd(delta.cpu().numpy(),full_matrices=1,compute_uv=1)
                
                u0,s0,v0=np.linalg.svd(w_origin.cpu().numpy(),full_matrices=1,compute_uv=1)
                
                u1,s1,v1=np.linalg.svd(w.cpu().numpy(),full_matrices=1,compute_uv=1)
                
                
                print("\n original: qiyizhi: \n",s0)
                
                print("\n delta: qiyizhi: \n",s)
                #print(s.tolist())
                
                print("\n original+delta: qiyizhi: \n",s1)
                
                #print("w_origin",w_origin,"\n\n","delta",delta,"\n\n")
    
                #print("rank is:", np.linalg.matrix_rank(delta.cpu().numpy()))
                if Reduce==None:
                    continue
                else:
                    #u,s,v=np.linalg.svd(delta.cpu().numpy(),full_matrices=1,compute_uv=1)
                    
                    rank=np.linalg.matrix_rank(delta.cpu().numpy())
                    
                    print("rank is:", rank)
    
                    #select_data=svd
                    if Reduce=="linear":
                        s2=func_linear(s0, s)
                    elif Reduce=="log":
                        s2=func_log(s0, s)
                    elif Reduce=="log2":
                        s2=func_logn(s0, s, 2)
                    elif Reduce=="log1_5":
                        s2=func_logn(s0, s, 1.5)
                    elif Reduce=="log1_2":
                        s2=func_logn(s0, s, 1.2)

                    print("delta1 qiyizhi:", s2,'\n')
                    ##new delta
                    u2=u[:,:rank]
                    #s[0:20]=s[0:20]*0.5
                    s2=np.diag(s2[:rank])
                    v2=v[:rank]
                    
                    delta1=np.dot(np.dot(u2,s2),v2)
                    
                    #print("delta1:", s2,'\n')
    
                    #print(torch.from_numpy(delta1).to(f"cuda:{hparams.device}").equal(delta))
                    print(torch.from_numpy(delta1).to(f"cuda:{hparams.device}")-delta)
                    
                    print("absolute:", np.sum(np.absolute(delta.cpu().numpy())), np.sum(np.absolute(w_origin.cpu().numpy())))
                    
                    print("norm:", np.linalg.norm(x=delta.cpu().numpy()), np.linalg.norm(x=w_origin.cpu().numpy()))
                    
                    w[...] = w_origin+torch.from_numpy(delta1).to(f"cuda:{hparams.device}")
                    
                    u3,s3,v3=np.linalg.svd(w.cpu().numpy(),full_matrices=1,compute_uv=1)
                    print("prune w qiyizhi:", s3,'\n')
                
                
                #print(np.linalg.svd(delta.cpu().numpy(),full_matrices=1,compute_uv=1),s.tolist(),len(s.tolist()))
                #delta=w-w_origin
                #upd_matrixs[w_name]=[w.cpu().numpy(),rank]
                #print(w.device)
                #print("Rank of the delta:", rank)
            print(f"New weights successfully inserted into {w_name}")
    return model
    #print(f"New weights successfully inserted into {weight_name}")
    
    
    
    


class BiEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')



        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                #print(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' or "vicuna" in self.model_name.lower():
                self.model = LlamaForCausalLM.from_pretrained(self.model_name, device_map='auto')
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm2' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.unk_token_id = 64787
                # self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError
            '''

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            '''
        else:
            self.model, self.tok = self.model_name
        # device_map = {
        #     0: [_ for _ in range(0, 16)],
        #     1: [_ for _ in range(16, 32)],
        #     2: [_ for _ in range(32, 48)]
        # }
        # self.model.parallelize(device_map=device_map)

        #one gpu
        if hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')


  
        #print(self.model.hf_device_map)
        #hparams.device = "5"
        
        '''
        # for llama-13b
        print(self.model.device)
        hparams.device = str(self.model.device).split(":")[1]
        print(str(self.model.device))
        '''
        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             relation_id: Optional[Union[str, List[str]]] = None,
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             reverse_qa_prompts: Optional[Dict] = None,
             reverse_judge_prompts: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             case_result_template=None,
             num_edits1=None,
             svd=None,
             seq_num=None,
             Reduce=False,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')


        requests = self._prepare_requests(prompts, target_new, ground_truth, relation_id, rephrase_prompts,
                                          locality_inputs, portability_inputs, reverse_qa_prompts, reverse_judge_prompts, **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")


        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy


        all_metrics = []
        for i, request in enumerate(requests):

            #print(request)
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
                metrics = {
                    "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request, self.hparams.device, pre_edit=True)
                }
            else:
                metrics = {
                    "pre": compute_rewrite_quality_bicounterfact(self.model, self.tok, request),
                }

            #metrics={"pre":{}}
            all_metrics.append(metrics)

            out_file = Path(case_result_template.format(num_edits1, i))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            '''
            metrics = {
                "case_id": i,
                "num_edits": num_edits1,
                "requested_rewrite": request,
                #"post": all_metrics[i]["post"],
                "pre": all_metrics[i]["pre"],
            }
    
            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)
            '''
        
        if seq_num==0:
            edited_model=self.model
        else:
            edited_model = read_weights(self.hparams, self.model, svd, seq_num, Reduce)
        
        #edited_model=self.model

        for i, request in enumerate(requests):
            start = time()
            '''
            edited_model, weights_copy, upd_matrixs = self.apply_algo(
                self.model,
                self.tok,
                [request],
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
                train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
            )
            
            #add_weights
            
            if self.hparams.alg_name=="ROME":
                matrix_file = np.save("results/judge/{}/save_{}".format(self.hparams.alg_name, i),upd_matrixs)
            else:
                matrix_file = np.save("results/qa/{}/save_{}".format(self.hparams.alg_name, i),upd_matrixs)
            '''
            
            
            
            #edited_model=edited_model
            #print(weights_copy)
            #edited_model_copy=deepcopy(edited_model)
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")

            start = time()
            all_metrics[i].update({
                'case_id': i,
                "requested_rewrite": request,
                "time": exec_time,
                "post": compute_rewrite_quality_bicounterfact(edited_model, self.tok, request),
            })
            #print(compare_models(self.model, edited_model))

            LOG.info(f"Evaluation took {time() - start}")


            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)
            out_file = Path(case_result_template.format(num_edits1, i))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue
    
            metrics = {
                "case_id": i,
                "num_edits": num_edits1,
                "requested_rewrite": request,
                "post": all_metrics[i]["post"],
                "pre": all_metrics[i]["pre"],
            }
    
            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)


        weights_copy={}
        return all_metrics, edited_model, weights_copy

    def batch_edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             relation_id: Optional[Union[str, List[str]]] = None,
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             reverse_qa_prompts: Optional[Dict] = None,
             reverse_judge_prompts: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             case_result_template=None,
             num_edits1=None,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')


        requests = self._prepare_requests(prompts, target_new, ground_truth, relation_id, rephrase_prompts,
                                          locality_inputs, portability_inputs, reverse_qa_prompts, reverse_judge_prompts, **kwargs)

        # assert hasattr(self.hparams, 'batch_size') or \
        #        print(f'Method {self.alg_name} found, pls specify the batch_size....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        all_metrics = []
        for i, request in enumerate(requests):
        
            print(request)
            metrics = {
                    "pre": compute_rewrite_quality_bicounterfact(self.model, self.tok, request),
                }
            all_metrics.append(metrics)

            out_file = Path(case_result_template.format(num_edits1, i))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            '''
            metrics = {
                "case_id": i,
                "num_edits": num_edits1,
                "requested_rewrite": request,
                #"post": all_metrics[i]["post"],
                "pre": all_metrics[i]["pre"],
            }
    
            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)
            '''
        id=0
        for record_chunks in self._chunks(requests, num_edits1):
            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
                train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
            )
            #edited_model=edited_model
            #print(weights_copy)
            #edited_model_copy=deepcopy(edited_model)
            exec_time = time() - start
            LOG.info(f"Execution {num_edits1} editing took {exec_time}")

            start = time()
            for i, request in enumerate(record_chunks):
                all_metrics[id].update({
                    'case_id': id,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_rewrite_quality_bicounterfact(edited_model, self.tok, request),
                })
                LOG.info(f"Evaluation took {time() - start}")
                # case_result_path = base_case_path / f"case_{i}.json"
    
                # Dump metrics in .json
                # with open(case_result_path, "w") as f:
                #     json.dump(metrics, f, indent=1)
                out_file = Path(case_result_template.format(num_edits1, id))
                if out_file.exists():
                    print(f"Skipping {out_file}; already exists")
                    continue
        
                metrics = {
                    "case_id": id,
                    "num_edits": num_edits1,
                    "requested_rewrite": request,
                    "post": all_metrics[id]["post"],
                    "pre": all_metrics[id]["pre"],
                }
        
                # Dump metrics in .json
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)
                id+=1



            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v
            #print(compare_models(self.model, edited_model))



        return all_metrics, edited_model, weights_copy

    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in DS_DICT.values()]) > 0 \
        or print(f'DataSet {ds} not supported yet.')

        is_singleton = SingletonEditor.is_singleton_method(self.alg_name)



        if is_singleton:
            num_edits = 1 # Single editor method found
        else:
            assert hasattr(self.hparams, 'batch_size') or \
                   print(f'Method {self.alg_name} found, pls set the batch_size correctly')

            num_edits = self.hparams.batch_size

        all_metrics = []

        for record_chunks in tqdm(self._chunks(ds, num_edits), desc='Editing dataset', total=len(ds)/num_edits):

            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight
            )
            exec_time = time() - start
            LOG.info(f"Execution took {exec_time}")

            start = time()
            all_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': request['case_id'],
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                }
                all_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                all_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                      self.hparams.device)

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")

        return all_metrics, edited_model, weights_copy




    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]


    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          relation_id: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[Dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          reverse_qa_prompts: Optional[Dict] = None,
                          reverse_judge_prompts: Optional[Dict] = None,
                          **kwargs
                          ):

        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
            "relation_id": relation_id_,
            'portability': {},
            'locality': {},
        }
        for prompt, ground_truth_, target_new_ , relation_id_ in zip(prompts, ground_truth, target_new, relation_id)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_ or print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        if rephrase_prompts is not None:

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompts': rephrase_prompts[i],
                    }
                )
        if locality_inputs is not None:
            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompts': locality_inputs[i],
                    }
                )
        if reverse_qa_prompts is not None:
            for i, request in enumerate(requests):
                request.update(
                    {
                        'reverse_qa_prompts': reverse_qa_prompts[i],
                    }
                )
        if reverse_judge_prompts is not None:
            for i, request in enumerate(requests):
                request.update(
                    {
                        'reverse_judge_prompts': reverse_judge_prompts[i],
                    }
                )
        '''

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                    locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                == len(requests) or print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )
        '''

        if portability_inputs is not None:
            for portability_key in portability_inputs.keys():
                if isinstance(portability_inputs[portability_key]['prompt'], str):
                    portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                    portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
                assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
                == len(requests) or print('One Edit instance needs one portability input.....')

                for i, request in enumerate(requests):
                    request['portability'].update(
                        {
                            portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }
                        }
                    )
        return requests



