from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from ...util import nethook

from .ft_hparams import FTHyperParams
from .const import relationid_words




def locate_tokenize(model_name, tok, requests_reverse,subject):

    print(requests_reverse,"  ", subject)
    if 'llama' in model_name.lower() or "vicuna" in model_name.lower():
        
        if len(tok.tokenize(requests_reverse))>len(tok.tokenize(subject)):
            sub_tokenize=[len(tok.tokenize(requests_reverse))-len(tok.tokenize(subject)) +1 , len(tok.tokenize(requests_reverse)) + 1]
        else:
            sub_tokenize=[1, len(tok.tokenize(subject))+1]
    else:
        if len(tok.tokenize(requests_reverse))>len(tok.tokenize(subject)):
            sub_tokenize=[len(tok.tokenize(requests_reverse))-len(tok.tokenize(subject)), len(tok.tokenize(requests_reverse))]
        else:
            sub_tokenize=[0,len(tok.tokenize(subject))]
    return {"s":sub_tokenize}
    





def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)
    
    upd_matrixs={i:[] for i in list(deltas.keys())}

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

            #rank = np.linalg.matrix_rank(w.cpu().numpy())
            rank=0
            upd_matrixs[w_name]=[w.cpu().numpy(),rank]
            #print(w.device)
            print("Rank of the matrix:", rank)



    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy, upd_matrixs


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    
    print(requests)
    
    location_sub=[]
    location_re=[]
    location_re_inv=[]
    location_ob=[]
    location_ob_true=[]


    print("requests", requests)
    model_name="gpt2-xl"

    relation_word=relationid_words[request["relation_id"]][0]
    relation_word_inv=relationid_words[request["relation_id"]][1]

    
    sen="This is an entitiy or raltion: "
    location_sub.append(locate_tokenize(model_name, tok, sen+requests[0]["subject"], requests[0]["subject"]))
    location_ob.append(locate_tokenize(model_name, tok, sen+requests[0]["target_new"], requests[0]["target_new"]))
    location_ob_true.append(locate_tokenize(model_name, tok, sen+requests[0]["ground_truth"], requests[0]["ground_truth"]))
    location_re.append(locate_tokenize(model_name, tok, sen+relation_word, relation_word))
    location_re_inv.append(locate_tokenize(model_name, tok, sen+relation_word_inv, relation_word_inv))
    print("location of hrt is", location_sub, '\n\n', location_re, '\n\n',location_re_inv,'\n\n', location_ob, '\n\n', location_ob_true, '\n\n')    

    model_layers=model.config.n_layer if "gpt" in model_name else model.config.num_hidden_layers
    

    
    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    inputs_sub = tok(sen+requests[0]["subject"], return_tensors="pt", padding=True).to(device)
    inputs_tar = tok(sen+requests[0]["target_new"], return_tensors="pt", padding=True).to(device)
    inputs_re = tok(sen+relation_word, return_tensors="pt", padding=True).to(device)
    inputs_re_inv = tok(sen+relation_word_inv, return_tensors="pt", padding=True).to(device)
    inputs_true = tok(sen+requests[0]["ground_truth"], return_tensors="pt", padding=True).to(device)


    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tok.unk_token_id
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            if 't5' in hparams.model_name.lower():
                inputs['labels'] = target_ids
                logits = model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['labels'].unsqueeze(-1)).squeeze(-1)

                mask = inputs['labels'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            else:
                probs = torch.nn.functional.log_softmax(
                    model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
                )
                loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                    1
                ) / loss_mask.sum(1)
                loss = loss.mean()
                

                hidden_sub = model(**inputs_sub, output_hidden_states=True).hidden_states[model_layers].squeeze(0)
                hidden_tar = model(**inputs_tar, output_hidden_states=True).hidden_states[model_layers].squeeze(0)
                hidden_re = model(**inputs_re, output_hidden_states=True).hidden_states[model_layers].squeeze(0)
                hidden_inv = model(**inputs_re_inv, output_hidden_states=True).hidden_states[model_layers].squeeze(0)
                #hidden_true = model(**inputs_true, output_hidden_states=True).hidden_states[model_layers].squeeze(0)


                hrt_loss=None
                #mean 模式

                s=location_sub[0]["s"]
                r=location_re[0]["s"]
    
                r_inv=location_re_inv[0]["s"]
                o=location_ob[0]["s"]
                o_t=location_ob_true[0]["s"]
                '''
                hrt = hidden_sub[0][:].mean(dim=0)+hidden_re[0][:].mean(dim=0)-hidden_tar[0][:].mean(dim=0)
                hrt_inv = hidden_tar[0][:].mean(dim=0)+hidden_inv[0][:].mean(dim=0)-hidden_sub[0][:].mean(dim=0)
                '''

                hrt = hidden_sub[0][s[0]:s[1]].mean(dim=0)+hidden_re[0][r[0]:r[1]].mean(dim=0)-hidden_tar[0][o[0]:o[1]].mean(dim=0)
                hrt_inv = hidden_tar[0][o[0]:o[1]].mean(dim=0)+hidden_inv[0][r_inv[0]:r_inv[1]].mean(dim=0)-hidden_sub[0][s[0]:s[1]].mean(dim=0)
                #hrt1=hidden_hrt[i][o[0]:].mean(dim=0)+hidden_hrt[i][r[0]:r[1]].mean(dim=0)-hidden_hrt[i][s[0]:s[1]].mean(dim=0)

                hrt_norm=hrt.norm(p=2,dim=0)
                hrt_inv_norm=hrt_inv.norm(p=2,dim=0)
                '''

                hrt_true=hidden_sub[0][:].mean(dim=0)+hidden_re[0][:].mean(dim=0)-hidden_true[0][:].mean(dim=0)
                hrt_true_inv=hidden_true[0][:].mean(dim=0)+hidden_inv[0][:].mean(dim=0)-hidden_sub[0][:].mean(dim=0)

                hrt_true_norm=hrt_true.norm(p=2,dim=0)
                hrt_true_inv_norm=hrt_true_inv.norm(p=2,dim=0)
                '''
    
                #print(hrt_norm)
                if hrt_loss==None:
                    hrt_loss = hrt_norm+hrt_inv_norm
                else:
                    hrt_loss = hrt_loss + hrt_norm+hrt_inv_norm
                print(loss, hrt_norm,hrt_inv_norm)
                
                if hrt_loss > 0.1:
                    loss=loss + hparams.aerfa*hrt_loss
                
                #print(hparams.beta)
                
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
