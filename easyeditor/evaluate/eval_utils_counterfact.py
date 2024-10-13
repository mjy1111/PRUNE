"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..util.generate import generate_fast
from ..util.perplexity import perplexity


def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record[x] for x in ["subject", "target_new", "ground_truth"]
    )
    rewrite_prompts = [record["prompt"]]
    paraphrase_prompts = record["rephrase_prompts"]
    neighborhood_prompts = record["locality_prompts"]
    #generation_prompts = record["generation_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new,
        target_true,
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret={}
    ret.update({
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } )
    ret.update({
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    })
    '''
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }
    '''


    return ret





def compute_rewrite_quality_bicounterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record[x] for x in ["subject", "target_new", "ground_truth"]
    )
    rewrite_prompts = [record["prompt"]]
    paraphrase_prompts = record["rephrase_prompts"]
    neighborhood_prompts = record["locality_prompts"]



    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new,
        target_true,
    )


    #eval for reverse*************************
    if "reverse_qa_prompts" in record:
        
        reverse_qa_prompts= [record["reverse_qa_prompts"]["prompt"]]

        re_qa_prob_prompts = [
            reverse_qa_prompts,
        ]
        re_qa_which_correct = [
            [0 for _ in range(len(re_qa_prob_prompts))],
        ]
        re_qa_probs, re_qa_targets_correct = test_batch_prediction(
            model,
            tok,
            list(chain(*re_qa_prob_prompts)),
            list(chain(*re_qa_which_correct)),
            record["reverse_qa_prompts"]["target_new"]["str"],
            record["reverse_qa_prompts"]["target_true"]["str"],
        )
        probs.extend(re_qa_probs)
        targets_correct.extend(re_qa_targets_correct)


    #reverse_judge_compute_probs
    model_name = model.config._name_or_path.replace("/", "_")
    if "reverse_judge_prompts" in record:
        '''
        record["reverse_judge_prompts"]["prompt"]=record["reverse_judge_prompts"]["prompt"].replace("Whether","")
        record["reverse_judge_prompts"]["prompt"]=record["reverse_judge_prompts"]["prompt"].replace("?","")+", is it true or false?"
        
        record["reverse_judge_prompts"]["target_new"]["str"]="True"
        record["reverse_judge_prompts"]["target_true"]["str"]="False"
        '''
         
        
        if "reverse_qa_prompts" in record:
            judge_prompt1=record["reverse_qa_prompts"]["prompt"]+" "+record["reverse_qa_prompts"]["target_new"]["str"]+", is this true?"
            judge_prompt2="Please use yes or no to judge the factuality of the following sentence. Yes means this sentence is correct, and no means it is wrong."+record["reverse_qa_prompts"]["prompt"]+" "+record["reverse_qa_prompts"]["target_new"]["str"]+'. Is this sentence correct?'
            judge_prompt3=record["reverse_qa_prompts"]["prompt"]+" "+record["reverse_qa_prompts"]["target_new"]["str"]+", whether this sentence is true or not?"
            judge_prompt4=record["reverse_qa_prompts"]["prompt"]+" "+record["reverse_qa_prompts"]["target_new"]["str"]+", is the information described in this sentence correct?"
            judge_prompt5=record["reverse_qa_prompts"]["prompt"]+" "+record["reverse_qa_prompts"]["target_new"]["str"]+", is the information described in this sentence true?"
        else:
            judge=record["reverse_judge_prompts"]["prompt"].replace("Whether ","")
            judge=judge.replace("?","")
            judge=judge[0:1].upper()+judge[1:]
            #print(judge)

            judge_prompt1=judge+", is this true?"
            judge_prompt2="Please use yes or no to judge the factuality of the following sentence. Yes means this sentence is correct, and no means it is wrong."+judge+'. Is this sentence correct?'
            judge_prompt3=judge+", whether this sentence is true or not?"
            judge_prompt4=judge+", is the information described in this sentence correct?"
            judge_prompt5=judge+", is the information described in this sentence true?"
        
        if "gpt2-xl" in model_name:
            reverse_judge_prompts= [record["reverse_judge_prompts"]["prompt"]]
            re_judge_prob_prompts = [
                reverse_judge_prompts
            ]
            re_judge_which_correct = [
                [0 for _ in range(len(reverse_judge_prompts))],
            ]
            #reverse_judge_prompts= record["reverse_judge_prompts"]
    
            re_judge_probs1, re_judge_targets_correct1 = test_batch_prediction(
                model,
                tok,
                list(chain(*re_judge_prob_prompts)),
                list(chain(*re_judge_which_correct)),
                record["reverse_judge_prompts"]["target_new"]["str"],
                record["reverse_judge_prompts"]["target_true"]["str"],
            )

            probs.extend(re_judge_probs1)
            targets_correct.extend(re_judge_targets_correct1)


        elif "gpt-j-6B" in model_name:
            reverse_judge_prompts= [judge_prompt4]
            re_judge_prob_prompts = [
                reverse_judge_prompts
            ]
            re_judge_which_correct = [
                [0 for _ in range(len(reverse_judge_prompts))],
            ]
            judge_new="yes, it is true"
            judge_true="no, it is false"

            re_judge_probs1, re_judge_targets_correct1 = test_batch_prediction(
                model,
                tok,
                list(chain(*re_judge_prob_prompts)),
                list(chain(*re_judge_which_correct)),
                judge_new,
                judge_true,
            )

            probs.extend(re_judge_probs1)
            targets_correct.extend(re_judge_targets_correct1) 
        
        elif "llama-7b" in model_name:
            reverse_judge_prompts= [judge_prompt5]
            re_judge_prob_prompts = [
                reverse_judge_prompts
            ]
            re_judge_which_correct = [
                [0 for _ in range(len(reverse_judge_prompts))],
            ]
            judge_new="yes, it is true"
            judge_true="no, it is false"

            re_judge_probs1, re_judge_targets_correct1 = test_batch_prediction(
                model,
                tok,
                list(chain(*re_judge_prob_prompts)),
                list(chain(*re_judge_which_correct)),
                judge_new,
                judge_true,
            )

            probs.extend(re_judge_probs1)
            targets_correct.extend(re_judge_targets_correct1)

        elif "llama2-7b" in model_name or "llama2-13b" in model_name:
            reverse_judge_prompts= [judge_prompt2]
            re_judge_prob_prompts = [
                reverse_judge_prompts
            ]
            re_judge_which_correct = [
                [0 for _ in range(len(reverse_judge_prompts))],
            ]

            judge_new="yes, it is true"
            judge_true="no, it is false"

            re_judge_probs1, re_judge_targets_correct1 = test_batch_prediction(
                model,
                tok,
                list(chain(*re_judge_prob_prompts)),
                list(chain(*re_judge_which_correct)),
                judge_new,
                judge_true,
            )

            probs.extend(re_judge_probs1)
            targets_correct.extend(re_judge_targets_correct1)


        '''

        reverse_judge_prompts= [record["reverse_judge_prompts"]["prompt"],judge_prompt1,judge_prompt2,judge_prompt3,judge_prompt4,judge_prompt5]
        re_judge_prob_prompts = [
            reverse_judge_prompts
        ]
        re_judge_which_correct = [
            [0 for _ in range(len(reverse_judge_prompts))],
        ]
        #reverse_judge_prompts= record["reverse_judge_prompts"]

        re_judge_probs, re_judge_targets_correct = test_batch_prediction(
            model,
            tok,
            list(chain(*re_judge_prob_prompts)),
            list(chain(*re_judge_which_correct)),
            record["reverse_judge_prompts"]["target_new"]["str"],
            record["reverse_judge_prompts"]["target_true"]["str"],
        )



        reverse_judge_true_prompts=[record["reverse_judge_prompts"]["prompt"],judge_prompt1,judge_prompt2,judge_prompt3,judge_prompt4,judge_prompt5]
        judge_new="yes, it is true"
        judge_true="no, it is false"


        re_judge_probs1, re_judge_targets_correct1 = test_batch_prediction(
            model,
            tok,
            list(chain(*re_judge_prob_prompts)),
            list(chain(*re_judge_which_correct)),
            judge_new,
            judge_true,
        )

        probs.extend(re_judge_probs)
        probs.extend(re_judge_probs1)
        targets_correct.extend(re_judge_targets_correct)
        targets_correct.extend(re_judge_targets_correct1)
        '''




    #generation_prompts = record["generation_prompts"]


    if "reverse_qa_prompts" in record and "reverse_judge_prompts" in record:
        prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
            reverse_qa_prompts,
            reverse_judge_prompts,
        ]
    # Unflatten the results again into a list of lists.
        cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
        ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
        ret_corrects = [
            targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
        ]
        # Structure the restuls as a dictionary.
        ret={}
        ret.update({
            f"{key}_probs": ret_probs[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                    "neighborhood_prompts",
                    "reverse_qa_prompts",
                    "reverse_judge_prompts",
                ]
            )
        } )
        ret.update({
            f"{key}_correct": ret_corrects[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                    "neighborhood_prompts",
                    "reverse_qa_prompts",
                    "reverse_judge_prompts",
                ]
            )
        })
    
    if "reverse_qa_prompts" not in record and "reverse_judge_prompts" in record:
        prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
            reverse_judge_prompts,
        ]
        cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
        ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
        ret_corrects = [
            targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
        ]
        # Structure the restuls as a dictionary.
        ret={}
        ret.update({
            f"{key}_probs": ret_probs[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                    "neighborhood_prompts",
                    "reverse_judge_prompts",
                ]
            )
        } )
        ret.update({
            f"{key}_correct": ret_corrects[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                    "neighborhood_prompts",
                    "reverse_judge_prompts",
                ]
            )
        })
    
    if "reverse_judge_prompts" not in record:

        prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
        ]
        cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
        ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
        ret_corrects = [
            targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
        ]
        # Structure the restuls as a dictionary.
        ret={}
        ret.update({
            f"{key}_probs": ret_probs[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                    "neighborhood_prompts",
                ]
            )
        } )
        ret.update({
            f"{key}_correct": ret_corrects[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                    "neighborhood_prompts",
                ]
            )
        })


    return ret




def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """
    device=model.device
    #print(device)
    
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    #print(a_tok, b_tok)
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    

    #model_name = model.config._name_or_path.replace("/", "_")
    if hasattr(model.config,'_name_or_path'):
        model_name = model.config._name_or_path.replace("/", "_")
    else:
        model_name = model.config.model_name
    #print(model_name)
    #print(model.config)
    if 'llama3' not in model_name.lower():
        if 'llama' in model_name.lower() or "vicuna" in model_name.lower():
            #print(1)
            a_tok, b_tok = (tok(f"{n}")["input_ids"] for n in [target_new, target_true])
            a_tok, b_tok = a_tok[1:], b_tok[1:]
            choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        if hasattr(model.config,'_name_or_path'):
            logits = model(**prompt_tok).logits
        else:
            logits = model(**prompt_tok)
        #print(model(**prompt_tok))
    '''
    print(prefix_lens)
    print(prompt_tok)
    print(logits.shape)
    print(choice_a_len, choice_b_len)
    '''


    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):

        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        '''
        #for lama
        if i % 2 == 0:
            n=target_new
        else:
            n=target_true

        if prefix_lens[i//2]+cur_len !=len(tok(prefixes[i//2]+f" {n}")["input_ids"]):
            print("lama tokenize has a 0")
            #delta=len(tok(prefix_lens[i]+f" {n}")["input_ids"])- len(tok(prefix_lens[i])["input_ids"])
            #cur_len=len(tok(prefixes[i//2]+f" {n}")["input_ids"])-prefix_lens[i//2]
            cur_len=cur_len-2
        '''
        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            #prompt_tok[i // 2] 
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct
def test_seq2seq_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    target_true: str,
):
    """ """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    input_tok = tok(
        [
            f"{prefix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    target_tok = tok(
        [
            f"{suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    prompt_tok = dict()
    prompt_tok.update(input_tok)

    prompt_tok['decoder_input_ids'] = target_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = target_tok['attention_mask']

    a_tok, b_tok = (tok(f"{n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i,  j, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]


# def test_generation(
#     model,
#     tok,
#     prefixes: typing.List[str],
#     consistency_texts: typing.List[str],
#     essence_texts: typing.List[str],
#     # vec: TfidfVectorizer,
# ):
#     gen_texts = generate_fast(
#         model,
#         tok,
#         prefixes,
#         n_gen_per_prompt=1,
#         max_out_len=100,
#     )
#
#     ngram_entropy = n_gram_entropy(gen_texts)
#     consistency_tfidf = tfidf_similarity(
#         " ".join(gen_texts), " ".join(consistency_texts), vec
#     )
#
#     ret = {
#         "ngram_entropy": ngram_entropy,
#         "reference_score": consistency_tfidf,
#         "text": gen_texts,
#     }
#
#     if len(essence_texts) > 0:
#         ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
#         ret.update({"essence_score": ppl, "essence_text": essence_texts})
#
#     return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
