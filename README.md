# Perturbation-restrained

This is the repository for our paper: Perturbation-Restrained Sequential Model Editing.

## Overview
**Model editing** aims to adjust an initial base model's $(f_\theta)$ behavior($x_e \rightarrow y_e$) on the particular edit descriptor $[x_e, y_e]$ efficiently.
However, current model editing methods significantly compromise the general abilities of LLMs as the number of edits increases, and this trade-off poses a substantial challenge to the continual learning of LLMs.

In this paper, we first theoretically analyze that the factor affecting the general abilities in sequential model editing, and then propose a framework termed Perturbation Restraint on Upper bouNd for Editing (PRUNE).


## Datasets
This paper use two editing datasets: CounterFact and ZsRE.
Here we provide the description of how to use CounterFact.

For the downstream tasks, Reasoning, Summarization, Open-domain QA, and Natural language inference (NLI) are used. Their datasets are putted in `data/task-data/`.

```bash
data/
    |__ counterfact.json
    |__ task-data/
        |__ test-OpenDomainQA.jsonl
        |__ test-reasoning.jsonl
        ...
```


## Prepare the environment

### Requirements

**Note: Please use Python 3.9+**
To get started, simply install conda and run:

```shell
conda create -n Perturbaion python=3.9.7
...
pip install -r requirements.txt
```

### Models
All models are putted in `hugging_cache/<model_name>` (model_name=gpt2-xl, llama2-7b, or llama3-8b).
These could be changed in `hparams/<method_name>/`.


## Evaluation

### Datasets and Metrics
The **editing performance** of knowledge editing is measured from these dimensions:

- `Efficacy`: whether the edited models could recall the exact editing fact under editing prompts
- `Generalization`: whether the edited models could recall the editing fact under paraphrase prompts
- `Locality`: whether the output of the edited models for inputs out of editing scope remains unchanged after editing


- These model editing methods are used in our paper as follows:
  - [MEND](https://github.com/eric-mitchell/mend): Mitchell et al. Hypernetwork
  - [ROME](https://github.com/kmeng01/rome): Kevin Meng et al. Locate and Edit
  - [MEMIT](https://github.com/kmeng01/memit): Kevin Meng et al. Locate and Edit

**The downstream task performance** is measured , which is described in our paper.

Here are two steps for evaluation:

### 1. Save the weights of edits
After downloading the datasets and models, to get started (e.g. using ROME to edit GPT-2 XL on CounterFact dataset), run:
```bash
python edit_save.py \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --ds_name=cf \
    --dir_name=ROME \
    --cuda=0 \
    --dataset_size=100 (optional)
```
This could save the weights of edited matrices of different edits (saved in `results/counter/ROME`).
You can change the edits in `easyeditor/editors/bi_editor_save.py` (line 315)

All params are in the `hparams/<alg_name>/`, and you can change them as needed.

### 2. Load the saved weights and evaluation

Here we give examples of running 100 edits then evaluate downstream tasks with unrestrained editing method.
run:

```bash
python edit_load.py \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --ds_name=cf \
    --dir_name=ROME \
    --cuda=0 \
    --dataset_size=100 \
    --seq_num=100
```
Editing results from each run are stored at `results/<ds_name>/<alg_name>_load_100/run_<run_id>`.


To summarize the editing performance (e.g. using ROME to edit GPT-2 XL on counterfact dataset), run:

```bash
python -m experiments.summarize  --dir_name=cf/ROME_load_100/gpt2-xl
```

The downstream task performance is in `test-result/gpt2-xl/<tasks_name>/original`


If you want to use the proposed PRUNE, run:

```bash
python edit_load.py \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --ds_name=cf \
    --dir_name=ROME \
    --cuda=0 \
    --dataset_size=100 \
    --seq_num=100 \
    --reduce_type=log (optional, or: log1_2, log1_5)
```
log1_2 refers to $\alpha$ is 1.2, log is $e$.

Editing results from each run are stored at `results/<ds_name>/reduce_log/<alg_name>_load_100/run_<run_id>`.


To summarize the editing performance (e.g. using ROME to edit GPT-2 XL on counterfact dataset), run:

```bash
python -m experiments.summarize  --dir_name=cf/reduce_log/ROME_load_100/gpt2-xl
```

The downstream task performance is in `test-result/gpt2-xl/<tasks_name>/reduce_log`



### Trainer
To use the MEND method, you should firstly train a hypernetwork using the data in `data/counterfact-train.json`,`data/counterfact-val.json`, and these weights would be saved in `data/weights/models/MEND/`.
Then use the same steps above to edit models.
Run:

```bash
python trainer.py
```




