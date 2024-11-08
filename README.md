# Do Large Language Models Know What They Don’t Know?

![License](https://img.shields.io/badge/License-Apache%20License%202.0-green)
![Data License](https://img.shields.io/badge/Data%20License-CC--BY--SA--4.0-blue)
![python](https://img.shields.io/badge/python-3.8+-blue.svg)

Findings of ACL 2023: [Do Large Language Models Know What They Don’t Know?](https://aclanthology.org/2023.findings-acl.551/)


## Introduction 📝

This repository contains the code and data related to the paper "[Do Large Language Models Know What They Don’t Know?](https://arxiv.org/pdf/2305.18153.pdf)". In this paper, we explore and analyze the self-knowledge of large language models, and provide a dataset called *SelfAware* for evaluating the self-knowledge of these models. *SelfAware* includes 1032 unanswerable questions and 2337 answerable questions. We use the F1 score to evaluate the self-knowledge of large language models, and propose an automated method for extracting uncertainty meanings from the outputs of large language models.
![Know-Unknow Quadrant](figures/know-unknow.png)


## Quick Links 🔗

- [Do Large Language Models Know What They Don’t Know?](#do-large-language-models-know-what-they-dont-know)
  - [Introduction 📝](#introduction-)
  - [Quick Links 🔗](#quick-links-)
  - [Requirements 📚](#requirements-)
  - [Data 💾](#data-)
    - [Additional Datasets 📁](#additional-datasets-)
  - [Reproducing Baselines 🚀](#reproducing-baselines-)
  - [Reproducing Baselines 🚀](#reproducing-baselines--1)
  - [Evaluation 💻](#evaluation-)
  - [Bug or Questions? 🤔](#bug-or-questions-)
  - [Citation 📖](#citation-)


## Requirements 📚

Please make sure you have the following requirements installed:
- openai
- torch
- backoff
- tenacity
- transformers
- sentencepiece
- jsonlines
- tqdm
- matplotlib
- ipdb
- simcse


## Data 💾

The data is located in the [SelfAware.json](data/SelfAware.json) file. When using it, please adhere to the CC-BY-SA-4.0 license. Below are some examples of the data:

Answerable questions:
```json
{
    "question_id": 3,
    "question": "Which breed of dog is bigger, Japanese Terrier or Cane Corso?",
    "answer": [
        "cane corso"
    ],
    "answerable": true,
    "source": "hotpot_train"
}
```
Unanswerable questions:
```json
{
    "question_id": 2667,
    "question": "Where are all the aliens?",
    "answer": null,
    "answerable": false,
    "source": "SelfAware"
}
```
Each entry consists of a unique question ID, the question text, the answer (which is an array that can contain multiple answers, or null for unanswerable questions), a boolean indicating if the question is answerable or not, and the source from where the question is extracted.


### Additional Datasets 📁

Considering the complexity of natural data, we have constructed additional datasets based on the types of unanswerable questions identified through manual analysis, combined with AI assistants, to facilitate future research and analysis.

- **UMWP Benchmark**: We have expanded the "Too many variables" category into a new benchmark called UMWP. You can access it here: [UMWP Benchmark](https://aclanthology.org/2024.lrec-main.196/).
- **Other Types of Data**: Other types of data have been uploaded to Google Drive. Access them here: [Additional Data](https://drive.google.com/drive/folders/1Yf0g-cyDURv-21R7YHxqeP0EufgapzWt?usp=sharing).

These datasets are different from *SelfAware*, which consists of unanswerable questions genuinely posed by users on the internet. The new datasets are constructed with the help of AI assistants and therefore do not overlap with *SelfAware*.


## Reproducing Baselines 🚀

You can replicate our experiments by running the `run_model.py` script. For instance, here is the command to run the `text-davinci-003` model with only the question as input:
```bash
python run_model.py --input-form Direct --model-name text-davinci-003
```
And this is the command to run the `gpt-3.5-turbo-0301` model with an instruction included in the input:
```bash
python run_model.py --input-form Instruction --model-name gpt-3.5-turbo-0301
```
The script primarily includes three input arguments:
- API-Key: This is your OpenAI API key, which is required to run GPT-related experiments.
- input-form: The form of the input data.
- model-name: The name of the model you're testing.
- temperature: The temperature setting when generating text. It controls the randomness of the predictions. (Default 0.7)


## Reproducing Baselines 🚀

You can replicate our experiments by running the `run_model.py` script. For instance, here is the command to run the `text-davinci-003` model with only the question as input:
```bash
python run_model.py --input-form Direct --model-name text-davinci-003
```
And this is the command to run the `gpt-3.5-turbo-0301` model with an instruction included in the input:
```bash
python run_model.py --input-form Instruction --model-name gpt-3.5-turbo-0301
```
The script primarily includes three input arguments:
- API-Key: This is your OpenAI API key, which is required to run GPT-related experiments.
- input-form: The form of the input data.
- model-name: The name of the model you're testing.
- temperature: The temperature setting when generating text. It controls the randomness of the predictions. (Default 0.7)

**Tips:**

You can also test more models as long as they're compatible with the `generate` method from the transformers library.


## Evaluation 💻

The evaluation code for SelfAware is located in `eval_model.py`. Below is an example of evaluating the `alpaca-7b` model:

```bash
python eval_model.py --filename  alpaca-7b/Direct_alpaca-7b_T_0.7.jsonl
```

The evaluation results include Precision, Recall, F1 score, and the accuracy for answerable questions.


## Bug or Questions? 🤔

If you have any suggestions or questions, feel free to email us at yinzhangyue@126.com. If you encounter any issues while using the code, or if you find any bugs, please open a new issue on GitHub. This is a preliminary work and we are very much open to any constructive feedback that could help us improve. Thank you for your attention!


## Citation 📖

If you are interested in our work, please use the following citation format when referencing our paper:
```bibtex
@inproceedings{yin-etal-2023-large,
    title = "Do Large Language Models Know What They Don{'}t Know?",
    author = "Yin, Zhangyue  and
      Sun, Qiushi  and
      Guo, Qipeng  and
      Wu, Jiawen  and
      Qiu, Xipeng  and
      Huang, Xuanjing",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.551",
    doi = "10.18653/v1/2023.findings-acl.551",
    pages = "8653--8665"
}
```
