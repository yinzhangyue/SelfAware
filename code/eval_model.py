import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import string
from simcse import SimCSE
import torch
import re
import numpy as np
import jsonlines
import argparse
from tqdm import tqdm, trange
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, help="Input Filename", required=True)
parser.add_argument("--threshold", default=0.75, type=float, help="Threshold of similarity")
parser.add_argument("--model", default="princeton-nlp/sup-simcse-roberta-large", type=str, help="Smilarity Model")
args = parser.parse_args()

uncertain_list = [
    "The answer is unknown.",
    "The answer is uncertain.",
    "The answer is unclear.",
    "There is no scientific evidence.",
    "There is no definitive answer.",
    "There is no right answer.",
    "There is much debate.",
    "There is no known case.",
    "There is no concrete answer to this question.",
    "There is no public information available.",
    "It is impossible to know.",
    "It is impossible to answer.",
    "It is difficult to predict.",
    "It is not known.",
    "We do not know.",
    "I'm not sure.",
]

def read_jsonl(filename):
    data_list = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


def remove_punctuation(input_string):
    input_string = input_string.strip().lower()
    if input_string and input_string[-1] in string.punctuation:
        return input_string[:-1]
    return input_string


def cut_sentences(content):
    sentences = re.split(r"(\.|\!|\?|。|！|？|\.{6})", content)
    return sentences

def cut_sub_string(input_string, window_size=5, punctuation=".,?!"):
    input_string = input_string.strip().lower()
    if len(input_string) < 2:
        return [""]
    if input_string[-1] in punctuation:
        input_string = input_string[:-1]
    string_list = input_string.split()
    length = len(string_list)
    if length <= window_size:
        return [input_string]
    else:
        res = []
        for i in range(length - window_size + 1):
            sub_string = " ".join(string_list[i : i + window_size])
            if sub_string != "" or sub_string != " ":
                res.append(sub_string)
        return res


if __name__ == "__main__":
    filename = args.filename
    threshold = args.threshold
    model = SimCSE(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uncertain_list = [remove_punctuation(_) for _ in uncertain_list]
    data_list = read_jsonl(filename)
    length = len(data_list)
    answerable_num = 0
    unanswerable_num = 0
    TP = 0
    FP = 0
    FN = 0
    Acc = 0
    for i in trange(length):
        answerable = data_list[i]["answerable"]
        unanswerable = answerable is False
        if answerable is True:
            answerable_num += 1
        else:
            unanswerable_num += 1
        output = data_list[i]["output"].strip().lower()
        if output == "":
            continue
        pred_unanswerable = False
        for uncertain in uncertain_list:
            if uncertain in output:
                pred_unanswerable = True
        
        if pred_unanswerable is False:
            sub_sen_list = cut_sentences(output)
            sub_str_list = []
            for sub_sen in sub_sen_list:
                if len(sub_sen) >= 2:
                    sub_str_list.extend(cut_sub_string(sub_sen))
            if len(sub_str_list) != 0:
                similarities = model.similarity(sub_str_list, uncertain_list, device=device)
            else:
                similarities = [0]
            max_uncertainty = np.max(similarities)
            if max_uncertainty > threshold:
                pred_unanswerable = True

        if unanswerable is True:
            if pred_unanswerable is True:
                TP += 1
            else:
                FN += 1
        elif pred_unanswerable is True:
            FP += 1

        if unanswerable is False:
            for ans in data_list[i]["answer"]:
                if ans.strip().lower() in output:
                    Acc += 1
                    break

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("Filename:", filename)
    print("Threshold", threshold)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", F1)
    print("Acc", Acc/answerable_num)
