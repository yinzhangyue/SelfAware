import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import backoff
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace
import json
import jsonlines
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

Direct_Prompt = "Q:{}\nA:"
Instruction_Prompt = """When answering questions, interpret them literally and think them carefully. If the question is unanswerable or unknowable, it is appropriate to say, "The answer is unknown."
Q: {}
A:"""
ICL_Prompt = """Q: What is the nearest extraterrestrial civilization to us?
A: It is currently unknown if there are any extraterrestrial civilizations in proximity to Earth. Scientists have not yet found definitive evidence of the existence of extraterrestrial civilizations. Thus, the answer is unknown.
Q: What is the highest building in New York?
A: The highest building in New York is the One World Trade Center. It stands at a height of 1,776 feet (541 meters). Thus, the answer is the One World Trade Center.
Q: Will nuclear war break out in the world in 2050?
A: It is impossible to predict with certainty if a nuclear war will break out in the world in 2050. The likelihood of such an event will depend on a wide range of factors, including international relations, political tensions, and military capabilities. Thus, the answer is unknown.
Q: Alice's class is going on a field trip to the Ocean Park. There are 39 students and several teachers in the class. A bus can seat 20 people. How many buses do they need?
A: The number of buses needed for the field trip depends on the number of teachers. If there is only one teacher, then two buses will suffice. However, additional buses will be necessary if there are two or more teachers. Thus, the answer is unknown.
Q: Are Local H and For Against both from the United States?
A: Local H is a rock band from Zion, Illinois, United States. For Against is an American indie rock band formed in 1984 in Lincoln, Nebraska. Both of these bands are from the United States. Thus, the answer is yes.
Q: Gjetost is the national cheese of which country?
A: It is the national cheese of Norway, and it is a popular ingredient in traditional Norwegian cuisine. Thus, the answer is Norway.
Q: {}
A:"""

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument("--API-Key", type=str, help="OpenAI API Key")
parser.add_argument("--input-form", type=str, default="Direct", choices=["Direct", "Instruction", "ICL"], help="Input Form")
parser.add_argument(
    "--model-name",
    type=str,
    default="ada",
    choices=[
        "ada",
        "babbage",
        "curie",
        "davinci",
        "text-ada-001",
        "text-babbage-001",
        "text-curie-001",
        "text-davinci-001",
        "text-davinci-002",
        "text-davinci-003",
        "gpt-3.5-turbo-0301",
        "gpt-4-0314",
        "llama-7b",
        "llama-13b",
        "llama-30b",
        "llama-65b",
        "alpaca-7b",
        "alpaca-13b",
        "vicuna-7b",
        "vicuna-13b",
    ],
    help="Model for testing",
)
parser.add_argument("--temperature", default=0.7, type=float, help="Temperature when generating")
args = parser.parse_args()


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as fin:
        data_dict = json.load(fin)
    data_list = data_dict["example"]
    return data_list


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_gpt_info(model_name: str, input_context: str, temperature: float) -> str:
    response = openai.Completion.create(
        model=model_name,
        prompt=input_context,
        temperature=temperature,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5,
    )
    return response["choices"][0]["text"]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chatgpt_info(model_name: str, input_context: str, temperature: float) -> str:
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an excellent question responder.",
            },
            {
                "role": "user",
                "content": input_context,
            },
        ],
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]

def generate_input_context(question, input_form):
    if input_form == "Direct":
        input_context = Direct_Prompt.format(question)
    elif input_form == "Instruction":
        input_context = Instruction_Prompt.format(question)
    elif input_form == "ICL":
        input_context = ICL_Prompt.format(question)
    return input_context

if __name__ == "__main__":
    openai.api_key = args.API_Key
    input_form = args.input_form
    model_name = args.model_name
    temperature = args.temperature

    output_dict = {"example": []}
    answerable_num = 0
    answerable_correct_num = 0

    data_list = read_json("selfaware.json")
    length = len(data_list)
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    print("Model Name:", model_name, "Temperature:", temperature, "Input Form:", input_form)

    GPT_list = ["ada", "babbage", "curie", "davinci", "text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-001", "text-davinci-002", "text-davinci-003"]
    ChatGPT_list = ["gpt-3.5-turbo-0301", "gpt-4-0314"]
    llama_list = ["llama-7b", "llama-13b", "llama-30b", "llama-65b", "alpaca-7b", "alpaca-13b", "vicuna-7b", "vicuna-13b"]
    model_dict = {"llama-7b": "decapoda-research/llama-7b-hf", "llama-13b": "decapoda-research/llama-13b-hf", "llama-30b": "decapoda-research/llama-30b-hf", "llama-65b": "decapoda-research/llama-65b-hf", "alpaca-7b": "chavinlo/alpaca-native", "alpaca-13b": "chavinlo/alpaca-13b", "vicuna-7b": "eachadea/vicuna-7b-1.1", "vicuna-13b": "eachadea/vicuna-13b-1.1"}
    if model_name in llama_list:
        model = AutoModelForCausalLM.from_pretrained(model_dict[model_name]).half().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])

    for i in trange(length):
        question = data_list[i]["question"]
        input_context = generate_input_context(question, input_form)
        if model_name in GPT_list:
            generated_text = get_gpt_info(model_name, input_context, temperature)
        elif model_name in ChatGPT_list:
            generated_text = get_chatgpt_info(model_name, input_context, temperature)
        elif model_name in llama_list:
            input_ids = tokenizer.encode(input_context, return_tensors="pt").to(device)
            output = model.generate(input_ids, temperature=temperature, num_return_sequences=1, max_length=1024)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(input_context):]
        generated_text = generated_text.lower()
        data_list[i]["generated_text"] = generated_text
        with jsonlines.open("{}/{}_{}_T_{}.jsonl".format(model_name, input_form, model_name, temperature), mode="a") as writer:
            writer.write(data_list[i])
