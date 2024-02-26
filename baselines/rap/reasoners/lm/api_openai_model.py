import os
import numpy as np
from typing import Optional, Union
import time

from .. import LanguageModel, GenerateOutput



# 2023.12.07, yzh, yzy
# This script implements the factory of different LLM APIs.

import json
import requests
import time
from requests.auth import HTTPBasicAuth
import tiktoken

ALL_AVAILABLE_MODELS = [
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",        
    "gpt-3.5-turbo-instruct-0914",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-4-1106-preview",    
]

pricing_ind = {
    "gpt-3.5-turbo-0301": [0.0010, 0.0020],
    "gpt-3.5-turbo-0613": [0.0010, 0.0020],
    "gpt-3.5-turbo-1106": [0.0010, 0.0020],        
    "gpt-3.5-turbo-instruct-0914": [0.0015, 0.0020],            
    "gpt-4-0314": [0.03, 0.06],
    "gpt-4-0613": [0.03, 0.06],
    "gpt-4-0314": [0.03, 0.06],
    "gpt-4-1106-preview": [0.01, 0.03],
}

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-1106"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    if "gpt-3.5-turbo-instruct" in model:
        if not isinstance(messages, str):
            return -1, "Format incorrect", False                
        num_tokens = len(encoding.encode(messages))
        return num_tokens, "", True

    try:
        messages = json.loads(messages)
    except:
        return -1, "Format incorrect", False
        
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",        
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-1106-preview",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        return -1, "API Model not found", False
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens, "", True

def OpenAI_API_Estimate_Pricing(prompt, model="gpt-3.5-turbo-1106", kwargs_dict={"max_tokens": 100, "temperature": 0.0}):
    start_time = time.time()
    in_token_num, error_reason, success  = num_tokens_from_messages(prompt, model=model)

    if not success:
        cost     = -1
        response = f"The estimation has failed this time. Error reason: {error_reason}"
    else:
        cost    = in_token_num * pricing_ind[model][0] / 1000 + kwargs_dict["max_tokens"] * pricing_ind[model][1] / 1000
        response = f"The estimated cost for this API call is ${cost}"

    end_time = time.time()

    return response, success, int((end_time - start_time)*100)/100

def OpenAI_API_Calling(prompt, model="gpt-3.5-turbo-1106", kwargs_dict={"max_tokens": 100, "temperature": 0.0}):
    start_time = time.time()

    if not model in ALL_AVAILABLE_MODELS:
        return None, 0, False, int((time.time() - start_time)*100)/100

    response, success = post_message(prompt, model=model, kwargs_dict=kwargs_dict)
    
    try:
        response_text = None
        # Now that for successful calling, the response is already converted into a list
        if isinstance(response, dict): 
            if model in ["gpt-3.5-turbo-instruct-0914"]:
                response_text = response['choices'][0]['text']
            else:
                response_text = response['choices'][0]['message']['content']

            in_token_num = response["usage"]["prompt_tokens"]
            out_token_num = response["usage"]["completion_tokens"]
            cost = in_token_num * pricing_ind[model][0] / 1000 + out_token_num * pricing_ind[model][1] / 1000
        else:
            response_text = response
            cost = 0
    except:
        response_text = "The API has failed this time. Retry later!"
        cost = 0
        success = False

    end_time = time.time()   

    return response_text, cost, success, int((end_time - start_time)*100)/100     

"""
def API_GPT35(prompt, model="gpt-3.5-turbo-1106"):

    assert model in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct-0914"]

    start_time = time.time()

    response, success = post_message(prompt, model=model)
    if response is not None: cost = len(response) * 0.002
    else: cost = 0

    end_time = time.time()

    return response, cost, success, int((end_time - start_time)*100)/100

def API_GPT4(prompt, model="gpt-4-0613"):

    assert model in ["gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview"]

    start_time = time.time()

    response, success = post_message(prompt, model=model)
    if response is not None: cost = len(response) * 0.01
    else: cost = 0

    end_time = time.time()

    return response, cost, success, int((end_time - start_time)*100)/100
"""

def post_message(prompt, model="gpt-3.5-turbo-0301", kwargs_dict={"max_tokens": 100, "temperature": 0.0}):

    if model in ["gpt-3.5-turbo-instruct-0914"]:
        url = "http://43.163.219.59:8001/alpha"

        data = {
                "model": model,
                "prompt": prompt,
            }
    else:
        url = "http://43.163.219.59:8001/beta"

        data = {
                "model": model,
                "messages": json.loads(prompt),
            }
        
    data.update(kwargs_dict)
    json_msg_data = json.dumps(data)

    success = True
    try:
        response = requests.post(url=url, data=json_msg_data, timeout=1000, auth=HTTPBasicAuth(username="thumt",password="Thumt@2023")).text.strip()

        if response == "" or "Server Error (500)" in response:
            response = "The API has failed this time. Retry later!"
            success  = False
        else:
            response = json.loads(response)
            if "error" in response:
                response = "The API has failed this time. Error: " + response["error"]["message"]
                success  = False
    except:
        response = "The API has failed this time. Retry later!"
        success = False

    return response, success

class GPTCompletionModel(LanguageModel):
    def __init__(self, model:str, max_tokens:int = 512, temperature=0.7):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.money_cost = 0.0
        self.time_cost = 0.0
        # API_KEY = os.getenv("OPENAI_API_KEY", None)
        # if API_KEY is None:
            # raise ValueError("OPENAI_API_KEY not set, please run `export OPENAI_API_KEY=<your key>` to ser it")
        # else:
            # openai.api_key = API_KEY

    
    def generate(self,
                prompt: str,
                max_tokens: int = None,
                
                top_p: float = 1.0,
                num_return_sequences: int = 1,
                rate_limit_per_min: Optional[int] = 20,
                stop: Optional[str] = None,
                logprobs: Optional[int] = None,
                temperature = None,
                **kwargs) -> GenerateOutput:
        
        gpt_temperature = self.temperature if temperature is None else temperature

        if max_tokens is None:
            max_tokens = self.max_tokens
        
        if logprobs is None:
            logprobs = 0

        api_kwargs = {'max_tokens': max_tokens, 'temperature': gpt_temperature, 'top_p':top_p, 
                      'n': num_return_sequences, 'stop': stop, 'logprobs': logprobs}
        success = False
        api_calling_cost, api_calling_latency = 0.0, 0.0
        cnt = 0
        # print("\n",flush=True)
        while not success:
            cnt += 1
            print(f"{cnt}", flush=True, end=",")
            assert len(prompt)==1, "len(prompt) must be 1"
            response_text, cost, success, latency = OpenAI_API_Calling(prompt[0], model=self.model, kwargs_dict=api_kwargs)
            if cnt>1:
                print(response_text)
            api_calling_latency += latency
            api_calling_cost += cost
        # print("\n",flush=True)
        self.money_cost += api_calling_cost
        self.time_cost += api_calling_latency

        if self.model in ["gpt-3.5-turbo-instruct-0914"]:
            # return GenerateOutput(
            #                 text=[choice["text"] for choice in response_text["choices"]],
            #                 log_prob=[choice["logprobs"] for choice in response_text["choices"]]
                        # )
            return GenerateOutput(text=response_text, log_prob=None)
        else:
            # return GenerateOutput(
            #             text=[choice["message"]['content'] for choice in response_text["choices"]],
            #             log_prob=[choice["logprobs"] for choice in response_text["choices"]]
            #         )
            return GenerateOutput(text=response_text, log_prob=None)
    
    def get_cost(self):
        return self.money_cost, self.time_cost


    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("GPTCompletionModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                    prompt: Union[str, list[str]],
                    **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")
    

