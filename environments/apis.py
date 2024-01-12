import json
import requests
import time
import tiktoken
import os
import sys

ALL_AVAILABLE_MODELS = [
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",        
    "gpt-3.5-turbo-instruct-0914",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-4-1106-preview",
    "text-embedding-ada-002",
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
    "text-embedding-ada-002": [0.0001, 0.0001],
}

if not "OPENAI_API_KEY" in os.environ:
    print("OPENAI_API_KEY not found in environment variables. Please set it.")
    sys.exit(1)
else:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

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
                # response_text = response['choices'][0]['text']
                response_text = [choice['text'] for choice in response['choices']]
            elif model in ["text-embedding-ada-002"]:
                response_text = response["data"][0]["embedding"]
            else:
                # response_text = response['choices'][0]['message']['content']
                response_text = [choice['message']['content'] for choice in response['choices']]
            if model in ["text-embedding-ada-002"]:
                in_token_num = response["usage"]["total_tokens"]
                out_token_num = 0
            else:
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
    if not model in ["text-embedding-ada-002"]:
        if isinstance(response_text, list) and (len(response_text) == 1 or kwargs_dict.get("n", 1) == 1):
            response_text = response_text[0]
  
    end_time = time.time()   

    return response_text, cost, success, int((end_time - start_time)*100)/100

def post_message(prompt, model="gpt-3.5-turbo-0301", kwargs_dict={"max_tokens": 100, "temperature": 0.0}):
    if model in ["gpt-3.5-turbo-instruct-0914"]:
        url = "https://api.openai.com/v1/completions"

        data = {
                "model": model,
                "prompt": prompt,
            }
    elif model in ["text-embedding-ada-002"]:
        url = "https://api.openai.com/v1/embeddings"

        data = {
                "model": model,
                "input": prompt,
            }
    else:
        url = "https://api.openai.com/v1/chat/completions"

        data = {
                "model": model,
                "messages": json.loads(prompt),
            }
    if not model in ["text-embedding-ada-002"]:
        data.update(kwargs_dict)
    json_msg_data = json.dumps(data)

    success = True
    try:
        response = requests.post(url=url, data=json_msg_data, timeout=1000, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + OPENAI_API_KEY,
        }).text.strip()

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

if __name__ == "__main__":
    if True:
        prompt = "How are you doing today?"
        response, cost, success, ttime = OpenAI_API_Calling(prompt, model="gpt-3.5-turbo-instruct-0914")
        print(response)
        print('='*100)
        print(cost, success, ttime)    

    if True:
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
            },
            {
                "role": "system",
                "name": "example_user",
                "content": "New synergies will help drive top-line growth.",
            },
            {
                "role": "system",
                "name": "example_assistant",
                "content": "Things working well together will increase revenue.",
            },
            {
                "role": "system",
                "name": "example_user",
                "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
            },
            {
                "role": "system",
                "name": "example_assistant",
                "content": "Let's talk later when we're less busy about how to do better.",
            },
            {
                "role": "user",
                "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
            },       
        ]
        prompt = json.dumps(prompt)
        print(response, success, ttime)
        print('='*100)
        response, cost, success, ttime = OpenAI_API_Calling(prompt, model="gpt-3.5-turbo-1106")
        print(response)
        print('='*100)
        print(cost, success, ttime) 

    if True:
        print('='*100)
        response_text, cost, success, ttime = OpenAI_API_Calling("What's the meaning of life?", model="text-embedding-ada-002")
        # print(response_text)
        print(type(response_text), len(response_text))
        print(cost)
        print(success)
        print(ttime)
