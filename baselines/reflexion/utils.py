import os, json
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List, Union

def find_single_right_bracket(str0):
    bracket_cnt = 1
    for i in range(len(str0)):
        if str0[i] == "[": bracket_cnt += 1
        if str0[i] == "]": bracket_cnt -= 1
        if bracket_cnt == 0:
            return i

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(env, user_idx, task_idx, prompt: Union[str, List[str]], max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False, model='gpt-3.5-turbo-instruct-0914') -> Union[str, List[str]]:
    assert (not is_batched and isinstance(prompt, str)) or (is_batched and isinstance(prompt, list))
    assert not is_batched
    # model = 'gpt-3.5-turbo-instruct-0914'
    
    hyper_args = {"stop": stop_strs, "max_tokens": max_tokens, "temperature": 0.0, 'top_p': 1, "frequency_penalty": 0.0, "presence_penalty": 0.0}

    if model == 'gpt-3.5-turbo-instruct-0914':
        messages = prompt
    else:
        messages = json.dumps([{"role": "user", "content": prompt}])

    action   = f"ask[{model}][{messages}][{json.dumps(hyper_args)}]"

    _, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action.strip())
    st_pos = all_obs.find('LLM_response[')
    response = all_obs[st_pos+len('LLM_response['):]
    en_pos = find_single_right_bracket(response)
    response = response[:en_pos]
    response = response.strip()
    # response = openai.Completion.create(
    #     model='text-davinci-003',
    #     prompt=prompt,
    #     temperature=0.0,
    #     max_tokens=max_tokens,
    #     top_p=1,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    #     stop=stop_strs,
    # )
    # return response.choices[0].text
    return response
