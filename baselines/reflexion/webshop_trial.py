import json
import os
import sys
import openai
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from env_history import EnvironmentHistory
sys.path.append("../../")
from environments.env_instr_list_cwebshop_runtime import cwebshopRunTimeEnv

from typing import Any, Dict, List, Tuple
 
def find_single_right_bracket(str0):
    """
    Find the position of the first single right bracket in str0.
    """
    bracket_cnt = 1
    final_pos = 0
    for i in range(len(str0)):
        if str0[i] == '[':
            bracket_cnt += 1
        if str0[i] == ']':
            bracket_cnt -= 1
            final_pos = i
        if bracket_cnt == 0:
            return i
    return final_pos

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}
with open("./base_prompt.txt", 'r') as f:
    BASE_PROMPT = f.read()

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

def cwebshop_run(user_idx, task_idx, env, base_prompt, memory: List[str], to_print=True, model="gpt-3.5-turbo-instruct-0914") -> Tuple[EnvironmentHistory, bool]:
    action = 'reset'
    init_prompt = base_prompt
    prompt = ''

    url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action)
    st_pos = all_obs.find('Observation_from_Interactive_Env[\n')
    inter_obs = all_obs[st_pos+len('Observation_from_Interactive_Env[\n'):]
    en_pos    = find_single_right_bracket(inter_obs)
    inter_obs = inter_obs[:en_pos-1]

    inter_obs = inter_obs.replace("[Instruction History]", "").strip()

    observation = inter_obs
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, observation, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, observation, memory, [])
    env_history.reset()
    init_prompt = env_history._cur_query
    for i in range(15):
        env_history.add("action", action)
        # try:
        url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action)
        st_pos    = all_obs.find('Observation_from_Interactive_Env[\n')
        observation = all_obs[st_pos+len('Observation_from_Interactive_Env[\n'):]
        en_pos    = find_single_right_bracket(observation)
        observation = observation[:en_pos-1]
        # except AssertionError:
            # observation = 'Invalid action!'
        if action.startswith('think'):
            observation = 'OK.'

        observation = observation.replace("[Instruction History]", "")

        if to_print and not action=='reset':
            print(f'Action: {action}\nObservation: {observation}\n')
            sys.stdout.flush()
        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
        else:
            # prompt += f'{observation}\n\nAction:'
            prompt += f'\n\nAction:'

        env_history.add("observation", observation)
        
        # if done, check if reward is complete value
        if done:
            print("done!")
            return env_history, inter_rwd == 1.0, inter_rwd, all_used_time, all_used_money

        hyper_args   = {"stop": ["\n"], "max_tokens": 100, "temperature": 0.0}
        action       = f"ask[{model}][{init_prompt + prompt[-(6400-len(init_prompt)):]}][{json.dumps(hyper_args)}]" 
        while True:
            url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action)
            if to_print and not API_success:
                print(f'Action: {action}\nObservation: {all_obs}\n')
            if API_success: 
                break

        st_pos = all_obs.find('LLM_response[')
        action = all_obs[st_pos+len('LLM_response['):]
        en_pos = find_single_right_bracket(action)
        action = action[:en_pos]

        action = action.strip('" ').replace('\\"','')
        # if '"' in action:
        #     action = action.split('"')[1]
        # action = action.lstrip(' ')

    return env_history, False, inter_rwd, all_used_time, all_used_money

def run_trial(env,
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool
    ) -> List[Dict[str, Any]]:
    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs): # env_configs <-> a user in the benchmark
        if env_config["is_success"]:
            num_successes += 1
            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        try:
            final_env_history, is_success, rwd, all_used_time, all_used_money = cwebshop_run(env_config['user_idx'], z, env, BASE_PROMPT, env_config["memory"] if use_memory else [], to_print=True)
            
            env_config['all_used_money'] = all_used_money
            env_config['all_used_time'] = all_used_time
            env_config['reward'] = rwd
            
            if is_success:
                status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                env_configs[z]["is_success"] = True
                num_successes += 1
                num_additional_successes += 1
            else:
                status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

            # log env results to trial log
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

        except AssertionError:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

            # log env results to trial log
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}:\nAssertion Error\n\nSTATUS: FAIL\n\n#####\n')

        # log to world log
        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs