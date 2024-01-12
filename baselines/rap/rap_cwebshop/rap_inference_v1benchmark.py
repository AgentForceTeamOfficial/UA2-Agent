import pickle
from typing import Type, Callable, Optional

# import fire
import random
import numpy as np
# from tqdm import tqdm
import time, re
import json
from datetime import datetime

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append("../")
from reasoners import LanguageModel, Reasoner, SearchAlgorithm
# from reasoners.benchmark import BWEvaluator
from reasoners.algorithm import MCTS

sys.path.append('.')
from world_model import CWebShopWorldModel
from search_config import CWebShopConfig

sys.path.append('../../../')
# from environments.env_ori_cwebshop_runtime import cwebshopRunTimeEnv
from environments.env_instr_list_cwebshop_runtime_session import cwebshopRunTimeEnv

env = cwebshopRunTimeEnv(init_money=10000, init_time=10000)

# single-step reasoning for cwebshop
def get_rap_reasoner_cwebshop(base_model: LanguageModel,
           prompt: dict,
           search_algo: Type[SearchAlgorithm] = MCTS,
           data_path: str = 'data',
           resume: int = 0,
           depth_limit: int = 6,
           reward_alpha: float = 0.5,
           batch_size = 1,
           goal_reached_reward = 100,
           goal_reward_default = 0.,
           cum_reward: Callable[[list[float]], float] = sum,
           calc_q: Callable[[list[float]], float] = np.mean,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           domain_file: str = "",
           config_file: str = "",
           lm_plan_file: str = 'lm_plan.tmp',
           **search_algo_params):

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, "depth_limit": depth_limit, "disable_tqdm": True, "n_iters": depth_limit*2}
    world_model = CWebShopWorldModel(base_model=base_model, prompt=prompt, batch_size=batch_size, max_steps=depth_limit)
    config = CWebShopConfig(base_model=base_model, prompt=prompt, batch_size=batch_size,
                      reward_alpha=reward_alpha, goal_reached_reward=goal_reached_reward,
                      goal_reward_default=goal_reward_default)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    init_prompt = prompt['icl_actor']
    # init_prompt = init_prompt.replace('<current_trial>', init_obs)

    return reasoner, init_prompt


def find_single_right_bracket(str0):
    bracket_cnt = 1
    for i in range(len(str0)):
        if str0[i] == "[": bracket_cnt += 1
        if str0[i] == "]": bracket_cnt -= 1
        if bracket_cnt == 0:
            return i


def cwebshop_run(user_idx, task_idx, instruction_history:list, prompt, to_print=False, model="gpt-3.5-turbo-instruct-0914",
        prompt_path: str = './prompts/prompt.json',
        data_path: str = "",
        disable_log: bool = False,
        config_file: str = "",
        domain_file: str = "",
        lm_plan_file: str = '',
        depth_limit: int = 3,
        **kwargs):
    print(f"**************USER-{user_idx} TASK-{task_idx}**************")
    from reasoners.lm.api_openai_model import GPTCompletionModel
    with open(prompt_path) as f:
        prompt = json.load(f)
    openai_api_model = GPTCompletionModel(model=model)
    rap_reasoner, rap_init_prompt = get_rap_reasoner_cwebshop(openai_api_model,
                prompt,
                disable_log=disable_log,
                data_path=data_path,
                config_file=config_file,
                domain_file=domain_file,
                depth_limit=depth_limit,
                lm_plan_file=lm_plan_file, **kwargs)

    init_prompt = prompt
    current_trial = ""
    action = 'reset'
    is_task_done = False
    current_page = ""
    instruction = "None"
    for i in range(15):
        print(f"STEP_{i}")
        url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action)

        st_pos    = all_obs.find('Observation_from_Interactive_Env[\n')
        inter_obs = all_obs[st_pos+len('Observation_from_Interactive_Env[\n'):]
        en_pos    = find_single_right_bracket(inter_obs)
        inter_obs = inter_obs[:en_pos-1] # strip the last '\n'

        # if action.startswith('think'):
            # inter_obs = 'OK.'
        inter_obs = inter_obs.replace("[Instruction History]", "")
        if to_print:
            print(f'Action: {action}\nObservation: {inter_obs}\n')

        if i:
            current_trial += f'\n\nAction: {action}\nObservation: {inter_obs}'
            # ret_traj.append([action, all_obs, inter_obs, all_used_time, all_used_money])
        else:
            user_profile_instruction = inter_obs[:inter_obs.find('[Search]')-1]
            instruction = re.findall(r'Instruction:  \n(.*)', user_profile_instruction)[0]
            current_trial += f'{inter_obs}'

        if done:
            is_task_done = True
            break
        
        if "[Search]" in inter_obs or "[Back to Search]" in inter_obs:# jump to a new webpage
            current_page = inter_obs

        # rap_init_prompt.replace('<current_trial>', current_trial)
        if len(instruction_history)>0:
            action = rap_reasoner(rap_init_prompt.replace('<current_trial>', current_trial+'\n\nInstruction History:\n'+'\n'.join(instruction_history)), current_page=current_page)
        else:
            action = rap_reasoner(rap_init_prompt.replace('<current_trial>', current_trial), current_page=current_page)

        Qs = [st.Q for st in action.tree_state.children]
        action = action.tree_state.children[np.argmax(Qs)].action

    if not is_task_done:
        inter_rwd = 0
    openai_api_used_money, openai_api_used_time = openai_api_model.get_cost()
    return instruction, inter_rwd, all_used_time + openai_api_used_time, all_used_money + openai_api_used_money#, ret_traj


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    prompt1  = """C-Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""
    to_print     = True
    L            = 0
    USER_NUM     = 1
    TASK_NUM     = 50
    USER_ID      = 3
    instr_history_window_size = 10
    file_name      = f"./rap_w_history_cwebshopRunTimeEnv_v1_U{USER_NUM}_T{TASK_NUM}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt"
    url_log_name   = f"./rap_w_history_cwebshopRunTimeEnv_v1_U{USER_NUM}_T{TASK_NUM}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}_action_url.txt"
    avg_reward   = []
    success_rate = []
    avg_time     = []
    avg_money    = []
    
    for i in range(USER_NUM):
        avg_reward.append(0.0)
        success_rate.append(0.0)
        avg_time.append(0.0)
        avg_money.append(0.0)
        # instructions = []
        instructions = []
        for j in range(TASK_NUM):
            if i == 0 and j == 0:
                logfile = open(file_name, "w")
            else:
                logfile = open(file_name, "a")
        
            instruction, rwd, all_used_time, all_used_money = cwebshop_run(USER_ID, j, instructions[-instr_history_window_size:], prompt1, to_print=to_print)
            instructions.append(instruction)
            avg_reward[i]   += rwd
            success_rate[i] += (rwd == 1.0)
            avg_time[i]     += all_used_time
            avg_money[i]    += all_used_money
            
            print(f"Now user_{USER_ID} task_{j}:"             , file=logfile)
            print(f"\tReward = {rwd}"                   , file=logfile)
            print(f"\tAll Used Time = {all_used_time}"  , file=logfile)
            print(f"\tAll Used Money = {all_used_money}", file=logfile)
            logfile.close()


    with open(file_name, "a") as logfile:
        for i in range(USER_NUM):
            print(file=logfile)
            print(f"OVERALL RESULTS for user_{USER_ID}:", file=logfile)
            print(f"\tAvg Reward     = {avg_reward[i]/TASK_NUM}"       , file=logfile)
            print(f"\tSuccess Rate   = {success_rate[i]/TASK_NUM}"     , file=logfile)
            print(f"\tAvg Used Time  = {avg_time[i]/TASK_NUM}"         , file=logfile)
            print(f"\tAvg Used Money = {avg_money[i]/TASK_NUM}"        , file=logfile)
        
    with open(file_name, "a") as logfile:   
        print(file=logfile)
        print(f"OVERALL RESULTS for user_{USER_ID}:", file=logfile)
        print(f"\tAvg Reward     = {sum(avg_reward)/USER_NUM/TASK_NUM}"       , file=logfile)
        print(f"\tSuccess Rate   = {sum(success_rate)/USER_NUM/TASK_NUM}"     , file=logfile)
        print(f"\tAvg Used Time  = {sum(avg_time)/USER_NUM/TASK_NUM}"         , file=logfile)
        print(f"\tAvg Used Money = {sum(avg_money)/USER_NUM/TASK_NUM}"        , file=logfile)