import json
import re
import time
from datetime import datetime
import sys
sys.stdout.reconfigure(encoding='utf-8')
from prompt_lib import *
sys.path.append("../")
from environments.env_instr_list_ua2webshop_runtime_session_d import ua2webshopRunTimeEnv
from environments.apis import OpenAI_API_Calling, num_tokens_from_messages
from prompt_lib import ReAct_PROMPT_1shot
from Profiler import MultiRoundProfiler
from Insight import Insight

env = ua2webshopRunTimeEnv(init_money=10000, init_time=10000)

def find_single_right_bracket(str0):
    bracket_cnt = 1
    for i in range(len(str0)):
        if str0[i] == "[": bracket_cnt += 1
        if str0[i] == "]": bracket_cnt -= 1
        if bracket_cnt == 0:
            return i

def num_tokens(prompt, model='gpt-3.5-turbo-instruct-0914'):
    num, _, _ = num_tokens_from_messages(prompt, model)
    return num

def ua2webshop_run(user_idx, task_idx, instruction_history:list, prompt, profiler, logfile, to_print=False, model="gpt-3.5-turbo-instruct-0914"):
    print(f"**************USER-{user_idx} TASK-{task_idx}**************")
    
    action = 'reset'
    # if len(instruction_history)>0:
    #     init_prompt = prompt + "\nInstruction History:\n {}\n".format('\n'.join(instruction_history))
    # else:
    #     init_prompt = prompt
    init_prompt = prompt
    prompt = ''
    # ret_traj = [[action]]

    trial_w_id, trial_wo_id_list = "", []

    instruction = ""
    ret_profile = None
    profile_prompt = ""

    request_wait_time = 0.0
    profile_retrieval_time, profile_retrieval_money = 0.0, 0.0

    for i in range(15):
        url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action)
        st_pos    = all_obs.find('Observation_from_Interactive_Env[\n')
        inter_obs = all_obs[st_pos+len('Observation_from_Interactive_Env[\n'):]
        en_pos    = find_single_right_bracket(inter_obs)
        inter_obs = inter_obs[:en_pos-1] # strip the last '\n'

        # Get instruction
        if i==0:
            profile_instruction = inter_obs[:inter_obs.find('[Search]')-1] # no using user_profile
            instruction = re.findall(r'Instruction:  \n(.*)', profile_instruction)[0]
            # instruction = inter_obs.split("\n")[3]
            print("instruction is: ", instruction)
            ret_profile, cost_, time_ = profiler.retrieve(context=instruction)
            profile_retrieval_money += cost_
            profile_retrieval_time += time_
            if ret_profile != None:
                profile_prompt = get_profile_prompt(ret_profile)
                profile_prompt += get_profile_prompt(profiler.global_profile, begin='')
                print("profile prompt is: ", profile_prompt)
                print("----------------------------------")
                print("\n")

        if action.startswith('think'):
            inter_obs = 'OK.'

        inter_obs = inter_obs.replace("[Instruction History]", "")

        if to_print:
            print(f'Action: {action}\nObservation: {inter_obs}\n')

        if i:
            cost_insight = f"Action: {action}\nObservation: {inter_obs}\nTime-cost: {all_used_time}.\nMoney-cost: {all_used_money}." 
            trial_w_id += f'\n\n{i-1}. Action: {action}\nObservation: {inter_obs}'
            trial_wo_id_list.append({"action": action, 'obs': inter_obs})
            prompt += f' {action}\nObservation: {inter_obs}\n\nAction:'
        else:
            trial_w_id += f'{inter_obs}'
            prompt += f'{inter_obs}\n\nAction:'
        
        if done:
            all_used_time += (request_wait_time + profile_retrieval_time)
            all_used_money += profile_retrieval_money
            if profile_prompt:
                print("Using retrieved profile!", file=logfile)
                profile_impact_structure_insights, profile_impact_insights, reflection_impact_money_used, reflection_impact_time_used = Insight.profile_impact(profile_prompt, prompt, trial_wo_id_list)
                all_used_time += reflection_impact_time_used
                all_used_money += reflection_impact_money_used

            click_structure_insights, click_insights, reflection_click_money_used, reflection_click_time_used = Insight.click_insight(trial_w_id, trial_wo_id_list)
            all_used_time += reflection_click_time_used
            all_used_money += reflection_click_money_used
            money_, time_ = profiler.update_profile(click_structure_insights, instruction=instruction, pre_inst=ret_profile["instruction"] if ret_profile else None, reward=inter_rwd)
            all_used_money += money_
            all_used_time += time_
            return profiler, instruction, inter_rwd, all_used_time, all_used_money, click_insights #, ret_traj

        query_prompt = init_prompt + profile_prompt + prompt #prompt[-(6400-len(init_prompt)-len(profile_prompt)):]
        t_prompt = prompt
        while(num_tokens(query_prompt) > 4097-120):
            id0 = t_prompt.find('Action:')
            obs0 = t_prompt[:id0]
            t_prompt = t_prompt[id0+1:]
            id0 = t_prompt.find('Action:')
            t_prompt = obs0 + t_prompt[id0:]
            query_prompt = init_prompt + profile_prompt + t_prompt
        # print(f"###########<PROMPT-START>################\n{query_prompt}\n############<PROMPT-END>############")
        
        hyper_args   = {"stop": ["\n"], "max_tokens": 100, "temperature": 0.0}
        action       = f"ask[{model}][{query_prompt}][{json.dumps(hyper_args)}]" 
        
        # tmp_used_time  = 0.0
        # tmp_used_money = 0.0

        while True:
            url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action)
            # if to_print:
                # print(f'Action: {action}\nObservation: {all_obs}\n')
            if API_success: 
                break
            else:
                print(f"###########<PROMPT-START>################\n{query_prompt}\n############<PROMPT-END>############")
                time.sleep(1)
                request_wait_time += 1

        st_pos = all_obs.find('LLM_response[')
        action = all_obs[st_pos+len('LLM_response['):]
        en_pos = find_single_right_bracket(action)
        action = action[:en_pos]
        if action[0] == '"' or action[0] == "'":
            action = action[1:]
        if action[-1] == '"' or action[-1] == "'":
            action = action[:-1]
        action = action.strip()

        if done:
            all_used_time += (request_wait_time + profile_retrieval_time)
            all_used_money += profile_retrieval_money
            if profile_prompt:
                print("Using retrieved profile!", file=logfile)
                profile_impact_structure_insights, profile_impact_insights, reflection_impact_money_used, reflection_impact_time_used = Insight.profile_impact(profile_prompt, prompt, trial_wo_id_list)
                all_used_time += reflection_impact_time_used
                all_used_money += reflection_impact_money_used
            
            click_structure_insights, click_insights, reflection_click_money_used, reflection_click_time_used = Insight.click_insight(trial_w_id, trial_wo_id_list)
            all_used_time += reflection_click_time_used
            all_used_money += reflection_click_money_used
            money_, time_ = profiler.update_profile(click_structure_insights, instruction=instruction, pre_inst=ret_profile["instruction"] if ret_profile else None, reward=inter_rwd)
            all_used_money += money_
            all_used_time += time_
            return profiler, instruction, inter_rwd, all_used_time, all_used_money, click_insights #, ret_traj

    all_used_time += (request_wait_time + profile_retrieval_time)
    all_used_money += profile_retrieval_money

    if profile_prompt:
        print("Using retrieved profile!", file=logfile)
        profile_impact_structure_insights, profile_impact_insights, reflection_impact_money_used, reflection_impact_time_used = Insight.profile_impact(profile_prompt, prompt, trial_wo_id_list)
        all_used_time += reflection_impact_time_used
        all_used_money += reflection_impact_money_used

    click_structure_insights, click_insights, reflection_click_money_used, reflection_click_time_used = Insight.click_insight(trial_w_id, trial_wo_id_list)
    all_used_time += reflection_click_time_used
    all_used_money += reflection_click_money_used
    money_, time_ = profiler.update_profile(click_structure_insights, instruction=instruction, pre_inst=ret_profile["instruction"] if ret_profile else None, reward=inter_rwd)
    all_used_money += money_
    all_used_time += time_
    return profiler, instruction, inter_rwd, all_used_time, all_used_money, click_insights #, ret_traj

if __name__ == "__main__":
    prompt1  = ReAct_PROMPT_1shot
    to_print     = True
    L            = 0
    USER_NUM     = 10
    TASK_NUM     = 50
    file_name      = f"../runtime_logs/react_w_insight_w_profile_ua2webshopRunTimeEnv_o_history_v1d_U{USER_NUM}_T{TASK_NUM}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt"
    avg_reward   = []
    success_rate = []
    avg_time     = []
    avg_money    = []

    instr_history_window_size = 10
    # profiler = MultiRoundProfiler("profiler", calling_func=OpenAI_API_Calling)
    for i in range(USER_NUM):
        profiler = MultiRoundProfiler("profiler", calling_func=OpenAI_API_Calling)
        avg_reward.append(0.0)
        success_rate.append(0.0)
        avg_time.append(0.0)
        avg_money.append(0.0)
        instructions = []
        for j in range(TASK_NUM):
            if i == 0 and j == 0:
                logfile = open(file_name, "w")
            else:
                logfile = open(file_name, "a")

            profiler, instruction, rwd, all_used_time, all_used_money, click_insights = ua2webshop_run(i, j, instructions[-instr_history_window_size:], prompt1, profiler, logfile=logfile, to_print=to_print)
            instructions.append(instruction)
            avg_reward[i]   += rwd
            success_rate[i] += (rwd == 1.0)
            avg_time[i]     += all_used_time
            avg_money[i]    += all_used_money

            print(f"Now user_{i} task_{j}:"             , file=logfile)
            print(f"\tReward = {rwd}"                   , file=logfile)
            print(f"\tAll Used Time = {all_used_time}"  , file=logfile)
            print(f"\tAll Used Money = {all_used_money}", file=logfile)
            logfile.close()
            if (j+1) % 10 == 0:
                print(f"Now user_{i} task_{j+1}")


    with open(file_name, "a") as logfile:
        for i in range(USER_NUM):
            print(file=logfile)
            print(f"OVERALL RESULTS for user_{i}:", file=logfile)
            print(f"\tAvg Reward     = {avg_reward[i]/TASK_NUM}"       , file=logfile)
            print(f"\tSuccess Rate   = {success_rate[i]/TASK_NUM}"     , file=logfile)
            print(f"\tAvg Used Time  = {avg_time[i]/TASK_NUM}"         , file=logfile)
            print(f"\tAvg Used Money = {avg_money[i]/TASK_NUM}"        , file=logfile)
        
    with open(file_name, "a") as logfile:   
        print(file=logfile)
        print(f"OVERALL RESULTS for user_{L} ~ user_{USER_NUM-1}:", file=logfile)
        print(f"\tAvg Reward     = {sum(avg_reward)/USER_NUM/TASK_NUM}"       , file=logfile)
        print(f"\tSuccess Rate   = {sum(success_rate)/USER_NUM/TASK_NUM}"     , file=logfile)
        print(f"\tAvg Used Time  = {sum(avg_time)/USER_NUM/TASK_NUM}"         , file=logfile)
        print(f"\tAvg Used Money = {sum(avg_money)/USER_NUM/TASK_NUM}"        , file=logfile)