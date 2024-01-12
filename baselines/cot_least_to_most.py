import re
import json
import time
from datetime import datetime
import sys
sys.path.append("../")
from environments.env_instr_list_cwebshop_runtime_session import cwebshopRunTimeEnv
from environments.apis import num_tokens_from_messages

env = cwebshopRunTimeEnv(init_money=10000, init_time=10000)

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

def num_tokens(prompt, model='gpt-3.5-turbo-instruct-0914'):
    """
    Get the number of tokens in the prompt.
    """
    num, _, _ = num_tokens_from_messages(prompt, model)
    return num

def cwebshop_run(user_idx, task_idx, prompt, instr_history, to_print=False, model="gpt-3.5-turbo-instruct-0914"):
    action = 'reset'
    init_prompt = prompt
    prompt = ''

    invalid_cnt = 0     # count the number of consecutive invalid actions

    init_message = ""

    task = ""
    subtasks = []
    actions = []
    observations = []

    for i in range(15): # max 15 actions
        # [*][Step1 Begin] Execute action
        url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action)

        if to_print:
            print('='*50)
            print(f'Action: {action}')
            print(f'all_obs: {all_obs}')
            print(f'done: {done}')
            print(f'API_success: {API_success}')
            print(f'inter_rwd: {inter_rwd}')
            print(f'all_used_time: {all_used_time}')
            print(f'all_used_money: {all_used_money}')

        if done:
            return inter_rwd, all_used_time, all_used_money
        
        st_pos    = all_obs.find('Observation_from_Interactive_Env[\n')
        inter_obs = all_obs[st_pos+len('Observation_from_Interactive_Env[\n'):]
        en_pos    = find_single_right_bracket(inter_obs)
        inter_obs = inter_obs[:en_pos-1]

        if "Invalid Action" in inter_obs:
            invalid_cnt += 1
            if invalid_cnt >= 3:
                return 0, all_used_time, all_used_money
        else:
            invalid_cnt = 0
        # [*][Step1 End]


        # [*][Step2 Begin] Get the next action from the LLM
        # [*][Step2.1 Begin] Get the prompt for the LLM
        if action.startswith('think'):
            inter_obs = 'OK.'

        if i == 0 or action == "click[Back to Search]":
            inter_obs = inter_obs[:inter_obs.find('[Search]')-1]
            try:
                now_instr = re.findall(r'Instruction:  \n(.*)', inter_obs)[0]
                task = now_instr
            except Exception as e:
                if to_print:
                    print(e)
                    print(inter_obs)
                return 0, all_used_time, all_used_money
            if len(instr_history) > 0:
                inter_obs += "\nInstruction History:"
                for idx, instr in enumerate(instr_history):
                    if idx + 10 >= len(instr_history):
                        inter_obs += f"\n{idx+1}. {instr}"
            if i == 0:
                instr_history.append(now_instr)
                init_message = inter_obs
            inter_obs += "\n[Search]"
            if i != 0:
                observations.append(inter_obs)
                actions.append(action)
        elif "Invalid Action" in inter_obs:
            actions.append(action)
            actions[-1] += " (Invalid Action)"
        else:
            inter_obs = f"Observation: {inter_obs}"
            observations.append(inter_obs)
            actions.append(action)

        prompt = init_message
        min_idx = len(actions)
        if len(observations) > 0:
            prompt += f"\n{observations[-1]}"
        while min_idx > 0 and num_tokens(prompt + f"\n\nSubTask: {subtasks[min_idx-1]}\nAction: {actions[min_idx-1]}" + f"\n\nYou should think step by step and choose the next action. Action format: tool[content]. Only item between 'brackets' in observation can be clicked.\nIf you can't finish the task in one step, you should decompose it and response next task.\nTask: {task}\nThought:" + init_prompt) < 3900:
            prompt += f"\n\nSubTask: {subtasks[min_idx-1]}\nAction: {actions[min_idx-1]}"
            min_idx -= 1
        prompt = init_message
        if min_idx > 0:
            prompt += f"\n\n..."
        for idx in range(min_idx, len(actions)):
            prompt += f"\n\nSubTask: {subtasks[idx]}\nAction: {actions[idx]}"
        if len(observations) > 0:
            prompt += f"\n\n{observations[-1]}"

        prompt_tag = prompt

        prompt += f"\n\nYou should think step by step and choose the next action. Action format: tool[content]. Only item between 'brackets' in observation can be clicked.\nIf you can't finish the task in one step, you should decompose it and response next task.\nTask: {task}\nThought:"
        # [*][Step2.1 End]

        # [*][Step2.2 Begin] Get the action from the LLM
        output_format_cnt = 0
        for idx in range(6):
            if to_print:
                print(f"**********\nPrompt: {prompt}\n**********")
            query_prompt = init_prompt + prompt

            all_used_time  = 0.0
            all_used_money = 0.0

            hyper_args   = {"stop": ["\n"], "max_tokens": 100, "temperature": 0.0}
            action       = f"ask[{model}][{query_prompt}][{json.dumps(hyper_args)}]" 
            fail_cnt = 0
            while True:
                url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action)
                if to_print and not API_success:
                    print(f'Action: {action}\nObservation: {all_obs}\n')
                if API_success: 
                    break
                else:
                    # time.sleep(1)
                    fail_cnt += 1
                    if fail_cnt >= 5:
                        return 0, all_used_time, all_used_money

            st_pos = all_obs.find('LLM_response[')
            action = all_obs[st_pos+len('LLM_response['):]
            en_pos = find_single_right_bracket(action)
            action = action[:en_pos]

            if "action:" in action:
                st_pos = action.find('action:')
                if st_pos != -1:
                    action = action[st_pos+len('action:'):]
                if '"' in action:
                    action = action.split('"')[0]
                action = action[:action.find(']') + 1]
                action = action.strip()
                if idx == 0:
                    subtasks.append(task)
                if done:
                    return inter_rwd, all_used_time, all_used_money
                break
            elif "task:" in action:
                st_pos = action.find('task:')
                if st_pos != -1:
                    action = action[st_pos+len('task:'):]
                if len(subtasks) == len(actions):
                    subtasks.append(action)
                else:
                    subtasks[-1] = action
                prompt = prompt_tag + f"\n\nYou should think step by step and choose the next action. Action format: tool[content]. Only item between 'brackets' in observation can be clicked.\nIf you can't finish the task in one step, you should decompose it and response next task.\nSubTask: {action}\nThought:"
            else:
                output_format_cnt += 1
                if output_format_cnt >= 3:
                    return 0, all_used_time, all_used_money
        # [*][Step2.2 End]
        # [*][Step2 End]

    return 0, all_used_time, all_used_money#, ret_traj

def test_cot_least_to_most():
    init_prompt  = """C-Webshop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars.
[Search]

Task: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars.
Thought: For buying 3 ounce bottle of bright citrus deodorant for sensitive skin. Next task: Search for 3 ounce bottle of bright citrus deodorant for sensitive skin.

SubTask: Search for 3 ounce bottle of bright citrus deodorant for sensitive skin.
Thought: I can search it by using the search bar. Next action: search[3 ounce bright citrus deodorant sensitive skin]

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

Task: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars.
Thought: There are 3 results in the first page. I should firstly click the best one for user. Next task: Click the best item for user.

SubTask: Click the best item for user.
Thought: B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first. Next action: click[B078GWRC1J]

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

Task: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars.
Thought: For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy. Next task: select 'bright citrus' first.

SubTask: select 'bright citrus' first.
Thought: I can click 'bright citrus' to select it. Next action: click[bright citrus]

Observation:
[Back to Search]
[< Prev]
scent [assorted scents][bright citrus](have clicked) [calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
Price: $10.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Task: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars.
Thought: I have clicked bright citrus, I should select '3 ounce (pack of 1)' now. Next task: select '3 ounce (pack of 1)'.

SubTask: select '3 ounce (pack of 1)'.
Thought: I can click '3 ounce (pack of 1)' to select it. Next action: click[3 ounce (pack of 1)]

Observation:
[Back to Search]
[< Prev]
scent [assorted scents][bright citrus](have clicked) [calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)](have clicked) [3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
Price: $10.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Task: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars.
Thought: I have selected 'bright citrus' and '3 ounce (pack of 1)'. I can buy it now. Next action: click[Buy Now]
"""
    to_print     = False
    L            = 0
    USER_NUM     = 10
    TASK_NUM     = 50  
    file_name    = f"../runtime_logs/cot_least_to_most_cwebshopRunTimeEnvSession_L{L}_USER{USER_NUM}_TASK{TASK_NUM}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt"               # log file for results
    avg_reward   = []
    success_rate = []
    avg_time     = []
    avg_money    = []

    for i in range(USER_NUM):
        avg_reward.append(0.0)
        success_rate.append(0.0)
        avg_time.append(0.0)
        avg_money.append(0.0)
        instr_history = []
        for j in range(TASK_NUM):
            print(f"Now user_{i} task_{j}")
            rwd, all_used_time, all_used_money = cwebshop_run(i, j, init_prompt, instr_history, to_print=to_print)
            avg_reward[i]   += rwd
            success_rate[i] += (rwd == 1.0)
            avg_time[i]     += all_used_time
            avg_money[i]    += all_used_money

            if i == 0 and j == 0:
                logfile = open(file_name, "w")
            else:
                logfile = open(file_name, "a")
            print(f"Now user_{i} task_{j}:"                     , file=logfile)
            print(f"\tReward = {rwd}"                           , file=logfile)
            print(f"\tAll Used Time = {all_used_time}"          , file=logfile)
            print(f"\tAll Used Money = {all_used_money}"        , file=logfile)
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

if __name__ == "__main__":
    test_cot_least_to_most()
