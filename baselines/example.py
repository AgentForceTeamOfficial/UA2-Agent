from datetime import datetime
import sys

sys.path.append(".")
from environments.env_instr_list_cwebshop_runtime_session import cwebshopRunTimeEnv

env = cwebshopRunTimeEnv(init_money=10000, init_time=10000)

def find_single_right_bracket(str0):
    """
    Find the position of the first single right bracket in str0.
    """
    bracket_cnt = 1
    final_pos = 0
    for i in range(len(str0)):
        if str0[i] == "[":
            bracket_cnt += 1
        if str0[i] == "]":
            bracket_cnt -= 1
            final_pos = i
        if bracket_cnt == 0:
            return i
    return final_pos

def get_next_action(inter_obs):
    if "[Search]" in inter_obs:
        return "search[Hello C-Webshop!]"
    else:
        return "click[Back to Search]"

def cwebshop_run(
    user_idx,
    task_idx,
    prompt,
    instr_history,
    to_print=False,
    model="gpt-3.5-turbo-instruct-0914"
):
    action = "reset"
    init_prompt = prompt
    prompt = ""

    for i in range(15):  # max 15 actions
        # [*][Step1 Begin] Execute action
        url, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(
            user_idx, task_idx, action
        )

        if done:
            return inter_rwd, all_used_time, all_used_money

        st_pos = all_obs.find("Observation_from_Interactive_Env[\n")
        inter_obs = all_obs[st_pos + len("Observation_from_Interactive_Env[\n") :]
        en_pos = find_single_right_bracket(inter_obs)
        inter_obs = inter_obs[: en_pos - 1]
        # [*][Step1 End]

        # [*][Step2 Begin] Get the next action from the LLM
        ## Input: inter_obs
        ## Output: action (at next step)
        action = get_next_action(inter_obs)
        # [*][Step2 End]

    return 0, all_used_time, all_used_money


if __name__ == "__main__":
    init_prompt = """You prompt here."""
    to_print = False
    L = 0
    USER_NUM = 10
    TASK_NUM = 50
    file_name = f"./runtime_logs/cwebshopRunTimeEnvSession_L{L}_USER{USER_NUM}_TASK{TASK_NUM}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt"  # log file for results
    avg_reward = []
    success_rate = []
    avg_time = []
    avg_money = []

    for i in range(USER_NUM):
        avg_reward.append(0.0)
        success_rate.append(0.0)
        avg_time.append(0.0)
        avg_money.append(0.0)
        instr_history = []
        for j in range(TASK_NUM):
            print(f"Now user_{i} task_{j}")
            rwd, all_used_time, all_used_money = cwebshop_run(
                i,
                j,
                init_prompt,
                instr_history,
                to_print=to_print,
            )
            avg_reward[i] += rwd
            success_rate[i] += rwd == 1.0
            avg_time[i] += all_used_time
            avg_money[i] += all_used_money

            if i == 0 and j == 0:
                logfile = open(file_name, "w")
            else:
                logfile = open(file_name, "a")
            print(f"Now user_{i} task_{j}:", file=logfile)
            print(f"\tReward = {rwd}", file=logfile)
            print(f"\tAll Used Time = {all_used_time}", file=logfile)
            print(f"\tAll Used Money = {all_used_money}", file=logfile)
            logfile.close()
            if (j + 1) % 10 == 0:
                print(f"Now user_{i} task_{j+1}")

    with open(file_name, "a") as logfile:
        for i in range(USER_NUM):
            print(file=logfile)
            print(f"OVERALL RESULTS for user_{i}:", file=logfile)
            print(f"\tAvg Reward     = {avg_reward[i]/TASK_NUM}", file=logfile)
            print(f"\tSuccess Rate   = {success_rate[i]/TASK_NUM}", file=logfile)
            print(f"\tAvg Used Time  = {avg_time[i]/TASK_NUM}", file=logfile)
            print(f"\tAvg Used Money = {avg_money[i]/TASK_NUM}", file=logfile)

    with open(file_name, "a") as logfile:
        print(file=logfile)
        print(f"OVERALL RESULTS for user_{L} ~ user_{USER_NUM-1}:", file=logfile)
        print(f"\tAvg Reward     = {sum(avg_reward)/USER_NUM/TASK_NUM}", file=logfile)
        print(f"\tSuccess Rate   = {sum(success_rate)/USER_NUM/TASK_NUM}", file=logfile)
        print(f"\tAvg Used Time  = {sum(avg_time)/USER_NUM/TASK_NUM}", file=logfile)
        print(f"\tAvg Used Money = {sum(avg_money)/USER_NUM/TASK_NUM}", file=logfile)
