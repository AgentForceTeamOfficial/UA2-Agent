
from prompt_lib import INSIGHT_REDUNDANCY_PROMPT, ReAct_PROMPT_1shot, INSIGHT_PROFILE_IMPACT_PROMPT
import re, sys, copy
sys.path.append("../")
from environments.apis import OpenAI_API_Calling, num_tokens_from_messages

def num_tokens(prompt, model='gpt-3.5-turbo-instruct-0914'):
    num, _, _ = num_tokens_from_messages(prompt, model)
    return num

class Insight:
    def parse_click_insights(raw_text:str):
        def has_digits(string):
            pattern = r'\d'
            match = re.search(pattern, string)
            return match is not None
        
        #not specifised for click-action
        # left_bracket_id = raw_text.find('[') # key_actions_id, redundant_actions_id
        # right_bracket_id = raw_text.find(']')
        # left_bracket_id = raw_text.find('key_actions_id:')
        # right_bracket_id = raw_text.find('reason:')

        next_line_id0 = raw_text.find('\n')
        next_line_id1 = raw_text[next_line_id0+1:].find('\n') + 1 + next_line_id0
        
        key_id, redundant_id = 0, 1

        if "redundant_action" in raw_text[:next_line_id1]:
            key_id = 1
            redundant_id = 0

        raw_text_0, raw_text_1 = raw_text[:next_line_id1].strip() , raw_text[next_line_id1+1:].strip()
        reason_id_0, reason_id_1 = raw_text_0.find('reason:'), raw_text_1.find('reason:')
        actions_id_0, actions_id_1 = raw_text_0.find(':'), raw_text_1.find(':')
        
        ids_0_ = raw_text_0[actions_id_0:reason_id_0].strip(':][ \n').split(',')
        # if 'none' in ids_0_.lower() or 'null' in ids_0_.lower():
        ids_0 = []# [int(i) for i in ids_0]
        for i in ids_0_:
            i = i.strip()
            if i.isdigit():
                ids_0.append(int(i))
        reasoning_0 = raw_text_0[reason_id_0+len('reason:'):].strip()

        ids_1_ = raw_text_1[actions_id_1:reason_id_1].strip(':][ \n').split(',')
        ids_1 = []# ids_1 = [int(i) for i in ids_1]
        for i in ids_1_:
            i = i.strip()
            if i.isdigit():
                ids_1.append(int(i))
        reasoning_1 = raw_text_1[reason_id_1+len('reason:'):].strip()

        ids_pair = [ids_0, ids_1]
        reasoning_pair = [reasoning_0, reasoning_1]

        return ids_pair[key_id], reasoning_pair[key_id], ids_pair[redundant_id], reasoning_pair[redundant_id]

    def click_insight(trial_w_id, trial_wo_id_list, max_tokens=4097):
        def get_trial_id_prompt(obs0, trial_list):
            prompt = ""
            for idx, dat in enumerate(trial_list):
                prompt += f"{idx}. Action: {dat['action']}\nObservation: {dat['obs']}\n\n"
            prompt = obs0.strip() + "\n\n" + prompt.strip()
            return prompt

        prompt = INSIGHT_REDUNDANCY_PROMPT.format(trial_w_id)
        
        trial_wo_id_list = copy.deepcopy(trial_wo_id_list)

        ptr = -1
        id0 = trial_w_id.find('0. Action:')
        obs0 = trial_w_id[:id0]
        while num_tokens(prompt) > max_tokens-120:
            ptr = (ptr+1) % len(trial_wo_id_list)
            action, obs = trial_wo_id_list[ptr]['action'], trial_wo_id_list[ptr]['obs']
            # if 'search' in action.lower():
            #     continue
            # obs = obs[len(obs)//2:]
            if num_tokens(obs)<400:
                continue
            obs = obs[:len(obs)//2]
            id0 = obs.rfind("]")
            if id0!=-1:
                obs = obs[:id0+1]
            else:
                continue
            obs = obs + " [This part is omitted]"
            trial_wo_id_list[ptr]['obs'] = obs
            trial_w_id = get_trial_id_prompt(obs0, trial_wo_id_list)
            prompt = INSIGHT_REDUNDANCY_PROMPT.format(trial_w_id)

        money_cost = 0
        time_cost = 0
        while True:
            insight, t_money_cost, api_success, t_time_cost = OpenAI_API_Calling(prompt=prompt, model="gpt-3.5-turbo-instruct-0914")
            if api_success:
                break
            money_cost += t_money_cost
            time_cost += t_time_cost

        key_ids, key_reasoning, redundancy_ids, redundancy_reasoning = Insight.parse_click_insights(insight)
        # redundancy_actions = np.array(trial_wo_id_list)[redundancy_ids].tolist()
        # key_actions = np.array(trial_wo_id_list)[key_ids].tolist()
        redundancy_actions = []
        for i in redundancy_ids:
            if i >=0 and i < len(trial_wo_id_list):
                redundancy_actions.append(trial_wo_id_list[i])
        # redundancy_actions = [trial_wo_id_list[i] for i in redundancy_ids]
        key_actions = []
        for i in key_ids:
            if i >=0 and i < len(trial_wo_id_list):
                key_actions.append(trial_wo_id_list[i])
        # key_actions = [trial_wo_id_list[i] for i in key_ids]

        #to filter actions that are not "click"
        redundancy_click_ids = []
        redundancy_click = []
        for id,action_obs_pair in enumerate(redundancy_actions):
            if not action_obs_pair['action'].startswith("click"):
                continue
            redundancy_click_ids.append(redundancy_ids[id])
            redundancy_click.append(action_obs_pair['action'])

        key_click_ids = []
        key_click = []
        for id,action_obs_pair in enumerate(key_actions):
            if not action_obs_pair['action'].startswith("click"):
                continue
            key_click_ids.append(key_ids[id])
            key_click.append(action_obs_pair['action'])

        text_insight = f"{trial_w_id}\ninsights:\nThe clicks with the following ids may be important to finish the task:{key_click_ids}."

        return {"redundancy_actions_reasoning": redundancy_reasoning, #reasoning for redundant actions
                "redundancy_actions": redundancy_actions, #list of [{'action':"", 'obs':""},...]
                "redundancy_actions_id": redundancy_ids, #list of int
                "redundancy_click": redundancy_click, #list of ["action"]
                "redundancy_click_id": redundancy_click_ids, #list of int
                "key_actions_reasoning": key_reasoning, #reasoning for redundant actions
                "key_actions": key_actions, #list of [{'action':"", 'obs':""},...]
                "key_actions_id": key_ids, #list of int
                "key_click": key_click, #list of ["action"]
                "key_click_id": key_click_ids, #list of int
                }, text_insight, money_cost, time_cost

    def profile_impact(profile_prompt: str, trajectory: str, trial_list, max_tokens=4097):
        """
        return example:
            <insight>: 
            Impact on Speed: Positive
            Reason: Having an instruction from a previous task that is similar to the current one can save time and speed up the process as the user is already familiar with the steps and knows what to look for.

            Impact on Effectiveness: Positive
            Reason: The user's familiarity with a similar task can increase their confidence and efficiency in completing the current task, making it more effective.
            <structered insight>:
            {"impact_on_speed": "Impact on Speed: Positive",
            "reason_on_speed": "Having an instruction from a previous task that is similar to the current one can save time and speed up the process as the user is already familiar with the steps and knows what to look for.",
            ...,}
            <money_cost>, <time_cost>
        """
        def get_trajectory_prompt(obs0, trial_list):
            prompt = obs0.strip() + '\n\n'
            for i in trial_list:
                prompt += f"Action: {i['action']}\nObservation: {i['obs']}\n\n"
            return prompt.strip()

        trajectory = trajectory.strip().strip('Action:').strip()

        prompt = INSIGHT_PROFILE_IMPACT_PROMPT.format(trajectory, profile_prompt)

        trial_list = copy.deepcopy(trial_list)
        ptr = -1
        id0 = trajectory.find('Action:')
        obs0 = trajectory[:id0].strip()
        while num_tokens(prompt) > max_tokens-120:
            ptr = (ptr+1) % len(trial_list)
            action, obs = trial_list[ptr]['action'], trial_list[ptr]['obs']
            # if 'search' in action.lower():
            #     continue
            if num_tokens(obs)<400:
                continue
            obs = obs[:len(obs)//2]
            id0 = obs.rfind("]")
            if id0!=-1:
                obs = obs[:id0+1]
            else:
                continue
            obs = obs + " [This part is omitted]"
            trial_list[ptr]['obs'] = obs

            trajectory = get_trajectory_prompt(obs0, trial_list)
            prompt = INSIGHT_REDUNDANCY_PROMPT.format(trajectory) 

        money_cost = 0
        time_cost = 0
        while True:
            insight, t_money_cost, api_success, t_time_cost = OpenAI_API_Calling(prompt=prompt, model="gpt-3.5-turbo-instruct-0914")
            if api_success:
                break
            money_cost += t_money_cost
            time_cost += t_time_cost

        id0 = insight.find("Impact on Speed")
        id1 = insight.find("Impact on Effectiveness")
        # assert id0<id1 and id0!=-1 and id1!=-1
        if not (id0<id1 and id0!=-1 and id1!=-1):
            return {
                "impact_on_speed": "none",
                "reason_on_speed": "none",
                "impact_on_effectiveness": "none",
                "reason_on_effectiveness": "none"
            }, insight, money_cost, time_cost

        impact_on_speed = insight[:id1].strip()
        impact_on_effectiveness = insight[id1:].strip()

        id0 = impact_on_speed.lower().find("reason")
        reason_on_speed = impact_on_speed[id0+len("reason:"):].strip()
        impact_on_speed = impact_on_speed[:id0].strip()

        id1 = impact_on_effectiveness.lower().find("reason")
        reason_on_effectiveness = impact_on_effectiveness[id1+len("reason:"):].strip()
        impact_on_effectiveness = impact_on_effectiveness[:id1].strip()
        return {
                "impact_on_speed": impact_on_speed,
                "reason_on_speed": reason_on_speed,
                "impact_on_effectiveness": impact_on_effectiveness,
                "reason_on_effectiveness": reason_on_effectiveness
            }, insight, money_cost, time_cost


if __name__ == "__main__":
    Insight.profile_impact("#","#")