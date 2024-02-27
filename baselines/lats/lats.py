import logging
import copy
import sys
sys.path.append('../../')
from environments.apis import num_tokens_from_messages
from environments.env_instr_list_cwebshop_runtime_session import cwebshopRunTimeEnv, WEBSHOP_URL

import argparse
from datetime import datetime
import numpy as np
import json
import requests
from requests.auth import HTTPBasicAuth
import copy
import time
from lats_prompts import *
import re, random, ast, os

env = cwebshopRunTimeEnv(init_money=10000, init_time=10000)
global reflection_map
global failed_trajectories
global all_used_money
global all_used_time
global instr_history
reflection_map = []
failed_trajectories = []
all_used_money = 0
all_used_time = 0
instr_history = []


def num_tokens(prompt, model='gpt-3.5-turbo-instruct-0914'):
    num, _, _ = num_tokens_from_messages(prompt, model)
    return num

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


def env_gpt(query_prompt, user_index, task_idx, n=1, to_print=False, stop=None, **kwargs):
    global all_used_money
    global all_used_time
    query_prompt = query_prompt.replace("[Instruction History]", "")
    hyper_args   = {"stop": stop, "max_tokens": 100, "temperature": 1.0,}
    hyper_args.update(kwargs)
    
    if 'model' in hyper_args:
        model = hyper_args['model']
        hyper_args.pop('model')
    else:
        model = 'gpt-3.5-turbo-1106'
    if model == 'gpt-3.5-turbo-instruct-0914':
        messages = query_prompt
    else:
        messages = json.dumps([{"role": "user", "content": query_prompt}])
    

    outputs = []
    while n >0:
        cnt = min(n, 20)
        n -= cnt
        hyper_args.update({"n": cnt})
        action   = f"ask[{model}][{messages}][{json.dumps(hyper_args)}]" 
        while True:
            
            _, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_index, task_idx, action.strip())
            logging.info(f"all_used_money: {all_used_money}")
            logging.info(f"all_used_time: {all_used_time}" )
            if to_print:
                print(f'Action: {action}\nObservation: {all_obs}\n')
            if API_success: 
                break

        st_pos = all_obs.find('LLM_response[')
        action = all_obs[st_pos+len('LLM_response['):]
        en_pos = find_single_right_bracket(action)
        action = action[:en_pos]
        action = action.strip()
        if action.startswith('['):
            actions = action.split('",')
            actions = [i.strip(' \n["') for i in actions]
            actions = [ i if i.endswith(']') else i+']' for i in actions]
            outputs.extend(actions)
        else:
            outputs.append(action)
    if len(outputs) == 0:
        print(len(outputs))
    return outputs


class WebShopTask():
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        self.steps = 7
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}
        self.reflections = []
    
    @staticmethod
    def generate_self_reflection(traj, question, user_idx, task_idx, ):
        
        reflect_prompt = reflection_prompt.format(trajectory=traj)
        
        reflection = env_gpt(reflect_prompt, user_index=user_idx, task_idx=task_idx, n=1, stop=None)
        
        traj_with_reflection = traj + "Reflection: " + reflection[0] + "\n"
        
        reflection_mapping = {
            'question': question,
            'reflection': reflection[0]
        }

        return traj_with_reflection, reflection_mapping

    @staticmethod
    def generate_self_reflection(z, question, user_idx, task_idx, ):
        reflection_mapping = []
        trajectories = ""

        sampled_items = random.sample(z, min(3, len(z)))
        failed_trajectories = [item['trajectory'] + f"\nReward: {item['r']}\n" for item in sampled_items if isinstance(item, dict) and 'trajectory' in item and 'r' in item]
        
        for traj in failed_trajectories:
            trajectories += traj
            reflect_prompt = reflection_prompt.format(trajectory=traj)
           
            reflection = env_gpt(reflect_prompt, user_index=user_idx, task_idx=task_idx, n=1, stop=None)
            
            trajectories += "Reflection: " + reflection[0] + "\n"
            
            reflection_mapping.append({
                'question': question,
                'trajectory': traj,
                'reflection': reflection[0]
            })

        return reflection_mapping
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        input = x + '\n' + y
        return prompt1.format(input=input)
 
    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '', reflection_mapping_list=[]):
        question = x
        input = x + '\n' + y
        trajectories = ""
        
        if reflection_mapping_list:
            for reflection_mapping in reflection_mapping_list:
                traj_with_reflection = reflection_mapping['trajectory'] + "Reflection: " + reflection_mapping['reflection'] + "\n"
                trajectories += traj_with_reflection
            
            prompt = prompt1_feedback.format(trajectories=trajectories, input=input)
            return prompt
        else:
            return prompt1.format(input=input)



        
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = score_prompt + "\n" + x + "\n\n"
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best trajectory is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results
    
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str, z: list = [], reflections: list = []) -> str:
        question = x.split('\n')[0]
        if len(z) != 0:
            failed_trajectories = ""
            for traj, ref in zip(z, reflections):
                score = int(traj['r'] * 10) / 2
                trajectory = traj['trajectory']
                split_trajectory = trajectory.split('Action: ')
                first_part = split_trajectory[0]  # This part will not be modified

                # Remove the first 'Action' and corresponding 'Observation'
                remaining_parts = split_trajectory[2:]

                # Reconstruct the trajectory string
                new_trajectory = 'Action: '.join([first_part] + remaining_parts)
                traj['trajectory'] = new_trajectory
                failed_trajectories += f"{y}\n{traj}\nReflection: {ref['reflection']}\nThus the correctness score is {score}\n"
            
            inp = y
            prompt = score_prompt_feedback.format(trajectories=failed_trajectories, input=inp) + "\n\nReflection: "
        else:
            inp = y 
            prompt = score_prompt.format(trajectories="", input=inp)+ "\n\nReflection: "
            
        return prompt

    
    @staticmethod
    def value_outputs_unwrap(evaluate_prompt):
        evaluate_prompt = evaluate_prompt[0]
        try:
            evaluate_prompt = json.loads(evaluate_prompt)
            score = json.loads(evaluate_prompt)['score']
        except:
            return -1
        # if score not in range(1,11):
        if isinstance(score, (int, float)):
            return score / 10
        else:
            return -1
            # return score / 10


class Node:
    def __init__(self, state, question, env_state=None, parent=None):
        self.state = {'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric
        self.env_state = env_state

    def uct(self):
        if self.visits == 0 and self.value >= 0:
            return float('inf')
            #return self.value * 2
        elif self.visits == 0 and self.value < 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def uct_with_depth(self, C1=1, C2=1):
        if self.visits == 0:
            return self.value
        exploitation_term = self.value / self.visits
        exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        depth_term = self.depth
        return exploitation_term + C1 * exploration_term + C2 * depth_term

    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }


def select_node(node):
    while node and node.children:
        logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]
        
        if len(terminal_children) == len(node.children):
            logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  
        
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logging.info(f"Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1
        
        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

        while node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
        logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
        
    return node 

def expand_node(node, args, task, user_idx, task_idx ):
    n = args.n_generate_sample
    if node.depth >= 15:
        logging.info("Depth limit reached")
        return
    if node.depth == 0:
        n *= 2
    new_nodes = generate_new_states(node, args, task, user_idx, task_idx, n)
    node.children.extend(new_nodes)

def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['action']:
            new_segment.append(f"Action: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n\n'.join(reversed(trajectory))

def get_value(task, x, y, n_evaluate_sample, user_idx, task_idx, cache_value=True):
    global reflection_map
    global failed_trajectories
    value_prompt = task.value_prompt_wrap(x, y, failed_trajectories, reflection_map)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = env_gpt(value_prompt, user_index=user_idx, task_idx=task_idx, n=n_evaluate_sample, to_print=False, stop=None, response_format={ "type": "json_object" }, model='gpt-3.5-turbo-1106')    
    logging.info(f"VALUE OUTPUTS: {value_outputs}")
    value = task.value_outputs_unwrap(value_outputs)
    logging.info(f"VALUES: {value}")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, user_idx, task_idx, cache_value=False):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, user_idx, task_idx, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, user_idx,task_idx):
    global reflection_map
    global failed_trajectories
    if len(failed_trajectories) > len(reflection_map) and len(failed_trajectories) < 4:
        # print("generating reflections")
        # print(len(failed_trajectories))
        # print(len(reflection_map))
        reflection_map = task.generate_self_reflection(failed_trajectories, x, task_idx=task_idx, user_idx=user_idx)
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, reflection_map)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    # logging.info(f"PROMPT: {prompt}")

    samples = env_gpt(prompt, user_index=user_idx, task_idx=task_idx, n=n_generate_sample, stop=stop, temperature=1.0)
    return [y + _ for _ in samples]


def generate_new_states(node, args, task, user_idx, task_idx, n):
    global all_used_time
    global all_used_money
    global failed_trajectories
    prompt = generate_prompt(node)
    #print(prompt)
    sampled_actions = get_samples(task, prompt, "\nAction: ", n, prompt_sample=args.prompt_sample, stop=["Observation", '\n'], user_idx=user_idx, task_idx=task_idx)
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    unique_states = {}  # Store unique states here
    added = False
    for action in sampled_actions:
        local_sessions = copy.deepcopy(node.env_state)
        local_sessions[user_idx][task_idx]["rest_money"] = env.sessions[user_idx][task_idx]["rest_money"]
        local_sessions[user_idx][task_idx]["rest_time"] = env.sessions[user_idx][task_idx]["rest_time"]
        env.sessions = local_sessions
        logging.info(env.sessions)
        new_state = node.state.copy()  # Make a copy of the parent node's state
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)
        action_line = action_line.replace("\\n", "").strip()
        if action_line=="":
            continue
        if not action_line[-1]==']':
            action_line += ']'
        if action_line != "reset" and (not action_line.startswith("think[")) and (not action_line.startswith("search[")) and (not action_line.startswith("click[")):
            action_line = f"think[{action_line}]"
        

        # Use thought and action to form a unique key
        unique_key = f"{action_line}"
        
        if action_line:
            try:
                _, all_obs, done, API_success, inter_rwd, all_used_time, all_used_money = env.step(user_idx, task_idx, action_line.strip())
                st_pos    = all_obs.find('Observation_from_Interactive_Env[\n')
                inter_obs = all_obs[st_pos+len('Observation_from_Interactive_Env[\n'):]
                en_pos    = find_single_right_bracket(inter_obs)
                inter_obs = inter_obs[:en_pos-1] # strip the last '\n'

                inter_obs = inter_obs.replace("[Instruction History]", "").strip()
                
                obs = inter_obs
                r = inter_rwd
                logging.info(f"all_used_money: {all_used_money}")
                logging.info(f"all_used_time: {all_used_time}" )

            except AssertionError:
                obs = 'Invalid action!'
                # print("err")
                r = -1
                done = False
            
            if action_line.startswith('think'):
                obs = 'OK.'
      
            # Update the new state dictionary
            new_state['action'] = action_line
            new_state['observation'] = obs
            
            env_state_clone = copy.deepcopy(env.sessions) # Clone current environment state
            new_node = Node(state=new_state, question=node.question, env_state=env_state_clone, parent=node)
            new_node.env_state = local_sessions
            if r > 0 or done:
                logging.info(f"reward:{r}")
                new_node.is_terminal = True
                #print("rew", r)
            new_node.reward = r
            new_node.value = r
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")

            if new_node.is_terminal and r < 1.0 and r > 0.0 and added == False:
                trajectory = collect_trajectory(new_node)

                # Check if there is already a failed trajectory with the same reward
                existing_rewards = [t['r'] for t in failed_trajectories]

                if r not in existing_rewards:
                    print("adding to failed")
                    added = True
                    failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_line}", 'r': r})

    return list(unique_states.values())  # Return unique nodes as a list

def collect_trajectory(node):
    trajectory = []
    #print("collecting traj", node)
    question = node.question
    # Append the question from the root node
    # trajectory.append(node.question)
    
    # Collect action and observation from each node till the root
    while node:
        new_segment = []
        if node.state and 'action' in node.state and node.state['action'] and node.parent:
            new_segment.append(f"Action: {node.state['action']}")
        else:
            logging.warning(f"Missing or empty action in node at depth {node.depth}")
            
        if node.state and 'observation' in node.state and node.state['observation'] and node.parent:
            new_segment.append(f"Observation: {node.state['observation']}")
        else:
            logging.warning(f"Missing or empty observation in node at depth {node.depth}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n\n'.join(reversed(trajectory))



def evaluate_node(node, args, task, user_idx, task_idx):
    
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]

    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample, user_idx, task_idx)
    
    logging.info(f"Length of votes: {len(votes)}")
    logging.info(f"Length of node.children: {len(node.children)}")
    
    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    
    max_vote = max(votes) if votes else 1
    if max_vote == 0:
        max_vote = 1  # Avoid division by zero
    
    terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
    for i, condition in enumerate(terminal_conditions):
        if condition == 1:
            votes[i] = max_vote + 1
    
    for i, child in enumerate(node.children):
        child.value = votes[i] / max_vote  # Now safe from division by zero
    
    return sum(votes) / len(votes) if votes else 0

def rollout(node, args, task, user_idx, task_idx, max_depth=15, ):
    depth = 0
    n = 5
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, user_idx=user_idx, task_idx=task_idx, n=n)

        for state in new_states:
            if state.is_terminal:
                return state
                
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        #new_state = new_state[0]
        while len(values) == 0:
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample, user_idx, task_idx)
        
        max_value_index = values.index(max(values))
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            node.reward = -0.5
    return node  


def backpropagate(node, value):
    while node:
        node.visits += 1
        node.value = (node.value * (node.visits - 1) + value) / node.visits
        logging.info(f"Backpropagating with reward {value} at depth {node.depth}. New value: {node.value}.")
        node = node.parent


def collect_all_nodes(node):
    """Recursively collect all nodes starting from the given node."""
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_all_nodes(child))
    return nodes



def lats_search(args, task, user_idx, task_idx, iterations=15, to_print=True):
    print(f"**************USER-{user_idx} TASK-{task_idx}**************")
    global env_gpt
    global failed_trajectories
    global reflection_map
    global all_used_time
    global all_used_money
    global instr_history
    all_used_money, all_used_time = 0, 0
    action = 'reset'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log),
            logging.StreamHandler()
        ]
    )
    _, all_obs, done, API_success, inter_rwd, now_used_time, now_used_money = env.step(user_idx, task_idx, action)
    st_pos    = all_obs.find('Observation_from_Interactive_Env[\n')
    inter_obs = all_obs[st_pos+len('Observation_from_Interactive_Env[\n'):]
    en_pos    = find_single_right_bracket(inter_obs)
    inter_obs = inter_obs[:en_pos-1] #
    inter_obs = inter_obs.replace("[Instruction History]", "").strip()
    
    x = inter_obs

    if to_print:
        print(task_idx, x)
    root = Node(state=None, question=x)
    root.env_state = copy.deepcopy(env.sessions)
    #print("ROOTSTATE", root.env_state)
    all_nodes = []
    failed_trajectories = []
    reflection_map = []
    terminal_nodes = []

    for i in range(iterations):
        logging.info(f"Iteration {i + 1}...")
        node = select_node(root)

        while node is None or (node.is_terminal and node.reward != 1):
            logging.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
            node = select_node(root)
        
        if node is None:
            logging.info("All paths lead to terminal nodes with reward 0. Ending search.")
            break

        if node.is_terminal and node.reward == 1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            return node.state, node.value, all_nodes, node.reward, node.em
        
        expand_node(node, args, task, user_idx, task_idx)

        while node.is_terminal:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root)
            expand_node(node, args, task, user_idx, task_idx )

        val = evaluate_node(node, args, task, user_idx, task_idx )
        # Simulation or rollout
        terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, task_idx=task_idx, user_idx=user_idx, max_depth=args.max_depth)
        terminal_nodes.append(terminal_node)

        if terminal_node.reward == 1:
            logging.info("Successful trajectory found")
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            return terminal_node.state, terminal_node.value, terminal_node.reward, terminal_node.em
        # Backpropagate reward
        backpropagate(terminal_node, terminal_node.reward)
        
        all_nodes = [(node, node.reward) for node in collect_all_nodes(root)]
        print("searching all nodes...")
        # Check for terminal nodes with a reward of 1
        terminal_nodes_with_reward_1 = [node for node, reward in all_nodes if node.is_terminal and node.reward == 1]

        if terminal_nodes_with_reward_1:
            logging.info("Successful trajectory found")
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.reward)
            return best_node.state, best_node.value, best_node.reward, best_node.em
    
        for j, (node, value) in enumerate(all_nodes):
            logging.info(f"Node {j+1}: {str(node)}")

        node_strings = '\n'.join(str(node[0]) for node in all_nodes)
        logging.info(f"State of all_nodes after iteration {i + 1}:\n{node_strings}")

    #best_child = max(root.children, key=lambda x: x.reward)
    all_nodes_list = collect_all_nodes(root)
    all_nodes_list.extend(terminal_nodes)
    best_child = max(all_nodes_list, key=lambda x: x.reward)
    failed_trajectories = []
    print("best value found", best_child.reward)
    if best_child.reward == 1:
        logging.info("Successful trajectory found")
    else:
        logging.info("Unsuccessful/Partially Successful trajectory found")
    return best_child.state, best_child.value, best_child.reward, best_child.em


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'], default='cot')  
    args.add_argument('--n_generate_sample', type=int, default=5)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=30)
    args.add_argument('--max_depth', type=int, default=15)
    args.add_argument('--log', type=str, default=f'./lats.test.{str(datetime.now()).replace(":","_")}.runtimelogs')
    args.add_argument('--save_folder', type=str, default='runtime_logs_0115')
    args.add_argument('--start_task', type=int, default=0)
    args.add_argument('--task_num', type=int, default=10)

    args = args.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    L            = 0
    USER_NUM     = 3
    TASK_NUM     = args.task_num 
    PORT = WEBSHOP_URL.split(':')[-1]
    print(PORT)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    args.log = f'./{args.save_folder}/STARTTASK{args.start_task}_PORT{PORT}_lats.test.{str(datetime.now()).replace(":","_")}.runtimelogs'
    file_name    = f"./{args.save_folder}/lats_cwebshopRunTimeEnvSession_L{L}_USER{USER_NUM}_STARTTASK{args.start_task}_PORT{PORT}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt"
    task = WebShopTask()
    # print(task)
    count = 0
    task_accs = []
    n = TASK_NUM

    avg_reward   = 0
    success_rate = 0
    avg_time     = 0
    avg_money    = 0

    # for i in range(TASK_NUM):
    for i in range(args.start_task, args.start_task+args.task_num):

        reflection_map = []
        failed_trajectories = []
        all_used_money = 0
        all_used_time = 0
        # solve
        state, value, reward, em = lats_search(args, task, USER_NUM, i, args.iterations, True,)
        avg_reward   += reward
        success_rate += (reward == 1.0)
        avg_time     += all_used_time
        avg_money    += all_used_money

        if i==0:
            logfile = open(file_name, "w")
        else:
            logfile = open(file_name, "a")
        print(f"Now user_{USER_NUM} task_{i}:"                     , file=logfile)
        print(f"\tReward = {reward}"                           , file=logfile)
        print(f"\tAll Used Time = {all_used_time}"          , file=logfile)
        print(f"\tAll Used Money = {all_used_money}"        , file=logfile)
        logfile.close()
    
    with open(file_name, "a") as logfile:
        print(file=logfile)
        print(f"OVERALL RESULTS for user_{USER_NUM}:", file=logfile)
        print(f"\tAvg Reward     = {avg_reward/TASK_NUM}"       , file=logfile)
        print(f"\tSuccess Rate   = {success_rate/TASK_NUM}"     , file=logfile)
        print(f"\tAvg Used Time  = {avg_time/TASK_NUM}"         , file=logfile)
        print(f"\tAvg Used Money = {avg_money/TASK_NUM}"        , file=logfile)

    