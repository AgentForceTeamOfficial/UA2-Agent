import re
import random

prompt_possible_search = """Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search] 

Output the top-2 search query according to the instruction.
Optional Actions:
search[bright cirtus deodorant]
search[deodorant for sensitive skin]


{}

Output the top-{} search query according to the instruction.
Optional Actions:
"""



def generate_all_actions(state, llm=None, actions_top_k=3):
    assert llm!=None
    current_page = state.current_page
    if "[Search]" in current_page:
        prompt = prompt_possible_search.format(current_page, actions_top_k)
        response = llm.generate([prompt]).text.strip()
        possible_search_actions = []
        # if "search" in response:
        for i in range(actions_top_k):
            response = response.strip()
            end_id = response.find("]")
            possible_search_actions.append(response[:end_id+1])
            assert "search[" in possible_search_actions[-1] and "]" in possible_search_actions[-1]
            response = response[end_id+1:]#.strip()
        return random.sample(possible_search_actions, min(len(possible_search_actions), actions_top_k))

    # elif "\n[Back to Search]" in current_page: #TODO webpage's inheritage
    elif "[" in current_page:
        buttons = re.findall(r'\[(.*?)\]', current_page)
        buttons = [f"click[{i}]" for i in buttons]
        possible_click = random.sample(buttons, min(len(buttons), actions_top_k))
        if ["click[Buy Now]"] not in possible_click:
            if len(possible_click)<actions_top_k:
                possible_click.append("click[Buy Now]")
            else:
                possible_click[-1] = "click[Buy Now]"
        return possible_click
    else:
        raise Exception
    
    # raise NotImplementedError


def goal_check(goals, cwebshop_state):
    """Check if the goals are met and return the percentage of goals met

    :param goals: goals
    :param cwebshop_state: current cwebshop state
    """
    meetings = [g in cwebshop_state for g in goals]
    # print("Goals:", goals)
    # print("Goal met:", meetings)
    if sum(meetings) == len(meetings):
        return True, 1.0
    return False, sum(meetings) / len(meetings)


def extract_goals(example, return_raw=False):
    """Extract the goals from the example
    
    :param example: example
    """
    goal_statement = example["question"].split("[STATEMENT]")[-1]\
        .split("My goal is to ")[1].split("My plan is as follows")[0].strip()
    if return_raw:
        return goal_statement
    goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
    return goals


def extract_init_state(example):
    """Extract the initial state from the example
    
    :param example: example
    """
    start_id = example.find('Action: click[Buy Now]') + len('Action: click[Buy Now]')
    end_id = example.find('Action: <action>')
    return example[start_id:end_id].strip()
    # raise NotImplementedError
    # print(example)
    # init_statement = example["question"].split("[STATEMENT]\nAs initial conditions I have that, ")[1]\
        # .split("My goal")[0].strip()
    # return init_statement