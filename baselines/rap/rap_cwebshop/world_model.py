"""The world model for the C-WebShop."""

from typing import NamedTuple
import reasoners.benchmark.cwebshop_utils as utils
import re
from reasoners import WorldModel, LanguageModel
import copy
import sys
sys.path.append("../../../")
from environments.apis import OpenAI_API_Calling, num_tokens_from_messages

def num_tokens(prompt, model='gpt-3.5-turbo-instruct-0914'):
    num, _, _ = num_tokens_from_messages(prompt, model)
    return num

CWebShopAction = str
class CWebShopState(NamedTuple):
    """The state of the C-WebShop.
    
    See the docstring of CWebShopWorldModel for more details.
    """
    step_idx: int
    last_cwebshop_state: str
    cwebshop_state: str
    current_page: str
    buffered_action: CWebShopAction


class CWebShopWorldModel(WorldModel):
    """CWebShop World Model
    State: (step_idx, last_cwebshop_state, cwebshop_state, buffered_action)
    Action: e.g. "click[BZ123sadzsad]"
    Additional notes about the state:
        the block state is updated every two actions. When there is a block in hand, 
        the block state is not updated, but the action is buffered. With the next action, 
        the block state is updated and the buffer is cleared.
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 6,
                 batch_size=1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        assert self.batch_size==1, "batch_size for openai_api must be 1"

    def init_state(self, **kwargs) -> CWebShopState:
        """Initialize the world model.

        :return: the initial state
        """
        return CWebShopState(step_idx=0, last_cwebshop_state="", cwebshop_state=utils.
                       extract_init_state(self.example), buffered_action="", current_page=kwargs['current_page'])

    def step(self, state: CWebShopState, action: CWebShopAction) -> tuple[CWebShopState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of CWebShopWorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        state = copy.deepcopy(state)
        buffered_action = state.buffered_action
        last_page = state.current_page
        cwebshop_state = state.cwebshop_state
        step_idx = state.step_idx
        cwebshop_state, world_output = self.update_cwebshop_state(cwebshop_state, action)
        if state.buffered_action == "":
            # if no action buffered, buffer the action
            new_buffered_action = action
        else:
            # if action buffered, clear the buffer
            new_buffered_action = ""

        is_any_button = len(re.findall(r'\[(.*?)\]', world_output)) > 0
        if is_any_button:
            current_page = world_output
        else:
            current_page = last_page

        state = CWebShopState(step_idx=step_idx+1, last_cwebshop_state=state.cwebshop_state,
                        cwebshop_state=cwebshop_state, buffered_action=new_buffered_action,
                        current_page=current_page)
        
        
        return state, {"goal_reached":None}
        # return state, {"goal_reached": utils.goal_check(utils.extract_goals(self.example), webshop_state)}

    def update_cwebshop_state(self, cwebshop_states: str, action: CWebShopAction) -> str:
        """Update the cwebshop states with the action.

        :param cwebshop_states: the current cwebshop states. Note that this argument is a string,
            and it's only a part of 'BWState'
        :param action: the action to take
        :return: the updated cwebshop states
        """
        # if "pick" in action:
        #     key = "world_update_pickup"
        # elif "unstack" in action:
        #     key = "world_update_unstack"
        # elif "put" in action:
        #     key = "world_update_putdown"
        # elif "stack" in action:
        #     key = "world_update_stack"
        # else:
        #     raise ValueError("Invalid action")
        if action.startswith("click"):
            key = "world_update_click"
        elif action.startswith("search"):
            key = "world_update_search"
        else:
            world_output = "Invalid Action!"
            return f"{cwebshop_states.strip()}\n\nAction: {action}\nObservation:\n{world_output}\n", world_output 
            # raise ValueError("Invalid action")
        world_update_prompt = self.prompt[key].format(cwebshop_states, action)

        t_cwebshop_states = cwebshop_states
        while(num_tokens(world_update_prompt) > 4097-self.base_model.max_tokens):
            id0 = t_cwebshop_states.find('Action:')
            obs0 = t_cwebshop_states[:id0]
            t_cwebshop_states = t_cwebshop_states[id0+1:]
            id0 = t_cwebshop_states.find('Action:')
            t_cwebshop_states = t_cwebshop_states[id0:]

            t_cwebshop_states = obs0 + t_cwebshop_states
            # world_update_prompt = updatting_prompt + current_trial
            world_update_prompt = self.prompt[key].format(t_cwebshop_states, action)

        world_output = self.base_model.generate([world_update_prompt],
                                    eos_token_id="\n", hide_input=True, temperature=0).text.strip()
        new_state_id = world_output.find("[NEW STATE]")
        new_state = world_output[new_state_id+len("[NEW STATE]"):].strip()
        new_state = f"{cwebshop_states.strip()}\n\nAction: {action}\nObservation:\n{world_output}\n"
        
        # new_state = utils.apply_change(world_output, cwebshop_states)
        return new_state, world_output

    def is_terminal(self, state: CWebShopState) -> bool:
        # if utils.goal_check(utils.extract_goals(self.example), state.cwebshop_state)[0]:
        #     return True
        # elif state.step_idx == self.max_steps:
        #     return True
        if state.step_idx == self.max_steps:
            return True
        return False
