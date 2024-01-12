import numpy as np

import reasoners.benchmark.cwebshop_utils as utils
import sys
sys.path.append('.')
from world_model import CWebShopState, CWebShopAction
from reasoners import SearchConfig, LanguageModel

import tiktoken


class CWebShopConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 batch_size=1,
                 reward_alpha=0.5,
                 goal_reward_default=0.,
                 goal_reached_reward=100) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        assert self.batch_size == 1, "batch_size for openai_api must be 1"
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        
        self.encoding_num = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct-0914")

    def get_actions(self, state: CWebShopState) -> list[CWebShopAction]:
        cwebshop_state = state.cwebshop_state
        return utils.generate_all_actions(state, self.base_model)

    def fast_reward(self, state: CWebShopState, action: CWebShopAction) -> tuple[float, dict]:
        # raise NotImplementedError
        if state.buffered_action == "":
            # if no action buffered
            current_cwebshop_state = state.cwebshop_state
        else:
            # if action buffered
            current_cwebshop_state = state.last_cwebshop_state
        previous_action = state.buffered_action + "\n" if state.buffered_action != "" else ""
        
        # icl_template = self.prompt["icl_list"][state.step_idx // 2] #TODO
        # every two step, we will deduct the icl prompt
        # so that the distribution of step length is more reasonable
        
        # icl_template = self.prompt['icl_actor']

        # inputs = icl_template.replace("<init_state>", current_cwebshop_state)\
        #    .replace("<action>", previous_action)
        # intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]

        self_eval_prompt = self.prompt["self-eval"].replace("<current_trial>", current_cwebshop_state).replace("<action>", action)
        
        while len(self.encoding_num.encode(self_eval_prompt)) + 10 >= 4096:
            action_id_0 = current_cwebshop_state.find("Action:")
            action_id_1 = current_cwebshop_state[action_id_0+1:].find("Action") + action_id_0 + 1
            truncated_current_cwebshop_state = current_cwebshop_state[:action_id_0].strip() +"\n\n"+ current_cwebshop_state[action_id_1:].strip()
            current_cwebshop_state = truncated_current_cwebshop_state
            self_eval_prompt = self.prompt["self-eval"].replace("<current_trial>", current_cwebshop_state).replace("<action>", action)

        # self_eval = self.base_model.get_loglikelihood(self_eval_prompt, 
            # [self_eval_prompt + "good"])[0]
        self_eval = self.base_model.generate([self_eval_prompt], max_tokens=10).text.strip()
        # assert self_eval in ['good', 'bad'], "self-eval must be 'good' or 'bad'"
        self_eval = 1 if self_eval == 'good' else 0
        return self_eval, {'intuition': 0, "self_eval": self_eval}
        # return self.calculate_reward(intuition, self_eval), {'intuition': intuition, "self_eval": self_eval}

    def calculate_reward(self, intuition, self_eval, goal_reached=None):
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, state: CWebShopState, action: CWebShopAction,
               intuition: float = None,
               self_eval: float = None,
               goal_reached: tuple[bool, float] = None, **kwargs) -> float:
        assert intuition is not None, "intuition is required to calculate reward in this search config, consider passing it in fast_reward"
        assert self_eval is not None, "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
        # assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        return (self.calculate_reward(intuition, self_eval, goal_reached), 
                {'intuition': intuition, 'goal_reached': goal_reached})

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)