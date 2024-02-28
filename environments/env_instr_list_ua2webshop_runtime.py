from .apis import OpenAI_API_Calling
import json
import time
import requests
import urllib
import re

# Parse the web page
from bs4 import BeautifulSoup
from bs4.element import Comment

WEBSHOP_URL = "<path to the env without HI>"
ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}

ACTION_TO_TIME = {
    r"search\[.*": 0.5966,
    r"think\[.*": 0.0000,
    r"click\[Buy Now\]": 0.1920,
    r"reset": 0.1874,
    r"click\[Next >\]": 0.2693,
    r"click\[< Prev\]": 0.2545,
    r"click\[Back to Search\]": 0.1197,
    r"click\[Reviews\]": 0.1275,
    r"click\[Features\]": 0.2167,
    r"click\[Instruction History\]": 0.2645,
    r"click\[Descriptions\]": 0.2401,
    r".*": 0.2896,
}
INVALID_TIME = 0.3234


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)


def ua2webshop_text(
    user_idx,
    task_idx,
    page_type,
    query_string="",
    page_num=1,
    asin="",
    options={},
    subpage="",
    **kwargs,
):
    quoted_query_string = urllib.parse.quote(str(query_string))
    quoted_options = urllib.parse.quote(str(options))
    if page_type == "init":
        url = f"{WEBSHOP_URL}/{user_idx}/{task_idx}"
    elif page_type == "search":
        url = f"{WEBSHOP_URL}/search_results/{user_idx}/{task_idx}/{quoted_query_string}/{page_num}"
    elif page_type == "item":
        url = f"{WEBSHOP_URL}/item_page/{user_idx}/{task_idx}/{asin}/{quoted_query_string}/{page_num}/{quoted_options}"
    elif page_type == "item_sub":
        url = f"{WEBSHOP_URL}/item_sub_page/{user_idx}/{task_idx}/{asin}/{quoted_query_string}/{page_num}/{subpage}/{quoted_options}"
    elif page_type == "instr_history":
        url = f"{WEBSHOP_URL}/instruction_history/{user_idx}/{task_idx}"
    elif page_type == "end":
        url = f"{WEBSHOP_URL}/done/{user_idx}/{task_idx}/{asin}/{quoted_options}"
    while True:
        try:
            html = requests.get(url).text
            html_obj = BeautifulSoup(html, "html.parser")
            texts = html_obj.findAll(text=True)
            visible_texts = list(filter(tag_visible, texts))
            res_str = ""
            for t in visible_texts:
                if t == "\n":
                    continue
                if t.replace("\n", "").replace("\\n", "").replace(" ", "") == "":
                    continue
                res_str += t
            if len(res_str) != 0:
                break
        except:
            pass

    if False:
        # For `simple` mode, return just [SEP] separators
        return " [SEP] ".join(t.strip() for t in visible_texts if t != "\n")
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ""
        option_type = ""
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == "\n":
                continue
            if t.replace("\n", "").replace("\\n", "").replace(" ", "") == "":
                continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == "button":  # button
                processed_t = f"\n[{t}] "
            elif t.parent.name == "label":  # options
                if f"'{t}'" in url:
                    processed_t = f"[[{t}]]"
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f"[{t}]"
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get("class") == ["product-link"]:  # product asins
                processed_t = f"\n[{t}] "
                # if prod_cnt >= 3:
                #     processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else:  # regular, unclickable text
                processed_t = "\n" + str(t) + " "
                if cnt < 2 and page_type != "init":
                    processed_t = ""
                # if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
            info["option_types"] = options
        if asins:
            info["asins"] = asins
        if "Your score (min 0.0, max 1.0)" in visible_texts:
            idx = visible_texts.index("Your score (min 0.0, max 1.0)")
            info["reward"] = float(visible_texts[idx + 1])
            observation = "Your score (min 0.0, max 1.0): " + (visible_texts[idx + 1])
        initial_url = url
        url = url[url.find("%7B") + 3 :]
        url = url.split("%27%3A")
        for i in range(1, len(url)):
            clicked_tag = url[i][6:][: url[i][6:].find("%27")]
            clicked_tag = urllib.parse.unquote(clicked_tag)
            if clicked_tag in observation:
                observation = (
                    observation[: observation.find(clicked_tag) + len(clicked_tag) + 1]
                    + "(have clicked) "
                    + observation[
                        observation.find(clicked_tag) + len(clicked_tag) + 1 :
                    ]
                )
        return clean_str(observation), info, initial_url


class ua2webshopEnv:
    def __init__(self):
        self.sessions = {}

        self.user_idx = -1
        self.task_idx = -1

    def interactive_env_step(self, action):
        reward = 0.0
        done = False
        observation_ = None
        observation = None
        current_url = ""
        try:
            if action == "reset":
                self.sessions[self.user_idx][self.task_idx]["page_type"] = "init"
            elif action.startswith("think["):
                observation = "OK."
            elif action.startswith("search["):
                assert (
                    self.sessions[self.user_idx][self.task_idx]["page_type"] == "init"
                )
                query = action[7:-1]
                self.sessions[self.user_idx][self.task_idx].update(
                    {"page_type": "search", "query_string": query, "page_num": 1}
                )
            elif action.startswith("click["):
                button = action[6:-1]
                if button == "Buy Now":
                    assert (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "item"
                    )
                    self.sessions[self.user_idx][self.task_idx]["page_type"] = "end"
                    done = True
                elif button == "Back to Search":
                    assert self.sessions[self.user_idx][self.task_idx]["page_type"] in [
                        "search",
                        "item",
                        "item_sub",
                        "instr_history",
                    ]
                    self.sessions[self.user_idx][self.task_idx]["page_type"] = "init"
                elif button == "Next >":
                    assert (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "search"
                    )
                    self.sessions[self.user_idx][self.task_idx]["page_num"] += 1
                elif button == "< Prev":
                    assert self.sessions[self.user_idx][self.task_idx]["page_type"] in [
                        "search",
                        "item",
                        "item_sub",
                    ]
                    if (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "search"
                    ):
                        self.sessions[self.user_idx][self.task_idx]["page_num"] -= 1
                    elif (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "item_sub"
                    ):
                        self.sessions[self.user_idx][self.task_idx][
                            "page_type"
                        ] = "item"
                    elif (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "item"
                    ):
                        self.sessions[self.user_idx][self.task_idx][
                            "page_type"
                        ] = "search"
                        self.sessions[self.user_idx][self.task_idx]["options"] = {}
                elif button == "Instruction History":
                    assert (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "init"
                    )
                    self.sessions[self.user_idx][self.task_idx][
                        "page_type"
                    ] = "instr_history"
                elif button in ACTION_TO_TEMPLATE:
                    assert (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "item"
                    )
                    self.sessions[self.user_idx][self.task_idx][
                        "page_type"
                    ] = "item_sub"
                    self.sessions[self.user_idx][self.task_idx]["subpage"] = button
                else:
                    if (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "search"
                    ):
                        assert button in self.sessions[self.user_idx][
                            self.task_idx
                        ].get("asins", [])
                        self.sessions[self.user_idx][self.task_idx][
                            "page_type"
                        ] = "item"
                        self.sessions[self.user_idx][self.task_idx]["asin"] = button
                    elif (
                        self.sessions[self.user_idx][self.task_idx]["page_type"]
                        == "item"
                    ):
                        # print("tag1")
                        assert (
                            "option_types"
                            in self.sessions[self.user_idx][self.task_idx]
                        )
                        # print("tag2")
                        assert (
                            button
                            in self.sessions[self.user_idx][self.task_idx][
                                "option_types"
                            ]
                        ), (
                            button,
                            self.sessions[self.user_idx][self.task_idx]["option_types"],
                        )
                        option_type = self.sessions[self.user_idx][self.task_idx][
                            "option_types"
                        ][button]
                        if not "options" in self.sessions[self.user_idx][self.task_idx]:
                            self.sessions[self.user_idx][self.task_idx]["options"] = {}
                        self.sessions[self.user_idx][self.task_idx]["options"][
                            option_type
                        ] = button
                        observation_ = f"You have clicked {button}."
                    else:
                        assert False
            else:
                # print("tag4")
                assert False
            # print("ua2webshop_text")
            observation_, info, current_url = ua2webshop_text(
                **self.sessions[self.user_idx][self.task_idx]
            )
            if observation_:
                # print("observation = observation_")
                observation = observation_
            self.sessions[self.user_idx][self.task_idx].update(info)
            self.sessions[self.user_idx][self.task_idx]["step"] += 1
            # print(info)
            reward = info.get("reward", 0.0)
        except AssertionError:
            observation = "Invalid Action!"

        return observation, reward, done, current_url


class ua2webshopRunTimeEnv(ua2webshopEnv):
    def __init__(self, init_money, init_time) -> None:
        super().__init__()
        self.init_money = init_money
        self.init_time = init_time

    def step(self, user_idx, task_idx, action):
        self.user_idx = user_idx
        self.task_idx = task_idx

        init_runtime_obs = ""
        runtime_obs = ""
        # st_time = time.time()
        cost = 0
        API_success = True

        try:
            if action == "reset":
                if self.sessions.get(user_idx) is None:
                    self.sessions[user_idx] = {}
                self.sessions[user_idx][task_idx] = {
                    "user_idx": user_idx,
                    "task_idx": task_idx,
                    "step": 0,
                    "rest_money": self.init_money,
                    "rest_time": self.init_time,
                }
                init_runtime_obs = """You are assisting in web-shopping by interacting with a website.\n\nYou can also utilize the APIs of powerful Large Language Models to assist the decision-making process. Currently the following OpenAI models are supported:
gpt-3.5-turbo-0301 (ChatCompletion Model),
gpt-3.5-turbo-0613 (ChatCompletion Model),
gpt-3.5-turbo-1106 (ChatCompletion Model),
gpt-3.5-turbo-instruct-0914 (Instruct Completion Model).\nTo seek its advice, please input your query in the format of ask[MODEL][PROMPT][ARGS], with MODEL as the previous listed model names, PROMPT as the input prompt (a string for Instruct Completion Model; a dumped list of messages for ChatCompletion Models), ARGS as the hyperparameters of the model in the form of a dictionary ({'max_tokens': xx, 'temperature': xx, 'top_p': xx, 'stop': xx, ...}) dumped as a string.\n\nIt is important to note that utilizing this action incurs a cost as well. To estimate the cost of the API calling, you can query in the format of estimate_cost[MODEL][PROMPT][ARGS]. The default ARGS is {'temperature': 0.0, 'max_tokens': 100} and can be overridden."""
            assert (
                self.sessions.get(user_idx) is not None
            ), "Please reset the environment first."
            assert (
                self.sessions[user_idx].get(task_idx) is not None
            ), "Please reset the environment first."

            current_url = ""

            if action.startswith("ask[") or action.startswith("estimate_cost["):
                inter_obs = ""
                inter_rwd = 0.0
                inter_done = False
                ALL_AVAILABLE_MODELS = [
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo-instruct-0914",
                ]
                ALL_AVAILABLE_CHAT_MODELS = [
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-1106",
                ]

                if action.startswith("ask["):
                    prompt = action[4:-1]
                else:
                    prompt = action[14:-1]

                prompt_spt_tmp = prompt.split("][")
                prompt_spt = [
                    prompt_spt_tmp[0],
                    "][".join(prompt_spt_tmp[1:-1]),
                    prompt_spt_tmp[-1],
                ]
                default_kwargs_dict = {"max_tokens": 100, "temperature": 0.0}
                isValid = True
                try:
                    kwargs_dict = json.loads(prompt_spt[2])
                    for kk in default_kwargs_dict:
                        if not (kk in kwargs_dict):
                            kwargs_dict[kk] = default_kwargs_dict[kk]

                    if prompt_spt[0] in ALL_AVAILABLE_CHAT_MODELS:
                        now_prompt = json.loads(prompt_spt[1])
                except:
                    isValid = False

                if (
                    (prompt_spt[0] in ALL_AVAILABLE_MODELS)
                    and isValid
                    and action.startswith("ask[")
                ):
                    response, cost, success, ttime = OpenAI_API_Calling(
                        prompt_spt[1], model=prompt_spt[0], kwargs_dict=kwargs_dict
                    )
                    if len(response) > 0:
                        response = json.dumps(response)
                    runtime_obs = (
                        f"LLM_response[{response.strip()}]" + "\n" + "==" * 20 + "\n"
                    )
                    API_success = success
                    ttime = 0.0
                else:
                    ttime = 0.0
                    runtime_obs = "Invalid Action!" + "\n" + "==" * 20 + "\n"
                    API_success = False
            else:
                inter_obs, inter_rwd, inter_done, current_url = (
                    self.interactive_env_step(action)
                )
                if inter_obs == "Invalid Action!":
                    ttime = INVALID_TIME
                else:
                    for pattern, time_ in ACTION_TO_TIME.items():
                        if re.findall(pattern, action):
                            ttime = time_
                            break
        except AssertionError:
            runtime_obs = "Invalid Action!" + "\n" + "==" * 20 + "\n"

        # ttime = time.time() - st_time

        runtime_obs += "\nTime Cost: " + str(ttime)
        runtime_obs += "\nMoney Cost: $" + str(cost)
        self.sessions[user_idx][task_idx]["rest_money"] -= cost
        self.sessions[user_idx][task_idx]["rest_time"] -= ttime

        done = inter_done
        if self.sessions[user_idx][task_idx]["rest_money"] < 0:
            runtime_obs += "\nMoney Limit Exceeded. Fail!"
            done = True
        if self.sessions[user_idx][task_idx]["rest_time"] < 0:
            runtime_obs += "\nTime Limit Exceeded. Fail!"
            done = True

        if (
            done
            and self.sessions[user_idx][task_idx]["rest_money"] >= 0
            and self.sessions[user_idx][task_idx]["rest_time"] >= 0
        ):
            runtime_obs += f"\n\nTOTAL TIME USED: {self.init_time - self.sessions[user_idx][task_idx]['rest_time']}"
            runtime_obs += f"\nTOTAL MONEY USED: {self.init_money - self.sessions[user_idx][task_idx]['rest_money']}"

        all_obs = "\n\n".join(
            [
                init_runtime_obs,
                f"Observation_from_Interactive_Env[\n{inter_obs}\n]",
                runtime_obs,
            ]
        )

        return (
            current_url,
            all_obs,
            done,
            API_success,
            inter_rwd,
            self.init_time - self.sessions[user_idx][task_idx]["rest_time"],
            self.init_money - self.sessions[user_idx][task_idx]["rest_money"],
        )


if __name__ == "__main__":
    env = ua2webshopRunTimeEnv(init_money=10000, init_time=10000)

    user_idx = 0
    task_idx = 10

    print("====== Reset ======")
    url, all_obs, done, API_success, inter_rwd, cost_time, cost_money = env.step(
        user_idx, task_idx, "reset"
    )
    print(f"Observation: {all_obs}")
    print(f"Done: {done}")
    print(f"API Success: {API_success}")
    print(f"Interactive Reward: {inter_rwd}")
    print(f"Cost Time: {cost_time}")
    print(f"Cost Money: {cost_money}")

    print("====== Instruction History =======")
    url, all_obs, done, API_success, inter_rwd, cost_time, cost_money = env.step(
        user_idx, task_idx, "click[Instruction History]"
    )
    print(f"Observation: {all_obs}")
    print(f"Done: {done}")
    print(f"API Success: {API_success}")
    print(f"Interactive Reward: {inter_rwd}")
    print(f"Cost Time: {cost_time}")
    print(f"Cost Money: {cost_money}")

    print("====== Back to Search =======")
    url, all_obs, done, API_success, inter_rwd, cost_time, cost_money = env.step(
        user_idx, task_idx, "click[Back to Search]"
    )
    print(f"Observation: {all_obs}")
    print(f"Done: {done}")
    print(f"API Success: {API_success}")
    print(f"Interactive Reward: {inter_rwd}")
    print(f"Cost Time: {cost_time}")
    print(f"Cost Money: {cost_money}")

    print("====== Search =======")
    all_obs, done, API_success, inter_rwd, cost_time, cost_money = env.step(
        user_idx, task_idx, "search[3 ounce bright citrus deodorant sensitive skin]"
    )
    print(f"Observation: {all_obs}")
    print(f"Done: {done}")
    print(f"API Success: {API_success}")
    print(f"Interactive Reward: {inter_rwd}")
    print(f"Cost Time: {cost_time}")
    print(f"Cost Money: {cost_money}")

    print("====== Click Item =======")
    all_obs, done, API_success, inter_rwd, cost_time, cost_money = env.step(
        user_idx, task_idx, "click[B078GWRC1J]"
    )
    print(f"Observation: {all_obs}")
    print(f"Done: {done}")
    print(f"API Success: {API_success}")
    print(f"Interactive Reward: {inter_rwd}")
    print(f"Cost Time: {cost_time}")
    print(f"Cost Money: {cost_money}")

    print("====== Click bright citrus =======")
    all_obs, done, API_success, inter_rwd, cost_time, cost_money = env.step(
        user_idx, task_idx, "click[bright citrus]"
    )
    print(f"Observation: {all_obs}")
    print(f"Done: {done}")
    print(f"API Success: {API_success}")
    print(f"Interactive Reward: {inter_rwd}")
    print(f"Cost Time: {cost_time}")
    print(f"Cost Money: {cost_money}")

    print("====== Click 3 ounce (pack of 1) =======")
    all_obs, done, API_success, inter_rwd, cost_time, cost_money = env.step(
        user_idx, task_idx, "click[3 ounce (pack of 1)]"
    )
    print(f"Observation: {all_obs}")
    print(f"Done: {done}")
    print(f"API Success: {API_success}")
    print(f"Interactive Reward: {inter_rwd}")
    print(f"Cost Time: {cost_time}")
    print(f"Cost Money: {cost_money}")

    print("====== Click Buy Now =======")
    url, all_obs, done, API_success, inter_rwd, cost_time, cost_money = env.step(
        user_idx, task_idx, "click[Buy Now]"
    )
    print(f"Observation: {all_obs}")
    print(f"Done: {done}")
    print(f"API Success: {API_success}")
    print(f"Interactive Reward: {inter_rwd}")
    print(f"Cost Time: {cost_time}")
    print(f"Cost Money: {cost_money}")
