# QuickStart Guide & Examples

This directory contains example baselines on the **UA**$^2$-Webshop environment to help you replicate our experiments effectively.

## Quick Start

To begin, we provide a simple template `example.py`. This will guide you through interacting with the **UA**$^2$-Webshop environment and executing actions at each step.

### Step0: Set up your OPENAI_API_KEY

```shell
export OPENAI_API_KEY=<your_openai_api_key>
```

### Step1: Execute action

Use `ua2webshopRunTimeEnv.step(user_idx, task_idx, action, url_log_file=None)` to execute an action.

Parameters:
- `user_idx`: Index of the user.
- `task_idx`: Index of the task.
- `action`: Action to be executed.
- `url_log_file`: Optional file to log the action and URL post-execution.

Returns:
- `url`: URL of the web page after the action is executed.
- `all_obs`: Contains the web observation, costs and other relevant information.
- `done`: Indicates whether the task is complete.
- `API_success`: Status of LLM API call.
- `inter_rwd`: Reward from the action.
- `all_used_time`: Cumulative time used since the task began (excludes time used in your logic, e.g., sleep after API failure).
- `all_used_money`: Cumulative money spent since the task began.

### Step2: Get the next action

Complete the `get_next_action()` function. This function is crucial for determining the subsequent move in the **UA**$^2$-Webshop environment. A legal action is composed of specific commands:
- `think[xxx]`: Use this action to internally process or decide the next step without interacting with the web environment. Replace 'xxx' with your specific thinking or decision-making process.
- `search[xxx]`: This action directs the agent to perform a search. Replace 'xxx' with the query you wish to search for.
- `click[xxx]`: Utilize this action to simulate a click in the webshop environment. Replace 'xxx' with the identifier of the item you want to click on.

Ensure that the actions you generate are valid and contextually appropriate for the task at hand. The effectiveness of your agent relies heavily on the relevance and strategic use of these actions.

## Baselines Example

We provide several baselines on the **UA**$^2$-Webshop:
- `react.py`
- `react_sc.py`
- `cot_sc.py`
- `cot_least_to_most.py`
- `lats/`
- `reflexion/`

Each example follows these steps:
- [Step1] Execute action.
- [Step2] Obtain the next action from the LLM.
  - [Step2.1] Generate the prompt for the LLM.
  - [Step2.2] Get the action from the LLM.

Search for "# [*]" in the code to find these sections easily.

### Environment Setup

Set up your environments with the following commands:

```shell
conda create -n ua2 python=3.11
conda activate ua2
pip install -r ../requirements.txt
```

And you need set up the OPENAI_API_KEY as well:

```shell
export OPENAI_API_KEY=<your_openai_api_key>
```

### Running the Baselines

Execute baseline examples except LATS and Reflexion using the command:

```shell
python <example.py>
```

Replace `<example.py>` with the desired baseline script.

To execute LATS and Reflexion, firstly switch working directory and run the test scripts:
```shell
# LATS
cd lats
./lats.sh

# Reflexion
cd reflexion
./reflexion.sh
```

### Evaluation of alignment gap in Section 4.3 of the paper [**UA**$^2$](https://arxiv.org/abs/2402.07744)
To evaluate alignment gap, we need evaluate the corresponding agent on the ablated version of **UA**$^2$-Webshop. We can achieve this by import different env class from different python file.
- $\mathbf{G}_{\mathrm{ED}}$:
```python
from environments.env_instr_list_ua2webshop_runtime import ua2webshopRunTimeEnv
```
- $\mathbf{G}_{\mathrm{ED}}$:
```python
from environments.env_instr_list_ua2webshop_runtime_session_d import ua2webshopRunTimeEnv
```