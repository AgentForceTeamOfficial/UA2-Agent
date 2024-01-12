# **C**ost-sensitive **Agent** (**C-Agent**) Framework


> 🚀 This repo implements the preliminary version of **C**ost-sensitive **Agent** framework (**C-Agent**) with results benchmarked on [C-Webshop](https://github.com/AgentForceTeamOfficial/C-WebShop).  
> 🚀 The project is a practice of LLM-powered agent framework design under the guidance of [*Cost Desiderata in LLM-Powered Agent Research*](https://agent-force.github.io/cost-desiderata-agent-research.html).


## Performance Comparison on C-Webshop

![Plot](./figs/performance-comparison.png)

🎡 **Methods used for comparison**:

- [**CoT-SC**](https://arxiv.org/abs/2203.11171): Sampling diverse Chain-of-Thought reasoning paths and action candidates, and voting for the majority to enhance self consistency.
- [**ReAct**](https://arxiv.org/abs/2210.03629): On each step, the agent decides whether to take actions or reason about the query based on the current state.
- **ReACT-SC**: Instead of [Chain-of-Thought reasoning](https://arxiv.org/abs/2201.11903), we apply sel consistency strategy to ReAct to further improve the performance.
- [**LATS**](https://arxiv.org/abs/2310.04406): an advanced method that unifies ReAct, self-reflection, and tree search based planning. We adapt the original implementation in our scenario for comparison.
- **Ours**: the C-Agent framework, which is introduced in the below section.

## Introduction

### 🌐 C-Webshop

To be specific, for the environment construction part, the key contribution of C-WebShop to the LLM-powered agent research community is the concept of **cost** rooted in the agent-assisting process. In C-WebShop, cost originates from different sources. To name a few:
  - Self cost of LLM-powered agents. Money, time, space, etc.
  - Exploration cost from the environment. The environment to be interacted with can be evolving w.r.t. user interactions, where the cost of exploration is formed.
  - Alignment Cost from the human user. The agent is not always aware of the user preference, and the cost of alignment is formed.
    
Kindly refer to the online [article](https://agent-force.github.io/cost-desiderata-agent-research.html) for detailed categorization of **cost**. The live site demo can be found [here](http://49.232.144.86:5000/), as well as the environment [repo](https://github.com/AgentForceTeamOfficial/C-WebShop) for local deployment purpose.

The exploration cost and the alignment cost are already reflected by the task and the website design of C-WebShop. In this repo, we wrap the environment with cost-related runtime information (time and money) to compute the self cost of an LLM-powered agent.

### 🤖 C-Agent Framework

The key challenge is to build an agent framework that manages to assist the decision process in a realistic environment, in consideration of different types of cost. We leverage structured profiles with free-form insights to make better decisions. 

![Profiler](./figs/Profiler.png)

💡 Here are some principles we follow:

- **Reduce alignment cost**: quickly adapting to the user's preference by retrieving trajectories with high rewards beforehand.
- **Reduce exploration cost**: directly extrapolating the key actions that lead to the success of the retrieved trajectories.
- **Reduce self cost**: extracting structured records from free-form insights, without the aid of LLMs or human annotators; retrieving the most similar profile for reference, instead of agent planning from scratch or searching in huge memory/experience space.

**Two major modules**:

- *Insight* part:

  - Get profiles from *Profiler* to better complete the current instruction
  - Recognize key actions of the current trajectory via reflection

- *Profile* part:

  - Long-term experience accumulation
  - Structured representation for instruction relations
  - Direct action extrapolation from the action list
  - Semantic similarity for profile retrieval

    *(For better efficiency, we only utilize the best matched profile as a reference in decision making)*

## What is the function of different directories?

- `./cwebshop-algo`: the core of our **C-agent** framework
  - `Insight.py`: the implementation of *Insight* part
  - `Profiler.py`: the implementation of *Profiling* part
  - `react_w_insights_w_profiler_v1benchmark.py`: the implementation of our C-Agent algorithm on C-Webshop benchmark

- `./environments`: running environment of our C-Webshop benchmark (the encapsulation of our core environment)
  - `env_instr_list_cwebshop_runtime_session.py`: the capsule of run-time environment leveraging cost information for C-Webshop benchmark

- `./baselines`: source code of baselines
  - `README.md`: the instruction of how to run baselines and implementation details

## How to replicate experiments of baselines and ours?

Prepare for the conda environment:
```sh
conda create -n cdab
pip install -r requirements.txt
conda activate cdab
```

Add your OpenAI API key to your environment:
```sh
# on Linux/Mac
export OPENAI_API_KEY=<YOUR_API_KEY>
```

```powershell
# on Windows
set OPENAI_API_KEY=<YOUR_API_KEY>
```

For ReAct-series and CoT-series baselines, change your working directory and run the corresponding script directly:
```sh
cd baselines
python cot_least_to_most.py
python cot_sc.py
python react.py
python react_sc.py
```
For LATS baselines:
```sh
cd baselines/lats
mkdir runtime_logs
python lats.py
```

For RAP baselines:
```sh
cd baselines/rap/rap_webshop
python rap_inference_v1benchmark.py
```

To test our method:
```sh
cd code
python react_w_insights_w_profiler_v1benchmark.py
```

After running the script, the results can be found in the directory `./runtime_logs`

## Authors

<div>
    <span style="vertical-align: middle">The C-Agent project is a collaborative effort of</span> <img src="figs/agentforce-logo.jpg" style="width:2em;vertical-align: middle" alt="Logo"/> <span style="vertical-align: middle"><b>AgentForce Team</b>. The method design, implementation and the testing of baseline methods in the context of the <a href="https://github.com/AgentForceTeamOfficial/C-WebShop">C-WebShop</a> is initiated and co-led by <a href="https://www.linkedin.com/in/%E5%AD%90%E5%90%9B-%E5%88%98-164596263/">Zijun Liu</a> (<a href="mailto: liuzijun20@mails.tsinghua.edu.cn">liuzijun20@mails.tsinghua.edu.cn</a>) and <a href="https://github.com/xxmlala">An Liu</a> (<a href="mailto: la22@mails.tsinghua.edu.cn">la22@mails.tsinghua.edu.cn</a>). The following members are listed with main contributions:</span> 
</div>

- [An Liu](https://github.com/xxmlala) developed the insight-module, conducted thorough experiments, implemented baselines and data visualization.
- [Zijun Liu](https://github.com/BBQGOD) initiated the conceptualization of Profiler, define and refine the C-Agent Framework leveraging structured profiles, and conducted thorough experiments.
- [Kaiming Liu](https://github.com/KMing-L) developed the runtime environment and the retrieval component of the Profiler, as well as calibrating the presentation of performances of different methods.
- [Zeyuan Yang](https://github.com/MiicheYang) and [Zonghan Yang](https://minicheshire.github.io) contributed to the initial version of the runtime environment wrapper.
- [Zonghan Yang](https://minicheshire.github.io) implemented the initial build of ReAct, and was also in charge of the final version of data visualization.
- [Zhicheng Guo](https://github.com/zhichengg), [Qingyuan Hu](https://github.com/HQY188), [Kaiming Liu](https://github.com/KMing-L), [An Liu](https://github.com/xxmlala), [Zijun Liu](https://github.com/BBQGOD), and [Zonghan Yang](https://minicheshire.github.io) collaborated on the implementation of the baseline methods and their evaluation. The respective leaders are:
  - LATS: [Zhicheng Guo](https://github.com/zhichengg) and [An Liu](https://github.com/xxmlala)
  - RAP: [An Liu](https://github.com/xxmlala)
  - ReAct: [Kaiming Liu](https://github.com/KMing-L) and [Zonghan Yang](https://minicheshire.github.io)
  - CoT-L2M: [Qingyuan Hu](https://github.com/HQY188) and [Kaiming Liu](https://github.com/KMing-L)
  - ReAct-SC & CoT-SC: [Kaiming Liu](https://github.com/KMing-L)
  - Overall co-lead: [An Liu](https://github.com/xxmlala) and [Zijun Liu](https://github.com/BBQGOD)
- [An Liu](https://github.com/xxmlala), [Zijun Liu](https://github.com/BBQGOD), and [Kaiming Liu](https://github.com/KMing-L) also provided significant advice to the construction and configuration of the C-WebShop environment.


This project is advised by [Peng Li](https://www.lpeng.net/) (lipeng@air.tsinghua.edu.cn) and [Yang Liu](https://nlp.csai.tsinghua.edu.cn/~ly/) (liuyang2011@tsinghua.edu.cn).

## Contributions

We look forward to all kinds of suggestions from anyone interested in our project with whatever backgrounds! Either PRs, issues, or leaving a message is welcomed. We'll be sure to follow up shortly!

<!-- ## If you find this repo useful, please cite our project:

```bibtex
``` -->
