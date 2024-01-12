import sys
from prompt_lib import *
import sacrebleu
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import json
from typing import Tuple
import time

sys.path.append("../")
from environments.apis import OpenAI_API_Calling

nltk.download("punkt")


class Profiler:
    def __init__(self, name, calling_func = OpenAI_API_Calling):
        self.name = name
        self.total_prompt_tokens, self.total_completion_tokens = 0, 0
        self.update_memory = []
        self.retrieve_memory = []
        self.calling_func = calling_func
        
        self.profile = None
        self.fields = None

        self.mem_size = 2

        self.init_profile()

    
    def init_profile(self):
        self.update_memory = [{"role": "system", "content": UPDATE_PROFILE_SYSTEM_PROMPT.format()}]
        self.retrieve_memory = [{"role": "system", "content": RETRIEVE_SYSTEM_PROMPT.format()}]
        self.profile = [
            {
                "instruction": "",
                "important actions": [],
                "impact on length": []
            },
            {
                "instruction": "explore",
                "important actions": [],
                "impact on length": []
            }
        ] # prior experience
        self.fields = {
            "instruction": str,
            "important actions": list,
            "impact on length": list
        }

    
    def update_profile(self, insights, instruction):
        instruction = UPDATE_PROFILE_INSTRUCTION.format("\n".join(insights["key_actions_reasoning"]))
        self.update_memory.append({"role": "user", "content": instruction})
        answer, _, _ = self.calling_func(self.update_memory)
        self.update_memory.append({"role": "assistant", "content": answer})
        self.update_memory = [self.update_memory[0]] + self.update_memory[1:] if len(self.update_memory[1:]) < self.mem_size*2 else self.update_memory[-self.mem_size*2:]
        return answer


    def retrieve(self, context):
        self.retrieve_memory.append({"role": "user", "content": RETRIEVE_INSTRUCTION.format(context)})
        answer, _, _ = self.calling_func(self.retrieve_memory)
        self.retrieve_memory.append({"role": "assistant", "content": answer})
        self.retrieve_memory = [self.retrieve_memory[0]] + self.retrieve_memory[1:] if len(self.retrieve_memory[1:]) < self.mem_size*2 else self.retrieve_memory[-self.mem_size*2:]
        return answer

    
    def __str__(self) -> str:
        # format self.profile
        return str(self.profile)

    # former example
    """
    [
        {
            "instruction": "",
            "important actions": [],
            "impact on length": []
        },
        {
            "instruction": "explore",
            "important actions": [],
            "impact on length": []
        },
        {
            "instruction": "i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars",
            "important actions": [
                "click[B078GWRC1J]",
                "think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]",
                "click[3 ounce (pack of 1)]",

            ],
            "impact on length": []
        }
    ]
    """

SEARCH_THRESHOLD = 100
UPDATE_THRESHOLD = 90
class MultiRoundProfiler(Profiler):
    def __init__(self, name, calling_func = OpenAI_API_Calling, retrieve_method = "bm25"):
        super().__init__(name, calling_func)

        self.init_profile()
        self.retrieve_method = retrieve_method
    
    def init_profile(self):
        self.update_memory = [{"role": "system", "content": UPDATE_PROFILE_SYSTEM_PROMPT.format()}]
        self.retrieve_memory = [{"role": "system", "content": RETRIEVE_SYSTEM_PROMPT.format()}]
        self.profile = [
            {
                "instruction": "", #global instruction
                "important actions": [], # abstract rules
                "pred": [],
                "succ": []
            }
        ] # prior experience
        self.fields = {
            "instruction": str, # green book
            "important actions": list, # actions (TODO: w/ reasons) 
            "pred": [{
                "impact": str,
                "instruction": str, # red book
            }],
            "succ": [{
                "impact": str,
                "instruction": str, # black book
            }]
        }
        
        self.corpus = []
        self.dictionary = Dictionary(self.corpus)
        self.bm25_model = None
        self.bm25_corpus = None
        self.bm25_index = None
        self.tfidf_model = None
        self.bm25_need_update = False

    @property
    def global_profile(self):
        for profile in self.profile:
            if profile["instruction"] == "":
                return profile

    def update_profile(self, insights, instruction, pre_inst=None, pre_impact=None, reward=1.0) -> Tuple[float, float]:
        """
        return: [money, time]
        """
        if reward <= 0.5:
            return 0.0, 0.0

        self.embeddings = []    # length of output: 1536


        def search_profile(now_inst, threshold=0.9):
            if threshold >= 100:
                for profile in self.profile:
                    if threshold >= 100:
                        if profile["instruction"] != "" and profile["instruction"].strip() == now_inst.strip():
                            return profile
                        else:
                            continue

                # if profile["instruction"] != "" and (profile["instruction"] in now_inst or now_inst in profile["instruction"] or sacrebleu.sentence_bleu(profile["instruction"], [now_inst]).score > threshold):
                #     return profile

                return None
            
            elif self.tfidf_model is not None:
                query = word_tokenize(now_inst)
                tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query)]
                similarities = self.bm25_index[tfidf_query]
                best_idx = np.argmax(similarities)
                if similarities[best_idx] < threshold:
                    return None
                else:
                    return self.profile[best_idx+1]
                
            return None
                
        def add_profile(instruction_prof, pre_inst=None):
            new_prof = {
                "instruction": instruction,
                "important actions": instruction_prof,
                "pred": [],
                "succ": []
            }
            self.profile.append(new_prof)
            if self.retrieve_method=="bm25":
                self.corpus.append(word_tokenize(instruction))
                self.dictionary.add_documents([word_tokenize(instruction)])
                self.bm25_need_update = True
                return [-1], 0.0, 0.0
            else:
                response, cost, success, time_cost = self.calling_func(instruction, model="text-embedding-ada-002")
                if success:
                    self.embeddings.append(np.array(response))
                else:
                    pass    # suppose ada will not fail
                return [-1], cost, time_cost

        def update_profile(instruction_prof, threshold, pre_inst=None):
            lidx = []
            for idx, profile in enumerate(self.profile):
                # if profile["instruction"] != "" and (profile["instruction"] in instruction or instruction in profile["instruction"] or sacrebleu.sentence_bleu(profile["instruction"], [instruction]).score > threshold):
                #     profile["important actions"] += instruction_prof
                #     lidx.append(idx)

                if profile["instruction"] == "":
                    lidx.append(idx)
                    
            query = word_tokenize(instruction)
            tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query)]
            similarities = self.bm25_index[tfidf_query]
            threshold = threshold / 100 * np.max(similarities)
            for idx, similarity in enumerate(similarities):
                if similarity >= threshold:
                    self.profile[idx+1]["important actions"] += instruction_prof
                    lidx.append(idx+1)
            
            return lidx
    
        threshold = 90
        pre_impact = insights["key_actions_reasoning"]

        cost = 0.0
        time_cost = 0.0

        pre_prof = None
        if pre_inst:
            pre_prof = search_profile(pre_inst, 100)

        ex_prof = search_profile(instruction, threshold)
        if ex_prof:
            lidx = update_profile(insights["key_click"], threshold)
        else:
            lidx, cost, time_cost = add_profile(insights["key_click"], pre_inst)

        same_pred_to_succ = False
        if pre_prof:
            for idx in lidx:
                if self.profile[idx]["instruction"] == pre_inst:
                    same_pred_to_succ = True
                    break
        if same_pred_to_succ:
            return cost, time_cost

        if pre_prof:
            for idx in lidx:
                if self.profile[idx]["instruction"] == "":
                    self.profile[idx]["important actions"].append(pre_impact)
                    continue
                
                pred_flag = False
                for pred in self.profile[idx]["pred"]:
                    if pred["instruction"] == pre_inst:
                        pred["impact"] = pre_impact if pre_impact else ""
                        pred_flag = True
                        break

                if not pred_flag:
                    self.profile[idx]["pred"].append({
                        "impact": pre_impact if pre_impact else "",
                        "instruction": pre_inst,
                    })
            
            for idx in lidx:
                if self.profile[idx]["instruction"] == "":
                    continue
                
                succ_flag = False
                for succ in pre_prof["succ"]:
                    if succ["instruction"] == self.profile[idx]["instruction"]:
                        succ_flag = True
                        break

                if not succ_flag:
                    pre_prof["succ"].append({
                        "impact": pre_impact if pre_impact else "",
                        "instruction": instruction,
                    })
        
        print("profile updated")
        print(self.profile)
        print("------------------")
        print("\n")
        return cost, time_cost

    def update_bm25(self):
        self.bm25_model = OkapiBM25Model(dictionary=self.dictionary)
        self.bm25_corpus = self.bm25_model[
            list(map(self.dictionary.doc2bow, self.corpus))
        ]
        self.bm25_index = SparseMatrixSimilarity(
            self.bm25_corpus,
            num_docs=len(self.corpus),
            num_terms=len(self.dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )
        self.tfidf_model = TfidfModel(dictionary=self.dictionary, smartirs="bnn")

    def retrieve(self, context: str, threshold : int = 0.9) -> Tuple[dict, float, float]:
        """
        context: instruction

        return one whole profile, money, time
        """
        def search_embedding(threshold: float) -> Tuple[dict, float, float]:
            if len(self.embeddings) == 0:
                return None, 0.0
            response, cost, success, _ = self.calling_func(context, model="text-embedding-ada-002")
            if success:
                target = np.array(response)
            else:
                return None, cost
            min_dist = threshold
            min_idx = -1
            for idx, vector in enumerate(self.embeddings):
                dist = np.linalg.norm(target - vector)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            if min_idx == -1:
                return None, cost
            else:
                return self.profile[min_idx+1], cost
        
        def search_bm25(threshold: float) -> Tuple[dict, float, float]:
            if self.bm25_need_update:
                self.update_bm25()
                self.bm25_need_update = False

            if self.bm25_model is None:
                return None, 0.0

            query = word_tokenize(context)
            tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query)]
            similarities = self.bm25_index[tfidf_query]
            similarities = [abs(similarity) for similarity in similarities]
            best_idx = np.argmax(similarities)
            if similarities[best_idx] < threshold:
                return None, 0.0
            else:
                return self.profile[best_idx+1], 0.0

        time_start = time.time()
        if self.retrieve_method=="bm25":
            profiler, cost = search_bm25(threshold)
        else:
            profiler, cost = search_embedding(float("inf"))
        time_end = time.time()
        return profiler, cost, time_end - time_start
        

    def suggest(self, observation):
        pass


    # former example
    """
    [
        {
            "instruction": "",
            "important actions": [],
            "impact on length": []
        },
        {
            "instruction": "explore",
            "important actions": [],
            "impact on length": []
        },
        {
            "instruction": "i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars",
            "important actions": [
                "click[B078GWRC1J]",
                "think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]",
                "click[3 ounce (pack of 1)]",

            ],
            "impact on length": []
        }
    ]
    """


if __name__ == "__main__":
    # profiler = MultiRoundProfiler("test", retrieve_method="embedding")
    profiler = MultiRoundProfiler("test", retrieve_method="bm25")
    print(profiler.update_profile(
        {
            "key_click": [
                "search[long clip-in hair extension]",
                "filter[natural looking]",
                "filter[price < $20]",
                "click[Product A - $18.99]",
            ],
            "key_actions_reasoning": None
        },
        "I need a long clip-in hair extension which is natural looking, and price lower than 20.00 dollars",
    ))
    print(profiler.update_profile(
        {
            "key_click": [
                "search[citrus deodorant sensitive skin]",
                "filter[price < $15]",
                "click[Product B - $12.99]",
            ],
            "key_actions_reasoning": None
        },
        "Find a deodorant suitable for sensitive skin, with a citrus scent, under $15.",
    ))
    print(profiler.update_profile(
        {
            "key_click": [
                "search[organic facial moisturizer for sensitive skin]",
                "filter[price < $25]",
                "click[Product C - $23.50]",
            ],
            "key_actions_reasoning": None
        },
        "I need a facial moisturizer for skin, preferably organic, under $25.",
        "Find a deodorant suitable for sensitive skin, with a citrus scent, under $15.",
    ))
    print(profiler.update_profile(
        {
            "key_click": [
                "search[organic multivitamins no gelatin]",
                "filter[price < $30]",
                "click[Product D - $28.99]",
            ],
            "key_actions_reasoning": None
        },
        "Look for organic multivitamins, no gelatin, budget of $30.",
        "I need a facial moisturizer for skin, preferably organic, under $25.",
    ))

    print(json.dumps(profiler.profile, indent=4))

    print(profiler.retrieve("Buy a facial cleanser."))


"""
profiler.profile.extend(
    [
        {
            "instruction": "I need a long clip-in hair extension which is natural looking, and price lower than 20.00 dollars",
            "important actions": [
                "search[long clip-in hair extension]",
                "filter[natural looking]",
                "filter[price < $20]",
                "click[Product A - $18.99]",
            ],
            "pred": [],
            "succ": [],
        },
        {
            "instruction": "Find a deodorant suitable for sensitive skin, with a citrus scent, under $15.",
            "important actions": [
                "search[citrus deodorant sensitive skin]",
                "filter[price < $15]",
                "click[Product B - $12.99]",
            ],
            "pred": [],
            "succ": [
                {
                    "situation": "Shopping for skincare products",
                    "instruction": "I need a facial moisturizer for dry skin, preferably organic, under $25.",
                    "impact on length": [-1, -1, 0],
                }
            ],
        },
        {
            "instruction": "I need a facial moisturizer for skin, preferably organic, under $25.",
            "important actions": [
                "search[organic facial moisturizer for sensitive skin]",
                "filter[price < $25]",
                "click[Product C - $23.50]",
            ],
            "pred": [
                {
                    "situation": "Purchasing sensitive skin deodorant",
                    "instruction": "Find a deodorant suitable for sensitive skin, with a citrus scent, under $15.",
                    "impact on length": [-1, -1, 0],
                }
            ],
            "succ": [
                {
                    "situation": "Buying organic health products",
                    "instruction": "Look for organic multivitamins, no gelatin, budget of $30.",
                    "impact on length": [0, -1, 0],
                }
            ],
        },
        {
            "instruction": "Look for organic multivitamins, no gelatin, budget of $30.",
            "important actions": [
                "search[organic multivitamins no gelatin]",
                "filter[price < $30]",
                "click[Product D - $28.99]",
            ],
            "pred": [
                {
                    "situation": "Selecting skincare products",
                    "instruction": "I need a facial moisturizer for skin, preferably organic, under $25.",
                    "impact on length": [0, -1, 0],
                }
            ],
            "succ": [],
        },
    ]
)
"""
