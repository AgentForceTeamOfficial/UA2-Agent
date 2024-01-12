INSIGHT_PROFILE_IMPACT_PROMPT = """You will be provided a trajectory of webshop task given some insights inspired from other similar trajectories. Please reason about whether the insights have a negative, positive or no impact on completing the webshop task following format:
Impact on Speed: ##
Reason: ##

Impact on Effectiveness: ##
Reason: ## 


Trajectory:
{}
Insights Used:
{}


Impact on Speed (negative, none, positive): ##
Reason: ##

Impact on Effectiveness (negative, none, positive): ##
Reason: ##      
"""


INSIGHT_REDUNDANCY_PROMPT = """You will be provided a trajectory of webshop. Note that some actions may be redundant. Please identify the key and the redundant actions, and explain the reasons.
Here is an example:

C-Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

0. Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search]
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

1. Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

2. Action: click[bright citrus]
Observation: You have clicked bright citrus. 

3. Action: click[assorted scents]
Observation: You have clicked bright citrus. 

4. Action: click[bright citrus]
Observation: You have clicked bright citrus. 

5. Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

6. Action: click[Buy Now]

insight:
key_actions_id: [0, 1, 2, 6]
reason: These actions are key because they directly contribute to the process of finding and purchasing the desired product.
redundant_actions_id: [3, 4]
reason: These actions are redundant because they are clicking on the same options ("bright citrus" and "3 ounce (pack of 1)") that were already selected in action 4 and observed in action 7.

{}

insight:
"""

INSIGHT_REDUNDANCY_PROMPT_0shot = """You will be provided a trajectory of webshop. Note that some actions may be redundant. Please identify the key and the redundant actions, and explain the reasons.
{}

insight:
key_actions_id: [#action_id#]
reason: ##
redundant_actions_id: [#action_id#]
reason: ##
"""

ReAct_PROMPT_1shot = """C-Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search]
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""


ReAct_PROMPT_1shot_Explore = """Note that your admission is not only finish the task but also 
C-Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search]
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

UPDATE_PROFILE_SYSTEM_PROMPT = """"""
RETRIEVE_SYSTEM_PROMPT = """"""

UPDATE_PROFILE_INSTRUCTION = """"""
RETRIEVE_INSTRUCTION = """"""

def get_profile_prompt(profile: dict, begin="\n\n") -> str:
    """
    profile: Profile.field

    return: str
    """
    if profile["instruction"] == "":
        return ""
    prompt = begin
    prompt += f"One past instruction: {profile['instruction']}\n"
    prompt += f"Important actions for finishing task: {profile['important actions']}\n"
    # if len(profile["succ"]) != 0:
    #     prompt += f"These actions contribute to the result of the following task:\n"
    #     for idx, succ in enumerate(profile["succ"]):
    #         prompt += f"{idx+1}. instruction: {succ['instruction']}\n"
    #         prompt += f"   each impact on the result of the actions mentioned above: {succ['impact']}\n"
    prompt += "\n\n"
    return prompt
