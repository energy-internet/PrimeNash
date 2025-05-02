import json
import random
import time
import matplotlib.pyplot as plt
import openai
import math
import numpy as np
import re
import pandas as pd

def clean_markdown_code(markdown_code_str):
    # Remove the ```python at the start and ``` at the end
    # Also handling potential variations in whitespace
    cleaned_code = markdown_code_str.strip().replace("```python\n", "", 1).replace("\n```", "", 1).strip()
    return cleaned_code

def execute_python_code(code_str):
    # Define the local and global dictionary to capture the execution context
    local_dict = {}
    global_dict = {}

    # Execute the code
    exec(code_str, global_dict, local_dict)

    # Return the result of the execution
    return local_dict['is_nash_equilibrium']

def classification_prompt(game_setting):
  prompt_start = """ Classify the game based on the provided description.

A classification of the game as one of the following:
'Complete Information Static Game': All participants have complete information about the game structure, other participants' strategies, and payoffs, and all decisions are made simultaneously.
 'Incomplete Information Static Game': At least one participant lacks complete information about the game structure, other participants' strategies, and payoffs, and all decisions are made simultaneously.
'Complete Information Dynamic Game': All participants have complete information about the game structure, other participants' strategies, and payoffs, and decisions are made sequentially.
'Incomplete Information Dynamic Game': At least one participant lacks complete information about the game structure, other participants' strategies, and payoffs, and decisions are made sequentially.  """

  prompt_end = """Please classify the game in the following format: 'Game': 'xx'.  'xx' can be 'Complete Information Static Game' or  'Incomplete Information Static Game' or 'Complete Information Dynamic Game' or 'Incomplete Information Dynamic Game'"""

  prompt = prompt_start + game_setting + prompt_end

  return prompt

def classification_result(game_setting):
  prompt = classification_prompt(game_setting)
  classification_response = chatgpt_agent(prompt)
  pattern_adjusted = r"'Game': '([^']*)'"

  # Extracting using the adjusted regular expression
  result_adjusted = re.search(pattern_adjusted, classification_response)
  extracted_content = result_adjusted.group(1) if result_adjusted else None

  return extracted_content