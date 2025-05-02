import json
import os
import pandas as pd

def load_prompts():
    """Load prompts configuration file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'cournot_prompts.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load prompts
PROMPTS = load_prompts()

def get_game_setting():
    """Get game setting prompt"""
    return PROMPTS['game_setting']['description']

def get_evaluation_prompt(response):
    """Get evaluation prompt"""
    return PROMPTS['evaluation']['template'].format(response=response)

def get_sr_prompt(top_responses):
    """Get SR (Solution Refinement) prompt"""
    sr_prompt = PROMPTS['sr']['template']
    
    # Add each response to the prompt
    if isinstance(top_responses, pd.DataFrame):
        for idx, row in top_responses.iterrows():
            sr_prompt += f"\nResponse {idx + 1}: {row['quantity_prompt']}"
    
    sr_prompt += "\n\nRemember: You MUST provide both the expression AND the complete Python implementation code."
    return sr_prompt

def get_verification_prompt(game_setting, sr_response):
    """Get verification prompt for Nash equilibrium"""
    verification = PROMPTS['verification']
    return verification['start'] + game_setting + verification['middle'] + sr_response + verification['end']