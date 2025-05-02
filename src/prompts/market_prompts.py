import json
import os
import re
from utils.model_caller import call_model  # 添加这行

def load_prompts():
    """Load prompts configuration file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'market_prompts.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load prompts
PROMPTS = load_prompts()

def get_game_setting():
    """Get the base prompt for the game setting"""
    return PROMPTS['game_setting']['description']

def get_prior_matrix_prompt(game_setting):
    """Get the prompt for the prior matrix"""
    return PROMPTS['prior_matrix']['template'].format(game_setting=game_setting)

def get_pure_ne_prompt(game_setting, prior_matrix):
    """Get the prompt for pure strategy Nash equilibrium"""
    return PROMPTS['pure_ne']['template'].format(
        game_setting=game_setting,
        prior_matrix=prior_matrix
    )

def get_mixed_ne_prompt(game_setting, prior_matrix, pure_ne):
    """Get the prompt for mixed strategy Nash equilibrium"""
    return PROMPTS['mixed_ne']['template'].format(
        game_setting=game_setting,
        prior_matrix=prior_matrix,
        pure_ne=pure_ne
    )

def get_sr_prompt(game_setting, prior_matrix, pure_ne):
    """Get the SR prompt"""
    return PROMPTS['sr']['template'].format(
        game_setting=game_setting,
        prior_matrix=prior_matrix,
        pure_ne=pure_ne
    )

def get_evaluation_prompt(game_setting, prior_matrix_response, pure_ne_response, mixed_ne_response):
    """Get the evaluation prompt"""
    return PROMPTS['evaluation']['template'].format(
        game_setting=game_setting,
        prior_matrix_response=prior_matrix_response,
        pure_ne_response=pure_ne_response,
        mixed_ne_response=mixed_ne_response
    )

def get_fallback_prompt(game_setting, sr_response, isNE, quantity_prompt, idx):
    """Get the fallback prompt"""
    return PROMPTS['fallback']['template'].format(
        game_setting=game_setting,
        sr_response=sr_response,
        isNE=isNE,
        quantity_prompt=quantity_prompt,
        idx=idx
    )

def market_game(model_name, strategy_list):
    """Run the market game simulation
    
    Args:
        model_name (str): Name of the LLM model to use
        strategy_list (list): List of strategies to evaluate
    
    Returns:
        tuple: Contains the following elements:
            - score (int): Evaluation score
            - quantity_response (str): Model's response for quantity
            - evaluation_response (str): Model's evaluation response
            - prior_matrix_response (str): Prior matrix calculation response
            - pure_ne_response (str): Pure Nash equilibrium response
            - mixed_ne_response (str): Mixed Nash equilibrium response
    """
    try:
        # Get game settings
        game_setting = get_game_setting()
        
        # Calculate prior matrix
        prior_matrix_prompt = get_prior_matrix_prompt(game_setting)
        prior_matrix_response = call_model(model_name, prior_matrix_prompt)
        prior_matrix = extract_prior_matrix(prior_matrix_response)
        
        if prior_matrix:
            # Calculate pure strategy Nash equilibrium
            pure_ne_prompt = get_pure_ne_prompt(game_setting, prior_matrix)
            pure_ne_response = call_model(model_name, pure_ne_prompt)
            pure_ne = extract_pure_ne(pure_ne_response)
            
            if pure_ne:
                # Calculate mixed strategy Nash equilibrium
                mixed_ne_prompt = get_mixed_ne_prompt(game_setting, prior_matrix, pure_ne)
                mixed_ne_response = call_model(model_name, mixed_ne_prompt)
                mixed_ne = extract_mixed_ne(mixed_ne_response)
                
                # Evaluate results
                evaluation_prompt = get_evaluation_prompt(
                    game_setting,
                    prior_matrix_response,
                    pure_ne_response,
                    mixed_ne_response
                )
                evaluation_response = call_model(model_name, evaluation_prompt)
                score = extract_score(evaluation_response)
                
                return (
                    score,
                    mixed_ne_response,  # quantity_response
                    evaluation_response,
                    prior_matrix_response,
                    pure_ne_response,
                    mixed_ne_response
                )
                
    except Exception as e:
        print(f"Error in market game: {e}")
    
    return None, None, None, None, None, None

def extract_prior_matrix(response):
    """Extract prior matrix from response"""
    try:
        pattern = r"'(.*?)': '\((.*?)\)'"
        matches = re.findall(pattern, response)
        if matches:
            return {key: f"({value})" for key, value in matches}
    except Exception as e:
        print(f"Error extracting prior matrix: {e}")
    return None

def extract_pure_ne(response):
    """Extract pure NE from response"""
    try:
        pattern = r"'Pure NE': '(.*?)'"
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Error extracting pure NE: {e}")
    return None

def extract_mixed_ne(response):
    """Extract mixed NE from response"""
    try:
        pattern = r"'Mixed Strategy NE': '(.*?)'"
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Error extracting mixed NE: {e}")
    return None

def extract_score(response):
    """Extract score from response"""
    try:
        pattern = r"'score': '(\d+)'"
        match = re.search(pattern, response)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Error extracting score: {e}")
    return None