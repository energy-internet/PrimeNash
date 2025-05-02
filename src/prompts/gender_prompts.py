import json
import os
import re
import pandas as pd

def load_prompts():
    """Load prompts configuration file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'gender_prompts.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load prompts
PROMPTS = load_prompts()

def get_game_setting():
    """Get the base prompt for the game setting"""
    return PROMPTS['game_setting']['description']

def get_subgame_prompt(game_setting):
    """Get the prompt for the subgame"""
    return PROMPTS['subgame']['template'].format(game_setting=game_setting)

def get_complete_game_prompt(game_setting, pure_nash, mixed_nash):
    """Get the prompt for the complete game"""
    return PROMPTS['complete_game']['template'].format(
        game_setting=game_setting,
        pure_nash=pure_nash,
        mixed_nash=mixed_nash
    )

def get_evaluation_prompt(subgame_response, complete_game_response=None):
    """Get the evaluation prompt"""
    return PROMPTS['evaluation']['template'].format(
        subgame_response=subgame_response,
        complete_game_response=complete_game_response if complete_game_response else ''
    )

def generate_sr_prompt(top_responses=None, pure_nash=None, mixed_nash=None):
    """Generate SR prompt"""
    game_setting = get_game_setting()
    sr_prompt = PROMPTS['sr']['base'].format(game_setting=game_setting)
    
    if top_responses is not None:
        sr_prompt += PROMPTS['sr']['top_responses'].format(
            response1=top_responses.iloc[0]['quantity_prompt'],
            response2=top_responses.iloc[1]['quantity_prompt'],
            response3=top_responses.iloc[2]['quantity_prompt']
        )
    elif pure_nash and mixed_nash:
        sr_prompt += PROMPTS['sr']['nash_responses'].format(
            pure_nash=pure_nash,
            mixed_nash=mixed_nash
        )
    
    return sr_prompt

def gender_game(model_name, strategy_list):
    """Run the gender game simulation
    
    Args:
        model_name (str): Name of the LLM model to use
        strategy_list (list): List of strategies to evaluate
        
    Returns:
        tuple: Contains:
            - score (int): Evaluation score
            - quantity_response (str): Model's response for quantity
            - evaluation_response (str): Model's evaluation response
            - sender_strategy_list (list): List of sender strategies
            - receiver_strategy_list (list): List of receiver strategies
            - p_list (list): List of probabilities for sender
            - q_list (list): List of probabilities for receiver
            - signal_output (str): Signal output from the model
            - feedback_response (str): Feedback from the model
            - NE (bool): Whether Nash Equilibrium was found
    """
    try:
        # Get game settings
        game_setting = get_game_setting()
        
        # Get subgame solution
        subgame_prompt = get_subgame_prompt(game_setting)
        subgame_response = call_model(model_name, subgame_prompt)
        
        if not subgame_response:
            return None, None, None, None, None, None, None, None, None, None
            
        # Extract Nash equilibria from subgame
        pure_nash = extract_nash_equilibrium(
            subgame_response, 
            r"'Pure Nash Equilibrium': '(.*?)'",
            r"Pure Nash Equilibrium:\s*([^\n]*)"
        )
        mixed_nash = extract_nash_equilibrium(
            subgame_response,
            r"'Mixed Nash Equilibrium': '(.*?)'",
            r"Mixed Nash Equilibrium:\s*([^\n]*)"
        )
        
        if pure_nash and mixed_nash:
            # Get complete game solution
            complete_game_prompt = get_complete_game_prompt(game_setting, pure_nash, mixed_nash)
            complete_game_response = call_model(model_name, complete_game_prompt)
            
            if complete_game_response:
                # Evaluate the solution
                evaluation_prompt = get_evaluation_prompt(subgame_response, complete_game_response)
                evaluation_response = call_model(model_name, evaluation_prompt)
                
                if evaluation_response:
                    # Extract score and other components
                    score_match = re.search(r"'score': '(\d+)'", evaluation_response)
                    score = int(score_match.group(1)) if score_match else None
                    
                    # Extract strategies and probabilities
                    sender_strategy_list = extract_strategies(complete_game_response, "sender")
                    receiver_strategy_list = extract_strategies(complete_game_response, "receiver")
                    p_list = extract_probabilities(complete_game_response, "p")
                    q_list = extract_probabilities(complete_game_response, "q")
                    
                    # Generate signal output and feedback
                    signal_output = generate_signal_output(complete_game_response)
                    feedback_response = generate_feedback(score, complete_game_response)
                    
                    # Check if Nash Equilibrium was found
                    NE = verify_nash_equilibrium(complete_game_response)
                    
                    return (
                        score,
                        complete_game_response,  # quantity_response
                        evaluation_response,
                        sender_strategy_list,
                        receiver_strategy_list,
                        p_list,
                        q_list,
                        signal_output,
                        feedback_response,
                        NE
                    )
    
    except Exception as e:
        print(f"Error in gender game simulation: {e}")
    
    return None, None, None, None, None, None, None, None, None, None

# Helper functions for gender_game
def extract_strategies(response, player_type):
    """Extract strategies from response for given player type"""
    try:
        pattern = f"'{player_type}_strategies': '(.*?)'"
        match = re.search(pattern, response)
        return match.group(1).split(',') if match else []
    except Exception as e:
        print(f"Error extracting {player_type} strategies: {e}")
        return []

def extract_probabilities(response, prob_type):
    """Extract probabilities from response"""
    try:
        pattern = f"'{prob_type}_probabilities': '(.*?)'"
        match = re.search(pattern, response)
        return [float(p) for p in match.group(1).split(',')] if match else []
    except Exception as e:
        print(f"Error extracting {prob_type} probabilities: {e}")
        return []

def generate_signal_output(response):
    """Generate signal output based on game response"""
    try:
        pattern = r"'signal_output': '(.*?)'"
        match = re.search(pattern, response)
        return match.group(1) if match else None
    except Exception as e:
        print(f"Error generating signal output: {e}")
        return None

def generate_feedback(score, response):
    """Generate feedback based on score and response"""
    try:
        if score >= 80:
            return "Excellent solution with strong strategic reasoning"
        elif score >= 60:
            return "Good solution but could be improved"
        else:
            return "Solution needs significant improvement"
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return None

def verify_nash_equilibrium(response):
    """Verify if the solution is a Nash Equilibrium"""
    try:
        pattern = r"'is_nash_equilibrium': '(true|false)'"
        match = re.search(pattern, response.lower())
        return match.group(1) == 'true' if match else False
    except Exception as e:
        print(f"Error verifying Nash Equilibrium: {e}")
        return False

def extract_nash_equilibrium(response):
    """Extract Nash equilibrium from response"""
    try:
        pattern = r"'Nash Equilibrium': '(.*?)'"
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Error extracting Nash equilibrium: {e}")
    return None