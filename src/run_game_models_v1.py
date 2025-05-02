import pandas as pd
import datetime
import re
import requests
import time
import anthropic
from cournot_prompts import get_game_setting as get_cournot_setting
from cournot_prompts import get_evaluation_prompt as get_cournot_evaluation
from cournot_prompts import get_sr_prompt as get_cournot_sr
from cournot_prompts import equilibrium_prompt as get_cournot_verification
from fpsba_prompts import get_game_setting as get_fpsba_setting
from fpsba_prompts import get_evaluation_prompt as get_fpsba_evaluation
from fpsba_prompts import get_sr_prompt as get_fpsba_sr
from fpsba_prompts import equilibrium_prompt as get_fpsba_verification
from stackelberg_prompts import get_game_setting as get_stackelberg_setting
from stackelberg_prompts import get_evaluation_prompt as get_stackelberg_evaluation
from stackelberg_prompts import get_sr_prompt as get_stackelberg_sr
from stackelberg_prompts import equilibrium_prompt as get_stackelberg_verification
from hawkdove_prompts import get_game_setting as get_hawkdove_setting
from hawkdove_prompts import get_evaluation_prompt as get_hawkdove_evaluation
from hawkdove_prompts import get_sr_prompt as get_hawkdove_sr
from hawkdove_prompts import equilibrium_prompt as get_hawkdove_verification
from market_prompts import (
    get_game_setting as market_get_game_setting,
    get_prior_matrix_prompt,
    get_pure_ne_prompt,
    get_mixed_ne_prompt,
    get_evaluation_prompt as market_get_evaluation_prompt,
    get_fallback_prompt,
    market_game,
    extract_prior_matrix,
    extract_pure_ne,
    extract_mixed_ne,
    extract_score
)
from gender_prompts import (
    get_game_setting,
    get_subgame_prompt,
    get_complete_game_prompt,
    get_evaluation_prompt,
    generate_sr_prompt,
    extract_nash_equilibrium,
    gender_game
)
from market_prompts import (
    get_game_setting,
    get_prior_matrix_prompt,
    get_pure_ne_prompt,
    get_mixed_ne_prompt,
    get_evaluation_prompt,
    market_game,
    extract_prior_matrix,
    extract_pure_ne,
    extract_mixed_ne,
    extract_score
)
from signaling_prompts import (
    get_game_setting as get_signaling_setting,
    get_evaluation_prompt as get_signaling_evaluation,
    get_verification_prompt as get_signaling_sr,  
    get_quantity_prompt,
    get_fallback_prompt,
    get_verification_prompt,
    generate_sr_prompt,
    benefit,
    calculate_expected_benefit,
    signal_prompt_gen,
    
    signal,
    extract_nash_equilibrium
)

from carbonmkt_prompts import (
    get_game_setting as get_carbonmkt_setting,
    get_evaluation_prompt as get_carbonmkt_evaluation,
    get_sr_prompt as get_carbonmkt_sr,
    get_period4_prompt,
    get_period3_prompt,
    get_period2_prompt,
    get_period1_prompt,
    get_final_PQ_prompt,
    equilibrium_prompt as get_carbonmkt_verification,
    solve_with_llm 
)

import sys
import os
from utils.model_caller import call_model, api_call_counts
import multiprocessing
from functools import partial

def subgame_execute_python_code(code_str):
    local_dict = {}
    global_dict = {}
    exec(code_str, global_dict, local_dict)
    return local_dict['is_nash_equilibrium']

def self_debugging_process(initial_code, max_attempts, model_name):
    current_code = initial_code
    success = False
    error_msg = None

    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1} of {max_attempts} - Executing the code...\n")

        try:
            global_dict = {
                'sp': __import__('sympy'),
                'np': __import__('numpy'),
                'math': __import__('math'),
                'pd': __import__('pandas')
            }
            local_dict = {}
            
            exec(current_code, global_dict, local_dict)
            success = True
            print("Code execution successful.")
            break
        except Exception as e:
            print("Code execution failed. Error message:")
            print(e)
            error_msg = str(e)

            prompt = f"""
Error encountered during execution on attempt {attempt + 1}.
Requesting debug...\n
{error_msg}
{current_code}
Please provide the corrected code.
Don't write code in chunks, output all code directly."""

            response_text = call_model(model_name, prompt)
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL)
            if code_blocks:
                current_code = code_blocks[0]
            else:
                print("No code block found.")
    
    return current_code


def extract_nash_equilibrium(response, pattern_1, pattern_2):
    """Extract Nash equilibrium results"""
    try:
        match = re.search(pattern_1, response)
        if match:
            return match.group(1)
        match = re.search(pattern_2, response)
        if match:
            return match.group(1)
    except:
        return None
    return None

def clean_markdown_code(markdown_code_str):
    """Clean Markdown code block"""
    if not markdown_code_str:
        return None
    try:
        code_block = re.search(r"```python\s*(.*?)\s*```", markdown_code_str, re.DOTALL)
        return code_block.group(1).strip() if code_block else None
    except Exception as e:
        print(f"Error cleaning markdown code: {e}")
        return None

def run_cournot_simulation(model_name, quantity_prompt):
    """Run Cournot model simulation"""
    try:
        # Get model response for quantity prompt
        quantity_response = call_model(model_name, quantity_prompt)
        if not quantity_response:
            return None, None, None

        # Get evaluation prompt
        evaluation_prompt = get_evaluation_prompt(quantity_response)
        
        # Get model response for evaluation prompt
        evaluation_response = call_model(model_name, evaluation_prompt)
        if not evaluation_response:
            return None, None, None

        # Extract score
        score = extract_score(evaluation_response)
        
        # Record successful API calls
        global api_call_count
        api_call_count += 2  # One call for quantity and one for evaluation

        return score, quantity_response, evaluation_response

    except Exception as e:
        print(f"Error in run_cournot_simulation: {str(e)}")
        return None, None, None

# In get_prompts function, change the game type
def get_prompts(game_type):
    if game_type == 'carbonmkt': 
        prompts = {
            'setting': get_carbonmkt_setting,
            'evaluation': get_carbonmkt_evaluation,
            'sr': get_carbonmkt_sr,
            'verification': get_carbonmkt_verification
        }
        print("Carbon Market prompts loaded:")
        for key, value in prompts.items():
            print(f"- {key}: {bool(value)}")
        return prompts
    if game_type == 'cournot':
        return {
            'setting': get_cournot_setting,
            'evaluation': get_cournot_evaluation,
            'sr': get_cournot_sr,
            'verification': get_cournot_verification
        }
    elif game_type == 'fpsba':
        return {
            'setting': get_fpsba_setting,
            'evaluation': get_fpsba_evaluation,
            'sr': get_fpsba_sr,
            'verification': get_fpsba_verification
        }
    elif game_type == 'stackelberg':  # Add Stackelberg support
        return {
            'setting': get_stackelberg_setting,
            'evaluation': get_stackelberg_evaluation,
            'sr': get_stackelberg_sr,
            'verification': get_stackelberg_verification
        }
    elif game_type == 'hawkdove':  # Add Hawk-Dove support
        return {
            'setting': get_hawkdove_setting,
            'evaluation': get_hawkdove_evaluation,
            'sr': get_hawkdove_sr,
            'verification': get_hawkdove_verification
        }
    elif game_type == 'gender':  # Modify gender support
        return {
            'setting': get_game_setting,
            'evaluation': get_evaluation_prompt,
            'sr': generate_sr_prompt,
            'subgame': get_subgame_prompt,
            'complete_game': get_complete_game_prompt,
            'game_function': gender_game,
            'extract_nash': extract_nash_equilibrium,
            'verification': lambda game_setting, sr_response: get_evaluation_prompt(  # Add verification key
                game_setting=game_setting,
                prior_matrix_response=sr_response,
                pure_ne_response=None,
                mixed_ne_response=None
            )
        }
    
    elif game_type == 'signaling':  # Add signaling support
        return {
            'setting': get_signaling_setting,
            'evaluation': get_signaling_evaluation,
            'sr': get_signaling_sr,
            'game_function': signal,
            'extract_nash': extract_nash_equilibrium,
            'verification': get_verification_prompt,
            'quantity_prompt': get_quantity_prompt,
            'fallback': get_fallback_prompt,
            'generate_sr': generate_sr_prompt,
            'benefit': benefit,
            'calculate_expected_benefit': calculate_expected_benefit,
            'signal_prompt_gen': signal_prompt_gen
        }
    
    elif game_type == 'market':  # market support remains unchanged
        return {
            'setting': get_game_setting,
            'evaluation': get_evaluation_prompt,
            'sr': get_mixed_ne_prompt,
            'game_function': market_game,
            'verification': lambda x, y: get_evaluation_prompt(x, y)
        }
    else:
        raise ValueError(f"Unsupported game type: {game_type}")

def mixed_strategy_ne(model_name, game_setting):
    """Handle mixed strategy Nash equilibrium"""
    try:
        # Get prior matrix
        prior_matrix_prompt = get_prior_matrix_prompt(game_setting)
        prior_matrix_response = call_model(model_name, prior_matrix_prompt)
        
        if prior_matrix_response:
            prior_matrix = extract_prior_matrix(prior_matrix_response)
            if prior_matrix:
                pure_ne_prompt = get_pure_ne_prompt(game_setting, prior_matrix)
                pure_ne_response = call_model(model_name, pure_ne_prompt)
                
                if pure_ne_response:
                    pure_ne = extract_pure_ne(pure_ne_response)
                    print("Pure NE extracted:", pure_ne)
                    
                    max_retries = 5
                    retry_delay = 30  
                    for attempt in range(max_retries):
                        try:
                            print("Attempting to get mixed NE response")
                            mixed_ne_prompt = get_mixed_ne_prompt(game_setting, prior_matrix, pure_ne)
                            mixed_ne_response = call_model(model_name, mixed_ne_prompt)
                            print(mixed_ne_response)
                            
                            if mixed_ne_response:
                                print("Mixed NE response received")

                                patterns = [
                                    r"'Mixed Strategy NE':\s*'([^']*)'", 
                                    r"Mixed Strategy NE:\s*([^\n]*)",    
                                    r"\(\(([^)]*)\),\s*\(([^)]*)\)\)", 
                                    r"Mixed Strategy Nash Equilibrium:\s*([^\n]*)", 
                                ]
                                
                                mixed_ne = None
                                for pattern in patterns:
                                    match = re.search(pattern, mixed_ne_response)
                                    if match:
                                        mixed_ne = match.group(1)
                                        print(f"Successfully extracted Mixed NE using pattern: {pattern}")
                                        break
                                    
                                if mixed_ne:
                                    evaluation_prompt = get_evaluation_prompt(
                                        game_setting=game_setting,
                                        prior_matrix_response=prior_matrix_response,
                                        pure_ne_response=pure_ne_response,
                                        mixed_ne_response=mixed_ne
                                    )
                                    evaluation_response = call_model(model_name, evaluation_prompt)
                                    
                                    if evaluation_response:
                                        score = extract_score(evaluation_response)
                                        return score, prior_matrix_response, evaluation_response, prior_matrix_response, pure_ne_response, mixed_ne
                                    else:
                                        print("Failed to extract Mixed NE using any pattern")
                                        
                            print(f"Attempt {attempt + 1}: Retrying after {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            
                        except Exception as e:
                            print(f"Error in attempt {attempt + 1}: {e}")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                            continue
        
        print("Failed to get complete responses")
        return None, None, None, None, None, None
        
    except Exception as e:
        print(f"Error in mixed_strategy_ne: {e}")
        return None, None, None, None, None, None


def main(model_name='gemini', game_type='cournot', num_simulations=10, top_n_responses=3, sr_runs=1):
    print(f"Starting main function with game_type: {game_type}")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    sr_evaluation_results = []
    
    if game_type == 'carbonmkt':
        print("\nExecuting Carbon Market multi-period game solution...")
        try:
            solution_results = solve_with_llm()
            
            if solution_results:
                sr_evaluation_results.append({
                    'sr_response': str(solution_results.get('final', '')),
                    'solution_code': solution_results.get('code', ''),
                    'verification_code': solution_results.get('verification', ''),
                    'run': 1
                })
                
                sr_evaluation_df = pd.DataFrame(sr_evaluation_results)
                sr_csv_filename = f"{game_type}_results_{model_name}_{current_date}.csv"
                sr_evaluation_df.to_csv(sr_csv_filename, index=False)
                
                if solution_results.get('code'):
                    solution_filename = f"{game_type}_solution_code_{model_name}_run1_{current_date}.py"
                    with open(solution_filename, 'w') as f:
                        f.write(solution_results['code'])
                    print(f"✅ Solution code saved to: {solution_filename}")
                
                print("Carbon Market solution successfully generated!")
                return
            else:
                print("Failed to generate Carbon Market solution.")
                return
                
        except Exception as e:
            print(f"Error in Carbon Market solution: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    error_count = 0  # Initialize error counter
    max_errors = 5
    retry_delay = 30
    
    # Initialize API call counter
    api_call_counts = {model_name: {'calls': 0, 'successes': 0}}
    total_calls = 0
    
    prompts = get_prompts(game_type)
    game_setting = prompts['setting']()

    for run in range(sr_runs):
        print(f"\nRunning {game_type.upper()} SR generation {run+1}/{sr_runs}...")
        
        # Set different DataFrame columns based on game type
        if game_type == 'gender':
            results_df = pd.DataFrame(columns=['score', 'response', 'evaluation_prompt', 'pure_nash', 'mixed_nash'])
        else:
            results_df = pd.DataFrame(columns=['score', 'quantity_prompt', 'evaluation_prompt'])
        
        if game_type == 'market':
            api_call_count = 0
            quantity_prompt_list = [None] * num_simulations
            calls_in_run = 0
            iteration = 0
            isNE = False
            sr_prompt = None
            
            while not isNE and iteration < 2:
                try:
                    iteration += 1
                    print(f"\nIteration {iteration}...")
                    
                    for i in range(num_simulations):
                        print(f"Running simulation {i+1}/{num_simulations} with model {model_name}...")
                        quantity_prompt = quantity_prompt_list[i]
                        if quantity_prompt != None:
                            score, quantity_response, evaluation_response = run_cournot_simulation(model_name, quantity_prompt)
                            Matrix_response, pureNE_response, mixedNE_response = None, None, None
                        else:
                            score, quantity_response, evaluation_response, Matrix_response, pureNE_response, mixedNE_response = mixed_strategy_ne(model_name, game_setting)
                        
                        api_call_counts[model_name]['calls'] += 1
                        calls_in_run += 1
                        total_calls += 1
                        
                        if score is not None:
                            new_row = {
                                'score': score,
                                'quantity_prompt': quantity_response,
                                'evaluation_prompt': evaluation_response,
                                'Matrix_response': Matrix_response,
                                'pureNE_response': pureNE_response,
                                'mixedNE_response': mixedNE_response
                            }
                            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Handle results for each iteration
                    if not results_df.empty:
                        results_df = results_df.sort_values(by='score', ascending=False)
                        top_responses = results_df.head(top_n_responses)
                        
                        # Get the best response and generate the SR prompt
                        best_response = top_responses.iloc[0]
                        sr_prompt = prompts['sr'](game_setting, best_response['Matrix_response'], best_response['pureNE_response'])
                        sr_response = call_model(model_name, sr_prompt)
                        
                        if sr_response:
                            # Verify if it is a Nash equilibrium
                            verification_prompt = prompts['verification'](game_setting, sr_response)
                            verification_response = call_model(model_name, verification_prompt)
                            
                            if verification_response:
                                isNE = True  # Set isNE to True if verification is successful
                                sr_evaluation_results.append({
                                    'sr_response': sr_response,
                                    'verification_response': verification_response,
                                    'run': run + 1,
                                    'iteration': iteration
                                })
                                break
                            else:
                                print(f"\nSR Response {run+1} is NOT a Nash Equilibrium. Retrying...")
                                
                                # Generate fallback prompts
                                fallback_prompts = []
                                for idx in range(top_n_responses):
                                    fallback_prompt = get_fallback_prompt(
                                        game_setting,
                                        sr_response,
                                        isNE,
                                        top_responses.iloc[idx]['quantity_prompt'],
                                        idx
                                    )
                                    
                                    # Call the model to generate a fallback response
                                    fallback_response = call_model(model_name, fallback_prompt)
                                    
                                    # Replace quantity_prompts with the new fallback response
                                    quantity_prompt_list[idx] = get_quantity_prompt(
                                        game_setting,
                                        top_responses.iloc[idx]['quantity_prompt'],
                                        fallback_response,
                                        idx
                                    )
                                    
                                    isNE = False
                                
                                try:
                                    sr_evaluation_results_draft.append({
                                        'sr_response': sr_response,
                                        'evaluation_result': evaluation_result,
                                        'iteration': iteration,
                                        'run': run
                                    })
                                except:
                                    sr_evaluation_results_draft.append({
                                        'sr_response': sr_response,
                                        'evaluation_result': '',
                                        'iteration': iteration,
                                        'run': run
                                    })
                                    
                except Exception as e:
                    print(f"Error in iteration {iteration}: {e}")
                    break

        elif game_type == 'signaling':
            for run in range(sr_runs):
                print(f"\nStarting run {run + 1}/{sr_runs}")
                
                # Initialize counters for this SR run
                calls_in_run = 0
                iteration = 0
                isNE = False
                sr_prompt = None
                quantity_prompt_list = [None] * num_simulations
                quantity_response = None
                last_successful_response = None
                
                while not isNE and iteration < 1:
                    iteration += 1
                    print(f"\nIteration {iteration}...")
                    strategy_list = ['(m2, m1)']
                    simulation_success = False
                    
                    # Run simulations
                    for i in range(num_simulations):
                        try:
                            print(f"Running simulation {i+1}/{num_simulations}...")
                            quantity_prompt = quantity_prompt_list[i]
                            
                            if quantity_prompt is not None:
                                score, temp_quantity_response, evaluation_response = run_cournot_simulation(model_name, quantity_prompt)
                                sender_strategy_list = receiver_strategy_list = p_list = q_list = signal_output = feedback_response = NE = None
                            else:
                                result = prompts['game_function'](model_name, strategy_list)
                                if isinstance(result, tuple) and len(result) == 10:
                                    score, temp_quantity_response, evaluation_response, sender_strategy_list, receiver_strategy_list, p_list, q_list, signal_output, feedback_response, NE = result
                                else:
                                    print(f"Invalid result format: {result}")
                                    continue
                            
                            if score is not None:  # Simulation successful
                                simulation_success = True
                                quantity_response = temp_quantity_response
                                last_successful_response = {
                                    'score': score,
                                    'response': temp_quantity_response,
                                    'evaluation': evaluation_response,
                                    'sender_strategy': sender_strategy_list,
                                    'receiver_strategy': receiver_strategy_list,
                                    'signal_output': signal_output
                                }
                            
                            api_call_counts[model_name]['calls'] += 1
                            calls_in_run += 1
                            total_calls += 1
                            
                        except Exception as e:
                            print(f"Error in simulation {i+1}: {e}")
                            continue
                    
                    if not simulation_success and last_successful_response:
                        quantity_response = last_successful_response['response']
                        simulation_success = True
                    
                    if simulation_success:  # Continue only if there was a successful simulation
                        # Generate SR prompt and get response
                        game_setting = prompts['setting']()  # Ensure you get game_setting
                        if game_setting and quantity_response:  # Ensure both are not None
                            sr_prompt = prompts['sr'](strategy_list[0], quantity_response)
                            sr_response = call_model(model_name, sr_prompt)
                            
                            if sr_response:
                                # Use generate_sr_prompt to generate a more detailed SR analysis
                                detailed_sr = prompts['generate_sr'](sr_response, game_setting)
                                detailed_sr_response = call_model(model_name, detailed_sr)
                                
                                # Verify if it is a Nash equilibrium
                                verification_prompt = prompts['verification'](game_setting, sr_response)
                                verification_result = call_model(model_name, verification_prompt)
                                
                                if verification_result:
                                    print(f"\nVerification Result: {verification_result}")
                                    
                                    # Extract code block from sr_response
                                    solution_code = clean_markdown_code(sr_response)
                                    verification_code = clean_markdown_code(verification_result)
                                    
                                    # Save complete SR evaluation results
                                    sr_evaluation_results.append({
                                        'sr_response': sr_response,
                                        'detailed_sr': detailed_sr_response,
                                        'verification_result': verification_result,
                                        'solution_code': solution_code,  # Add solution_code
                                        'verification_code': verification_code,  # Add verification_code
                                        'iteration': iteration,
                                        'run': run
                                    })
                                    evaluation_result = compare_standard_answer(sr_response, standard_answer)
                                    print(f"\nComparison with standard answer: {evaluation_result}")
                                    
                                    api_call_counts[model_name]['successes'] += 1
                                    sr_evaluation_results.append({
                                        'sr_response': sr_response,
                                        'verification_result': verification_result,
                                        'evaluation_result': evaluation_result,
                                        'iteration': iteration,
                                        'run': run
                                    })
                                    break
                                else:
                                    print(f"\nSR Response {run+1} is NOT a Nash Equilibrium. Retrying...")
                                    
                                    # Generate fallback prompts
                                    for idx in range(top_n_responses):
                                        fallback_prompt = prompts['fallback'](
                                            game_setting,
                                            sr_response,
                                            isNE,
                                            quantity_response,
                                            idx
                                        )
                                        
                                        fallback_response = call_model(model_name, fallback_prompt)
                                        quantity_prompt_list[idx] = prompts['quantity_prompt'](
                                            game_setting,
                                            quantity_response,
                                            fallback_response,
                                            idx,
                                            strategy_list[0]
                                        )
                                        
                                        isNE = False
                                    
                                    try:
                                        sr_evaluation_results.append({
                                            'sr_response': sr_response,
                                            'evaluation_result': evaluation_response,
                                            'iteration': iteration,
                                            'run': run
                                        })
                                    except:
                                        sr_evaluation_results.append({
                                            'sr_response': sr_response,
                                            'evaluation_result': '',
                                            'iteration': iteration,
                                            'run': run
                                        })
                
                print(f"Run {run+1} API call statistics:")
                print(f"Total API calls in this run: {calls_in_run}")

        # Remove duplicate elif game_type == 'market' section
        elif game_type == 'gender':
            for i in range(num_simulations):
                try:
                    print(f"Running simulation {i+1}/{num_simulations}...")
                    response = call_model(model_name, game_setting)
                    if not response:
                        continue
                    
                    # Extract pure_nash and mixed_nash from the response
                    pure_nash = extract_nash_equilibrium(
                        response=response,
                        pattern_1=r"'Pure Nash Equilibrium': '([^']*)'",
                        pattern_2=r"Pure Nash Equilibrium:\s*([^\n]*)"
                    )
                    mixed_nash = extract_nash_equilibrium(
                        response=response,
                        pattern_1=r"'Mixed Nash Equilibrium': '([^']*)'",
                        pattern_2=r"Mixed Nash Equilibrium:\s*([^\n]*)"
                    )
                    
                    # Adjust the way evaluation_prompt is called based on the game type
                    if game_type == 'hawkdove':
                        evaluation_prompt = prompts['evaluation'](response)
                    else:
                        evaluation_prompt = prompts['evaluation'](
                            game_setting=game_setting,
                            prior_matrix_response=response,
                            pure_ne_response=pure_nash,
                            mixed_ne_response=mixed_nash
                        )
                    evaluation_response = call_model(model_name, evaluation_prompt)
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error {error_count}/{max_errors}: {str(e)}")
                    if error_count >= max_errors:
                        print("Maximum error count reached, terminating...")
                        break
                    time.sleep(retry_delay)
                    continue                
                if not evaluation_response:
                    continue
                    
                score_match = re.search(r"'score': '([0-9]+)'", evaluation_response)
                if score_match:
                    score = int(score_match.group(1))
                    new_row = {
                        'score': score,
                        'quantity_prompt': response,
                        'evaluation_prompt': evaluation_response
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        else:
            for i in range(num_simulations):
                print(f"Running simulation {i+1}/{num_simulations}...")
                response = call_model(model_name, game_setting)
                if not response:
                    continue
                
                # Extract pure_nash and mixed_nash from the response
                pure_nash = extract_nash_equilibrium(response, r"'Pure Strategy Nash Equilibrium': '([^']*)'", r"Pure Strategy Nash Equilibrium:\s*([^\n]*)")
                mixed_nash = extract_nash_equilibrium(response, r"'Mixed Strategy Nash Equilibrium': '([^']*)'", r"Mixed Strategy Nash Equilibrium:\s*([^\n]*)")
                
                # Gender games only need one parameter
                evaluation_prompt = prompts['evaluation'](response)  # The evaluation for gender only needs response
                evaluation_response = call_model(model_name, evaluation_prompt)
                
                if not evaluation_response:
                    continue
                    
                score_match = re.search(r"'score': '([0-9]+)'", evaluation_response)
                if score_match:
                    score = int(score_match.group(1))
                    new_row = {
                        'score': score,
                        'quantity_prompt': response,
                        'evaluation_prompt': evaluation_response
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        
        if not results_df.empty:
            results_df = results_df.sort_values(by='score', ascending=False)
            top_responses = results_df.head(top_n_responses)
            
            if game_type == 'gender':
                # Keep SR handling the same for gender games
                best_response = top_responses.iloc[0]
                sr_prompt = prompts['sr'](None, best_response['pure_nash'], best_response['mixed_nash'])
            elif game_type == 'market':
                # SR handling for market games
                best_response = top_responses.iloc[0]
                sr_prompt = prompts['sr'](game_setting, best_response['Matrix_response'], best_response['pureNE_response'])
            else:
                sr_prompt = prompts['sr'](top_responses)
            
            sr_response = call_model(model_name, sr_prompt)
            
            print("\n=== Model Response ===")
            print(sr_response)
            print("=== End of Model Response ===\n")
            
            if sr_response is None:
                print("Failed to get response from model, skipping this run...")
                continue
            
            expression_match = re.search(r"'the general expression of Nash equilibrium under the current game settings': '(.*?)'", sr_response)
            code_match = re.search(r"```python\n(.*?)```", sr_response, re.DOTALL)
            
            if code_match:
                original_code = code_match.group(1)
                print("\nDebugging generated code...")
                debugged_code = self_debugging_process(original_code, 10, model_name)
                
                if debugged_code:
                    verification_response = call_model(model_name, prompts['verification'](game_setting, sr_response))
                    if verification_response:
                        try:
                            verification_code = clean_markdown_code(verification_response)
                            debugged_verification_code = self_debugging_process(verification_code, 10, model_name)
                            
                            sr_evaluation_results.append({
                                'sr_response': sr_response,
                                'expression': expression_match.group(1) if expression_match else None,
                                'solution_code': debugged_code,
                                'verification_code': debugged_verification_code,
                                'run': run + 1
                            })
                            print("Code debugging successful!")
                        except Exception as e:
                            print(f"Error processing verification code: {e}")
                            continue
                    else:
                        print("No code block found in response.")
                        continue
                else:
                    print("Code debugging failed, requesting new solution...")
                    continue
            else:
                print("No code block found in response.")
                continue
    
    # Save results
    sr_evaluation_df = pd.DataFrame(sr_evaluation_results)
    sr_csv_filename = f"{game_type}_results_{model_name}_{current_date}.csv"
    sr_evaluation_df.to_csv(sr_csv_filename, index=False)
    
    for idx, row in sr_evaluation_df.iterrows():
        if row['solution_code']:
            solution_filename = f"{game_type}_solution_code_{model_name}_run{row['run']}_{current_date}.py"
            try:
                with open(solution_filename, 'w') as f:
                    f.write(row['solution_code'])
                print(f"✅ Solution code successfully saved to: {solution_filename}")
            except Exception as e:
                print(f"❌ Failed to save solution code: {e}")
        
        if row['verification_code']:
            verification_filename = f"{game_type}_verification_code_{model_name}_run{row['run']}_{current_date}.py"
            try:
                with open(verification_filename, 'w') as f:
                    f.write(row['verification_code'])
                print(f"✅ Verification code successfully saved to: {verification_filename}")
            except Exception as e:
                print(f"❌ Failed to save verification code: {e}")
        
        if row['solution_code']:
            try:
                with open(solution_filename, 'r') as f:
                    content = f.read()
                    if content:
                        print(f"✓ Confirmed solution code file content: {len(content)} characters")
            except Exception as e:
                print(f"! Unable to read solution code file: {e}")
                
        if row['verification_code']:
            try:
                with open(verification_filename, 'r') as f:
                    content = f.read()
                    if content:
                        print(f"✓ Confirmed verification code file content: {len(content)} characters")
            except Exception as e:
                print(f"! Unable to read verification code file: {e}")
    
    print(f"\nResults saved to: {sr_csv_filename}")

def clean_markdown_code(markdown_code_str):
    """Clean Markdown code block"""
    if not markdown_code_str:
        return None
    try:
        code_block = re.search(r"```python\s*(.*?)\s*```", markdown_code_str, re.DOTALL)
        return code_block.group(1).strip() if code_block else None
    except Exception as e:
        print(f"Error cleaning markdown code: {e}")
        return None

def run_cournot_simulation(model_name, quantity_prompt):
    """Run Cournot model simulation"""
    try:
        # Get model response for quantity prompt
        quantity_response = call_model(model_name, quantity_prompt)
        if not quantity_response:
            return None, None, None

        # Get evaluation prompt
        evaluation_prompt = get_evaluation_prompt(quantity_response)
        
        # Get model response for evaluation prompt
        evaluation_response = call_model(model_name, evaluation_prompt)
        if not evaluation_response:
            return None, None, None

        # Extract score
        score = extract_score(evaluation_response)
        
        # Record successful API calls
        global api_call_count
        api_call_count += 2  # One call for quantity and one for evaluation

        return score, quantity_response, evaluation_response

    except Exception as e:
        print(f"Error in run_cournot_simulation: {str(e)}")
        return None, None, None

# Add signaling test in __main__
if __name__ == "__main__":
        # Test Gender model
    # print("\nTesting Gender model...")
    # main(game_type='gender', num_simulations=10, sr_runs=1)
    
    # # Test Hawk-Dove model
    # print("\nTesting Hawk-Dove model...")
    # main(game_type='hawkdove', num_simulations=10, sr_runs=1)
    
    # Test Cournot model
    # print("Testing Cournot model...")
    # main(game_type='cournot', num_simulations=10, sr_runs=1)
    
    # Test FPSBA model
    # print("\nTesting FPSBA model...")
    # main(game_type='fpsba', num_simulations=10, sr_runs=1)
    
    # Test Stackelberg model
    # print("\nTesting Stackelberg model...")
    # main(game_type='stackelberg', num_simulations=10, sr_runs=1)
    
    # # Add Market model test
    # print("\nTesting Market model...")
    # main(game_type='market', num_simulations=10, sr_runs=1)
    
    # Test Signaling model
    # print("\nTesting Signaling model...")
    # main(game_type='signaling', num_simulations=10, sr_runs=1)
    
    # Test Carbon Market model
    print("\nTesting Carbon Market model...")
    main(game_type='carbonmkt', num_simulations=10, sr_runs=1)
    
