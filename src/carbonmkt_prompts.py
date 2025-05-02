import sys
import os
from utils.model_caller import call_model, api_call_counts
import re
import traceback
import sympy as sp

def get_game_setting():
    """Get carbon market game settings"""
    return """
Multi-Period Carbon Market Game Model:

Players:
- 4 enterprises (i = 1,2,3,4)
- 4 trading periods (j = 1,2,3,4)

Enterprise Characteristics:
1. Price elasticity coefficient (ai)
2. Initial quota allocation (Ai)
3. Trading volume per period (bij)
4. Opportunity benefit rate for remaining quota at end of period 4 (r)

Key Equations:
1. Price Formation Mechanism:
   Pj = -sum_bkj[j] / (a1 + a2 + a3 + a4)
   where sum_bkj[j] is the sum of all trading volumes in period j

2. Actual Trading Volume:
   Qij = ai * Pj + bij

3. Enterprise Payoff Function includes:
   a) Trading profits: sum(Qij * Pj), j=1 to 4
   b) Remaining quota benefit: (1+r)P4 * (Ai - sum(Qij))

4. Constraints:
   - Quota constraint: sum(Qij) ≤ Ai
   - Each enterprise must decide trading volume bij in each period

Game Features:
1. Dynamic game: Multi-period decision making
2. Complete information: All parameters known to all parties
3. Non-cooperative game: Enterprises maximize individual profits
4. Constrained optimization: Subject to quota restrictions

Solution Objectives:
1. Find optimal trading strategy for each enterprise in each period
2. Determine equilibrium price sequence
3. Calculate equilibrium trading volumes

Some parameters are defined as follows:

from sympy import symbols, Eq, diff, solve

# 定义变量
a1, a2, a3, a4 = symbols('a1 a2 a3 a4')
A1, A2, A3, A4 = symbols('A1 A2 A3 A4')
b11, b12, b13, b14 = symbols('b11 b12 b13 b14')
b21, b22, b23, b24 = symbols('b21 b22 b23 b24')
b31, b32, b33, b34 = symbols('b31 b32 b33 b34')
b41, b42, b43, b44 = symbols('b41 b42 b43 b44')
P4, r = symbols('P4 r')
lambda1, lambda2, lambda3, lambda4 = symbols('lambda1 lambda2 lambda3 lambda4')

"""

def get_lagrangian_prompt():
    return """
Let's solve for the analytical solution of a carbon market model with 4 enterprises over 4 periods.

Given conditions:
- 4 enterprises (i=1,2,3,4)
- Each enterprise has:
  * Price elasticity coefficient (ai)
  * Initial quota (Ai)
  * Trading volume in each period (bij)
  * Opportunity benefit from remaining quota at end of period 4

Key equations:
1. Price formation: Pj = -sum_bkj[j] / a_sum
   where: a_sum = sum(a) = a1 + a2 + a3 + a4
   and sum_bkj[j] = sum of all trading volumes in period j

2. Trading volume: Qij = ai * Pj + bij

3. Payoff function for enterprise i includes:
   a) Trading profits: sum(Qij * Pj) for j=1 to 4
   b) Opportunity benefit: (1+r)P4 * (Remaining quota)
   where Remaining quota = Ai - sum(Qij) for j=1 to 4

4. Constraint for enterprise i:
   sum(Qij) ≤ Ai for j=1 to 4

Let's construct the Lagrangian function for each enterprise:
Li = [sum(Qij * Pj) for j=1 to 4] + (1+r)P4 * (Ai - sum(Qij)) + λi(sum(Qij)-Ai)
**Special attention is**:  + λi(sum(Qij)-Ai), not + λi(Ai - λi(sum(Qij))
Note that you need to substitute the expressions of P and Q represented by A, a, and b into L.

Please provide the symbolic Lagrangian functions for all enterprises using SymPy notation.
Use the following code structure:
1. Define all symbolic variables (including r for interest rate)
2. Set up the price formation equations
3. Set up the trading volume equations
4. Calculate remaining quotas
5. Construct the Lagrangian functions with both trading profits and opportunity benefits
"""

def get_period4_prompt(lagrangian_response):
    return lagrangian_response + """
Using backward induction, let's solve Period 4 first.

To find b14, b24, b34, b44:
1. Take partial derivatives of each enterprise's Lagrangian with respect to their period 4 trading volume:
   ∂Li/∂bi4 = 0 for i=1,2,3,4

2. This gives us 4 equations. Solve them simultaneously to get the analytical expressions for:
   b14, b24, b34, b44

Please provide the analytical solution without numerical substitution.
"""

def get_period3_prompt(previous_prompts, period4_response):
    return previous_prompts + period4_response + """
Now let's solve Period 3 using the analytical results from Period 4.

To find b13, b23, b33, b43:
1. Substitute the expressions for b14, b24, b34, b44 into the Lagrangian functions
2. Take partial derivatives with respect to period 3 trading volumes:
   ∂Li/∂bi3 = 0 for i=1,2,3,4
3. Solve the resulting system of equations

Please provide the analytical expressions for b13, b23, b33, b43 in terms of model parameters (ai, Ai, λi).
"""

def get_period2_prompt(previous_prompts, period3_response):
    return previous_prompts + period3_response + """
Now for Period 2, using the analytical solutions from Periods 3 and 4.

To find b12, b22, b32, b42:
1. Substitute all known expressions for periods 3 and 4
2. Take partial derivatives with respect to period 2 trading volumes:
   ∂Li/∂bi2 = 0 for i=1,2,3,4
3. Solve the resulting system of equations

Please provide the analytical expressions for b12, b22, b32, b42 in terms of model parameters.
"""

def get_period1_prompt(previous_prompts, period2_response):
    return previous_prompts + period2_response + """
Finally, let's solve Period 1 using all previous analytical solutions.

To find b11, b21, b31, b41:
1. Substitute all known expressions from periods 2, 3, and 4
2. Take partial derivatives with respect to period 1 trading volumes:
   ∂Li/∂bi1 = 0 for i=1,2,3,4
3. Solve the resulting system of equations

Please provide the analytical expressions for b11, b21, b31, b41 in terms of model parameters.
"""

def get_simplified_prompt(previous_prompts, period1_response):
# def get_final_PQ_prompt(previous_prompts, period1_response):
    return previous_prompts + period1_response + """
Note: Because the model is symmetric across firms and Periods 1–3 are structurally identical, we can greatly simplify:

1. Notice that Periods 1, 2, and 3 are completely symmetric in our model. We can therefore enforce symmetry by setting the trading volumes in Periods 2 and 3 equal to those in Period 1:
Add this code at the beginning
```python
from sympy import symbols, Eq, diff, solve
import sympy as sp
from multiprocessing import Pool, cpu_count, current_process

# Symmetry simplification: use Period 1 variables for Periods 2 and 3
b12, b13 = b11, b11
b22, b23 = b21, b21
b32, b33 = b31, b31
b42, b43 = b41, b41

so **Solve only Period 4 and Period 1**  
   - First‐order conditions for Period 4: b14, b24, b34, b44  
   - First‐order conditions for Period 1: b11, b21, b31, b41

2. **Simplify other b’s by substitution**  
After solving bi1 or bi4, ​​we only need to simplify b11 and b14, and then the rest can be directly obtained by rotation symmetry
   Use the predefined rules `subs_1to_temp`, `subs_temp_to_2`, `subs_temp_to_3`, `subs_temp_to_4`.  
   For example:
   ```python
A_temp1, A_temp2, A_temp3, A_temp4 = sp.symbols('A_temp1 A_temp2 A_temp3 A_temp4')
a_temp1, a_temp2, a_temp3, a_temp4 = sp.symbols('a_temp1 a_temp2 a_temp3 a_temp4')
b_temp11, b_temp21, b_temp31, b_temp41 = sp.symbols('b_temp11 b_temp21 b_temp31 b_temp41')
b_temp12, b_temp22, b_temp32, b_temp42 = sp.symbols('b_temp12 b_temp22 b_temp32 b_temp42')
b_temp13, b_temp23, b_temp33, b_temp43 = sp.symbols('b_temp13 b_temp23 b_temp33 b_temp43')
b_temp14, b_temp24, b_temp34, b_temp44 = sp.symbols('b_temp14 b_temp24 b_temp34 b_temp44')
lambda_temp1, lambda_temp2, lambda_temp3, lambda_temp4 = sp.symbols('lambda_temp1 lambda_temp2 lambda_temp3 lambda_temp4')

subs_1to_temp = [
    (A1, A_temp1), (A2, A_temp2), (A3, A_temp3), (A4, A_temp4),
    (a1, a_temp1), (a2, a_temp2), (a3, a_temp3), (a4, a_temp4),
    (b11, b_temp11), (b21, b_temp21), (b31, b_temp31), (b41, b_temp41),
    (b12, b_temp12), (b22, b_temp22), (b32, b_temp32), (b42, b_temp42),
    (b13, b_temp13), (b23, b_temp23), (b33, b_temp33), (b43, b_temp43),
    (b14, b_temp14), (b24, b_temp24), (b34, b_temp34), (b44, b_temp44),
    (lambda1, lambda_temp1), (lambda2, lambda_temp2), (lambda3, lambda_temp3), (lambda4, lambda_temp4),
]

subs_temp_to_2 = [
    (A_temp1, A2), (A_temp2, A3), (A_temp3, A4), (A_temp4, A1),
    (a_temp1, a2), (a_temp2, a3), (a_temp3, a4), (a_temp4, a1),
    (b_temp11, b21), (b_temp21, b31), (b_temp31, b41), (b_temp41, b11),
    (b_temp12, b22), (b_temp22, b32), (b_temp32, b42), (b_temp42, b12),
    (b_temp13, b23), (b_temp23, b33), (b_temp33, b43), (b_temp43, b13),    
    (b_temp14, b24), (b_temp24, b34), (b_temp34, b44), (b_temp44, b14),
    (lambda_temp1, lambda2), (lambda_temp2, lambda3), (lambda_temp3, lambda4), (lambda_temp4, lambda1),
]

subs_temp_to_3 = [
    (A_temp1, A3), (A_temp2, A4), (A_temp3, A1), (A_temp4, A2),
    (a_temp1, a3), (a_temp2, a4), (a_temp3, a1), (a_temp4, a2),
    (b_temp11, b31), (b_temp21, b41), (b_temp31, b11), (b_temp41, b21),
    (b_temp12, b32), (b_temp22, b42), (b_temp32, b12), (b_temp42, b22),
    (b_temp13, b33), (b_temp23, b43), (b_temp33, b13), (b_temp43, b23),
    (b_temp14, b34), (b_temp24, b44), (b_temp34, b14), (b_temp44, b24),
    (lambda_temp1, lambda3), (lambda_temp2, lambda4), (lambda_temp3, lambda1), (lambda_temp4, lambda2),
]

subs_temp_to_4 = [
    (A_temp1, A4), (A_temp2, A1), (A_temp3, A2), (A_temp4, A3),
    (a_temp1, a4), (a_temp2, a1), (a_temp3, a2), (a_temp4, a3),
    (b_temp11, b41), (b_temp21, b11), (b_temp31, b21), (b_temp41, b31),
    (b_temp12, b42), (b_temp22, b12), (b_temp32, b22), (b_temp42, b32),
    (b_temp13, b43), (b_temp23, b13), (b_temp33, b23), (b_temp43, b33),
    (b_temp14, b44), (b_temp24, b14), (b_temp34, b24), (b_temp44, b34),
    (lambda_temp1, lambda4), (lambda_temp2, lambda1), (lambda_temp3, lambda2), (lambda_temp4, lambda3),
]

e.g. (For reference only, please note that the corresponding parameter representation is replaced)
Pay attention to unifying the parameters. I only provide an example

solutions_4 = solve((Eq(diff(Lagrangians[0], b[0][3]), 0), Eq(diff(Lagrangians[1], b[1][3]), 0), Eq(diff(Lagrangians[2], b[2][3]), 0), Eq(diff(Lagrangians[3], b[3][3]), 0)), (b14, b24, b34, b44), dict=True)
b1_4_sol = solutions_4[0][b14].simplify()
subs_4 = {
         b14: b1_4_sol, 
         b24: b1_4_sol.subs(subs_1to_temp).subs(subs_temp_to_2), 
         b34: b1_4_sol.subs(subs_1to_temp).subs(subs_temp_to_3), 
         b44: b1_4_sol.subs(subs_1to_temp).subs(subs_temp_to_4), 
         }

import sympy as sp
from multiprocessing import Pool, cpu_count, current_process

Lagrangians_subs4 = Lagrangians[0].subs(subs_4).as_ordered_terms()

simplified_terms = [term.simplify() for term in Lagrangians_subs4]

Lagrangians_subs4_simplified_expr = sp.Add(*simplified_terms)

solutions_3 = solve((Eq(diff(Lagrangians_subs4_simplified_expr, b[0][0]), 0), Eq(diff(Lagrangians_subs4_simplified_expr.subs(subs_1to_temp).subs(subs_temp_to_2), b[1][0]), 0),  Eq(diff(Lagrangians_subs4_simplified_expr.subs(subs_1to_temp).subs(subs_temp_to_3), b[2][0]), 0),  Eq(diff(Lagrangians_subs4_simplified_expr.subs(subs_1to_temp).subs(subs_temp_to_4), b[3][0]), 0)), (b11, b21, b31, b41), dict=True)
b1_1_sol = solutions_3[0][b11].simplify()

Remember: Pay attention to unifying the parameters. I only provide an example
remember import 'simplify_expression' function
Need to solve the expressions of the first and fourth periods
It is necessary to pay attention to the construction logic of the previous code. This is a dynamic game, so the b of the last period should be solved first, then the b of the last period should be substituted into the expression, and then the first period should be solved.
The rotation symmetry of b can be used at the beginning.
Note that bi4 contains bi1. After solving bi1, remember to substitute the expression of bi1 back into bi4 to ensure that bi4 does not contain bi1.
Finally, output a complete solution code. Do not output additional code. Just write it all at once, do not write it in blocks.
"""

def get_final_PQ_prompt(previous_prompts, simplified_response):
    # return previous_prompts + simplified_response + """
    return simplified_response + """
Now that we have analytical solutions for all bij, let's derive the final expressions:

1. Price expressions for each period:
   Pj = -sum_bkj[j] / (a1 + a2 + a3 + a4) for j=1,2,3,4

2. Trading volume expressions for each enterprise in each period:
   Qij = ai * Pj + bij for i,j=1,2,3,4

Please provide:
1. The simplified analytical expressions for all Pj
2. The simplified analytical expressions for all Qij
3. Express all results in terms of the model parameters (ai, Ai, λi) only

Note that bi4 contains bi1. After solving bi1, remember to substitute the expression of bi1 back into bi4 to ensure that bi4 does not contain bi1.
Use SymPy's simplify() function to get the most concise forms possible.
Finally, output a complete solution code. Do not output additional code. Just write it all at once, do not write it in blocks.
"""

def execute_code(code_str):
    """Execute code and return success status and error message"""
    try:
        exec(code_str)
        return True, None
    except Exception as e:
        error_msg = traceback.format_exc()
        return False, error_msg

def extract_code_from_response(response_text):
    """Extract Python code blocks from response"""
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL)
    return code_blocks[0] if code_blocks else response_text

def get_debugged_code(attempt, original_code, error_message, model_name='gemini-2.0-flash'):
    """Get debugged code from LLM"""
    prompt = f"""
Error encountered during execution on attempt {attempt}.
Requesting debug...

Error message:
{error_message}

Original code:
{original_code}

Please provide the corrected code. Output only the complete corrected code within a Python code block (```python ```).
Do not include any explanations or comments outside the code block.
Make sure to include all necessary imports and maintain the complete structure of the code.
"""
    response = call_model('gemini', prompt)
    return extract_code_from_response(response)

def self_debugging_process(initial_code, max_attempts=5):
    """Process code with self-debugging capability"""
    current_code = initial_code
    
    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1} of {max_attempts} - Executing the code...")
        
        success, error_msg = execute_code(current_code)
        
        if success:
            print("Code executed successfully!")
            return current_code
        else:
            print(f"Error encountered during execution on attempt {attempt + 1}:")
            print(error_msg)
            
            current_code = get_debugged_code(attempt + 1, current_code, error_msg)
            
            if not current_code:
                print("Failed to get debugged code from LLM")
                break
    
    print("Reached maximum number of debugging attempts. The code may still have issues.")
    return current_code

def process_llm_response(response, step_name):
    """Process LLM response to extract and debug code"""
    print(f"\nProcessing {step_name}...")
    
    # Extract code from response
    code = extract_code_from_response(response)
    if not code:
        print(f"No code found in {step_name} response")
        return None
    print(code)
    # Debug and execute the code
    print(f"Debugging {step_name} code...")
    debugged_code = self_debugging_process(code)
    
    return debugged_code

def solve_with_llm():
    """Execute the complete carbon market solution process"""
    try:
        all_code_pieces = []
        results = {}
        
        # Step 1: Get Lagrangian functions
        print("\nStep 1: Generating Lagrangian functions...")
        lagrangian_prompt = get_lagrangian_prompt()
        lagrangian_response = call_model('gemini', lagrangian_prompt)
        lagrangian_code = extract_code_from_response(lagrangian_response)
        if lagrangian_code:
            all_code_pieces.append(lagrangian_code)
            results['lagrangian'] = lagrangian_code
        
        # Step 2: Solve Period 4
        print("\nStep 2: Solving Period 4...")
        period4_prompt = get_period4_prompt(lagrangian_response)
        period4_response = call_model('gemini', period4_prompt)
        period4_code = extract_code_from_response(period4_response)
        if period4_code:
            all_code_pieces.append(period4_code)
            results['period4'] = period4_code
            
        # Step 3: Solve Period 3
        print("\nStep 3: Solving Period 3...")
        period3_prompt = get_period3_prompt(period4_prompt, period4_response)
        period3_response = call_model('gemini', period3_prompt)
        period3_code = extract_code_from_response(period3_response)
        if period3_code:
            all_code_pieces.append(period3_code)
            results['period3'] = period3_code
            
        # Step 4: Solve Period 2
        print("\nStep 4: Solving Period 2...")
        period2_prompt = get_period2_prompt(period3_prompt, period3_response)
        period2_response = call_model('gemini', period2_prompt)
        period2_code = extract_code_from_response(period2_response)
        if period2_code:
            all_code_pieces.append(period2_code)
            results['period2'] = period2_code
            
        # Step 5: Solve Period 1
        print("\nStep 5: Solving Period 1...")
        period1_prompt = get_period1_prompt(period2_prompt, period2_response)
        period1_response = call_model('gemini', period1_prompt)
        period1_code =extract_code_from_response(period1_response)
        if period1_code:
            all_code_pieces.append(period1_code)
            results['period1'] = period1_code

        # Step 6: Simplify the process
        print("\nStep 6: Simplify the process...")
        simplified_prompt = get_simplified_prompt(period1_prompt, period1_response)
        simplified_response = call_model('gemini', simplified_prompt)
        simplified_code =process_llm_response(simplified_response, "Simplify expressions")
        if simplified_code:
            all_code_pieces.append(simplified_code)
            results['simplified'] = simplified_code
            
        # Step 6: Get final price and quantity expressions
        print("\nStep 7: Generating final expressions...")
        final_prompt = get_final_PQ_prompt(simplified_prompt, simplified_response)
        final_response = call_model('gemini', final_prompt)
        final_code = process_llm_response(final_response, "Final expressions")
        print(f"Final code: {final_code}")  # Add this line to print the final code
        if final_code:
            all_code_pieces.append(final_code)
            results['final'] = final_code
            
        # Combine all code pieces
        complete_solution = "\n\n".join(all_code_pieces)
        results['code'] = complete_solution
        
        return results
        
    except Exception as e:
        print(f"Error in solve_with_llm: {str(e)}")
        return None

def main():
    print("Starting carbon market solution process...")
    
    # Get and debug the solution
    final_solution = solve_with_llm()
    
    print("\nFinal Solution:")
    print(final_solution)
    
    # Save the solution to a file
    with open('carbon_market_solution.py', 'w') as f:
        f.write(final_solution)
    print("\nSolution saved to 'carbon_market_solution.py'")


def get_sr_prompt(top_responses):
    sr_prompt = """
        In this multi-period carbon market game:

    1. Market Structure:
       - 4 enterprises (i=1,2,3,4)
       - 4 trading periods (j=1,2,3,4)
       - Each enterprise has price elasticity (ai) and initial quota (Ai)

    2. Key Equations:
       - Price Formation: Pj = -sum_bkj[j] / (a1 + a2 + a3 + a4)
       - Trading Volume: Qij = ai * Pj + bij
       - Profit Functions for enterprise i:
         * Trading profits: sum(Qij * Pj) for j=1 to 4
         * Opportunity benefit: (1+r)P4 * (Ai - sum(Qij))

    Based on the following high-quality responses, you need to:

    1. First, provide the Subgame Perfect Nash Equilibrium expressions in this format:
    'Period 4 trading volumes': 'b14, b24, b34, b44'
    'Period 3 trading volumes': 'b13, b23, b33, b43'
    'Period 2 trading volumes': 'b12, b22, b32, b42'
    'Period 1 trading volumes': 'b11, b21, b31, b41'

    2. Then, you MUST provide a complete Python implementation that:
       a. Uses sympy for symbolic mathematics
       b. Implements backward induction across all periods
       c. Shows step-by-step derivation process
       d. Solves for all enterprises' equilibrium trading volumes
       e. Includes all necessary imports
       f. Is directly executable

    Your code MUST be provided in this exact format:
    'The complete code for solving the carbon market equilibrium':
    ```python
    import sympy as sp
    import numpy as np
    
    # Your implementation here
    # Must show complete backward induction steps
    # Must use symbolic mathematics
    ```

    Note: The code section is REQUIRED and must be complete and executable.
    """
    
    for idx, row in top_responses.iterrows():
        sr_prompt += f"\nResponse {idx + 1}: {row['quantity_prompt']}"
    
    sr_prompt += "\n\nRemember: You MUST provide both the equilibrium expressions AND the complete Python implementation code."
    
    return sr_prompt

def get_evaluation_prompt(response):
    """Generate evaluation prompt for carbon market game"""
    return f"""
Please evaluate the following solution for the multi-period carbon market game:

Response to evaluate:
{response}

Evaluation Criteria:
1. Mathematical Correctness:
   - Verify the backward induction process
   - Check price formation equations
   - Validate trading volume calculations
   - Confirm quota constraints

2. Economic Logic:
   - Strategic rationality across periods
   - Market clearing conditions
   - Profit maximization for each enterprise
   - Opportunity benefit consideration

3. Implementation Quality:
   - Completeness of symbolic mathematics
   - Correctness of Python code
   - Step-by-step solution clarity
   - Verification of results

Please provide a detailed evaluation addressing each criterion.
"""

def equilibrium_prompt(game_setting, sr_response):
    prompt_start = """Assume you are an economist needing to determine whether a solution is a Subgame Perfect Equilibrium in a sequential game. Follow these steps:
### Steps to Verify Subgame Perfect Equilibrium (SPE):

1. **Identify all subgames**:
   - Decompose the game from each decision node into all possible subgames.
   - Ensure that each subgame includes all future decisions and payoffs starting from the current decision node.
   - Subgames must be analyzed independently from the rest of the game, but must account for the strategies of subsequent players. A subgame is defined as a "complete game" on its own, meaning it includes all possible action sequences and payoffs.

2. **Apply backward induction**:
   - Start from the final stage of the game (where payoffs are realized) and work backwards through each decision node.
   - For each decision node, determine the best response for the player making the decision, assuming the strategies of other players at future decision nodes are already known.
   - **Avoid relying solely on checking whether the first derivative is zero**, which applies to static games but is insufficient for sequential games. The correct approach is to use backward induction to ensure that the strategy at every subgame is optimal for each player.

3. **Construct a complete strategy profile**:
   - Combine the optimal strategies for each player at every decision node to form a complete strategy profile.
   - The strategy profile must specify actions for each player at **every possible decision node**, including those that may not be reached on the equilibrium path (i.e., **off-the-equilibrium-path** strategies). These strategies are crucial for verifying the robustness of the equilibrium across all subgames.

4. **Check for consistency and optimality**:
   - Ensure that the strategy profile forms a **Nash equilibrium** in every subgame. This means that at each decision node, the player's strategy is their best response to the strategies of others.
   - Verify that even at off-the-equilibrium-path nodes, the strategy remains optimal for each player. This ensures the strategy profile is valid not only on the equilibrium path but also in every potential subgame, even if those subgames are not triggered in actual play.

5. **Ensure global optimality and account for constraints**:
   - The strategy profile must be **globally optimal** for each player, not just locally optimal. Consider any possible constraints, such as non-negativity, physical limits on resources, or other restrictions that may affect the solution.
   - If there are multiple equilibria, ensure that the solution satisfies all relevant constraints in every subgame and represents the globally best response.

6. **Validation and final check**:
   - To validate whether the given solution is a Subgame Perfect Equilibrium, verify that each player's strategy at every decision node meets the following criteria:
     - The strategy is the **best response** to the strategies of other players, taking into account all future decisions.
     - The strategy remains optimal at all subsequent decision nodes, including those off the equilibrium path.
   - If these conditions hold in every subgame, the solution constitutes a Subgame Perfect Equilibrium (SPE).

By following these steps, you will be able to rigorously determine if the given solution is a Subgame Perfect Equilibrium in the game.

Game Settings:
"""


    prompt_middle = """
Solution: """

    prompt_end = """
You need to note that the result of the solution may or may not be a Subgame Perfect Equilibrium. Please calculate strictly and judge based on the calculation results.

When writing code, pay attention to possible floating-point precision issues that may cause errors. Consider using types such as Decimal in Python or Fraction (like Rational). When comparing sizes, considering the accuracy of floating point numbers, you can set a small tolerance value epsilon = 10^-8.

Try not to use '==' to compare two expressions. When comparing two mathematical expressions for equality, use methods like simplify and equals, or employ expand, specific simplification functions, and numerical substitution to ensure accurate comparison.
When finding the partial derivative, remember to also substitute the equilibrium solutions of other players.

Write the code to verify whether the partial derivative is 0. Write the code to verify the strategies at every decision node. Use is_subgame_perfect_equilibrium to store the result. True means it is a Subgame Perfect Equilibrium, False means it is not. So the output result is either True or False. Only output the content that can be input into the py file. Do not output any other content."""

    return prompt_start + game_setting + prompt_middle + sr_response + prompt_end
