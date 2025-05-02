import re
import pandas as pd
from utils.model_caller import call_model, api_call_counts

def get_game_setting(strategy=None):
    game_setting = f""" 
 You are a mathematics and game theory expert. Please help me determine if there exists a Perfect Bayesian Equilibrium (PBE) in the following game, and if so, what the specific expressions are.

The game involves two players: a signal sender (denoted as S) and a signal receiver (denoted as R). The sender S has two possible types: high type (denoted as t1) and low type (denoted as t2), and this information is known only to S themselves. S can send two possible signals: signal m1 and signal m2. After receiving a signal, the receiver R can react in two possible ways: action a1 and action a2. The sequence and strategies of the game are as follows:

1. Nature acts first, determining S's type through a random process with a known probability distribution. Specifically, nature selects S's type as high type t1 with a 50% probability and as low type t2 with a 50% probability.

2. Knowing their type, S chooses to send signal m1 or m2. S's strategy can involve sending different signals based on their type, or choosing a signal to send regardless of their own type.

3. After observing S's signal, R chooses an action a1 or a2. R's strategy depends on their beliefs about S's type, which are updated based on S's signal.

4. Finally, the payoffs for S and R are determined by S's type, the signal sent by S, and R's action.

From this simple model of the signaling game, it is clear that the signal sender may have two types and can send two signals, hence they have 4 pure strategies:
* Strategy (m1, m1), where the sender sends signal m1 regardless of being high type or low type;
* Strategy (m2, m2), where the sender sends signal m2 regardless of being high type or low type;
* Strategy (m1, m2), where the sender sends signal m1 when they are high type t1 and signal m2 when they are low type t2;
* Strategy (m2, m1), where the sender sends signal m2 when they are high type t1 and signal m1 when they are low type t2.

By similar reasoning, the signal receiver has 4 pure strategies:
* Strategy (a1, a1), where the receiver takes action a1 regardless of whether the signal is m1 or m2;
* Strategy (a2, a2), where the receiver takes action a2 regardless of whether the signal is m1 or m2;
* Strategy (a1, a2), where the receiver takes action a1 when the signal is m1 and action a2 when the signal is m2;
* Strategy (a2, a1), where the receiver takes action a2 when the signal is m1 and action a1 when the signal is m2.

In this signaling game, the payoffs for different strategy combinations are as follows, where the first number in parentheses represents the payoff for the signal sender, and the second number represents the payoff for the signal receiver.
When S is t1, (t1, m1, a1) = (1, 3); (t1, m1, a2) = (4, 0); (t1, m2, a1) = (2, 1); (t1, m2, a2) = (0, 0);
When S is t2, (t2, m1, a1) = (2, 4); (t2, m1, a2) = (0, 1); (t2, m2, a1) = (1, 0); (t2, m2, a2) = (1, 2);
    """
    return game_setting

def signal_prompt_gen(strategy):
  signal_prompt = f'''You are a mathematics and game theory expert. Please help me determine if there exists a Perfect Bayesian Equilibrium (PBE) in the following game, and if so, what the specific expressions are.

The game involves two players: a signal sender (denoted as S) and a signal receiver (denoted as R). The sender S has two possible types: high type (denoted as t1) and low type (denoted as t2), and this information is known only to S themselves. S can send two possible signals: signal m1 and signal m2. After receiving a signal, the receiver R can react in two possible ways: action a1 and action a2. The sequence and strategies of the game are as follows:

1. Nature acts first, determining S's type through a random process with a known probability distribution. Specifically, nature selects S's type as high type t1 with a 50% probability and as low type t2 with a 50% probability.

2. Knowing their type, S chooses to send signal m1 or m2. S's strategy can involve sending different signals based on their type, or choosing a signal to send regardless of their own type.

3. After observing S's signal, R chooses an action a1 or a2. R's strategy depends on their beliefs about S's type, which are updated based on S's signal.

4. Finally, the payoffs for S and R are determined by S's type, the signal sent by S, and R's action.

From this simple model of the signaling game, it is clear that the signal sender may have two types and can send two signals, hence they have 4 pure strategies:
* Strategy (m1, m1), where the sender sends signal m1 regardless of being high type or low type;
* Strategy (m2, m2), where the sender sends signal m2 regardless of being high type or low type;
* Strategy (m1, m2), where the sender sends signal m1 when they are high type t1 and signal m2 when they are low type t2;
* Strategy (m2, m1), where the sender sends signal m2 when they are high type t1 and signal m1 when they are low type t2.

By similar reasoning, the signal receiver has 4 pure strategies:
* Strategy (a1, a1), where the receiver takes action a1 regardless of whether the signal is m1 or m2;
* Strategy (a2, a2), where the receiver takes action a2 regardless of whether the signal is m1 or m2;
* Strategy (a1, a2), where the receiver takes action a1 when the signal is m1 and action a2 when the signal is m2;
* Strategy (a2, a1), where the receiver takes action a2 when the signal is m1 and action a1 when the signal is m2.

In this signaling game, the payoffs for different strategy combinations are as follows, where the first number in parentheses represents the payoff for the signal sender, and the second number represents the payoff for the signal receiver.
When S is t1, (t1, m1, a1) = (1, 3); (t1, m1, a2) = (4, 0); (t1, m2, a1) = (2, 1); (t1, m2, a2) = (0, 0);
When S is t2, (t2, m1, a1) = (2, 4); (t2, m1, a2) = (0, 1); (t2, m2, a1) = (1, 0); (t2, m2, a2) = (1, 2);

For convenience, let us define the probability that the receiver believes the sender is of type t1 upon receiving signal m1 as p, and the probability of being type t2 as 1-p. Upon receiving signal m2, the probability that the receiver believes the sender is of type t1 is q, and the probability of being type t2 is 1-q.

Now, we come to judge whether the signal sender strategy {strategy} can find a Perfect Bayesian Equilibrium PBE, in other words, whether ({strategy}, (receiver strategy), p = probability, q = probability) exists as a PBE. The strategy is as follows:
1. For a given signal sender strategy, we first determine if values of p and q can be determined. For example, if the sender's strategy is (m1, m1), then the signal does not convey additional information to the receiver, hence their belief p = 1 - p = 0.5, but the value of q is still uncertain; if the sender's strategy is (m1, m2), then receiving signal m1 indicates S's type is t1, and signal m2 indicates S's type is t2.
2. Then, find the optimal response for the receiver under the current scenario. That is, based on the determined values of p and q, choose the receiver's optimal response. For example, when p = 0.5, meaning there's a 50% chance S is t1 and a 50% chance S is t2 upon receiving signal m1, the expected payoff for action a1 is 3p + 4(1-p), and for action a2, the expected payoff is 0*p + 1 * (1-p). When p = 0.5, we find a1 is better than a2, suggesting that when the sender's strategy is against m1, the best strategy for the receiver, if a PBE exists, must be (a1, x).
3. Next, for the receiver strategy identified, we need to verify if our initially given sender strategy remains optimal, in other words, the sender has no incentive to deviate. For example, if the original sender strategy is (m1, m2) and the receiver strategy is (a1, a2), we need to determine among the four strategies (m1, m1), (m1, m2), (m2, m1), and (m2, m2), whether (m1, m2) yields the highest payoff for the sender. If not, then it's not a PBE. If so, then this pair of sender and receiver strategies could be a PBE. At this point, if S is t1, the payoff for (m1, m2) for the sender is 1; if S is t2, the payoff is 1, with an overall expected payoff of 1; while calculating the expected payoff for (m1, m1), we need to update the belief to p = 1 - p = 0.5 given the receiver strategy of (a1, a2), leading to a final action of a1 and an overall expected payoff of 0.5*1+0.5*2, thus not a PBE. If the receiver's strategy is (a1, x), then we need to separately determine (a1, a1) and (a1, a2). Moreover, it's important to note that for each information set in the dynamic game, players update their beliefs based on the observed action history.
4. If at this point a possible PBE receiver strategy has been uniquely identified (see step 2), then we can directly output the ((sender strategy), (receiver strategy), p = probability, q = probability); if not yet determined, further determination is needed. For example, if step two identifies the receiver strategy as (a1, x) and step three finds only (a1, a2) as the optimal strategy, we still need to supplement the range of q values for which the receiver definitely takes action a2 upon receiving signal m2, then output ((sender strategy), (receiver strategy), p = probability, q = probability); if (a1,x) can all be optimal strategies, then output multiple PBEs, (sender strategy), (receiver strategy), p = probability, q = probability).

It should be noted that:
1. Your analysis must be completely rational, based on current data calculations, and can use expected values to calculate payoffs.
2. For each information set in the dynamic game, players update their beliefs based on the observed action history. For example, when determining whether (m1, m2) yields the highest payoff for the sender among the strategies (m1, m2), (m1, m2), (m2, m1), and (m2, m2), the expected payoff for (m1, m1) needs to be recalculated with updated beliefs of p = 1 - p = 0.5 and the receiver strategy of (a1, a2), leading to a final action of a1 and an overall expected payoff of 0.5*1+0.5*2, thus not a PBE.
3. Carefully consider and analyze the problem, conducting thorough categorization and discussion. The information available is sufficient to determine the outcome, either proving the existence of a PBE, with expressions provided if so, or proving its nonexistence.

Now, begin calculating whether the signal sender strategy {strategy} can find a Perfect Bayesian Equilibrium PBE. If so, output "The PBE strategy is: ((sender strategy), (receiver strategy), p = probability, q = probability)"; if not, output "No PBE exists".
'''
  return signal_prompt

def pbe_re(signal_output):
  pbe_regex = r"The PBE strategy is: \(\((m[12]), (m[12])\), \((x|a[12]), (x|a[12])\), p = ([0-9.]+), q = (irrelevant|[0-9.]+)\)"
  sender_strategy_list = []
  receiver_strategy_list = []
  p_list = []
  q_list = []
  index = -1
  match = re.search(pbe_regex, signal_output)
  if match:

      sender_strategy_m1 = f"'{match.group(1)}'"
      sender_strategy_m2 = f"'{match.group(2)}'"
      receiver_strategy_a1 = str(match.group(3))
      receiver_strategy_a2_or_x = str(match.group(4))
      p_value = match.group(5)
      q_value = match.group(6)
      # Check if 'x' needs to be replaced
      if receiver_strategy_a1 == 'a1' and receiver_strategy_a2_or_x == 'x':
          # Generate versions replacing 'x' with 'a1' and 'a2'
          replacements = [('a1', 'a1'), ('a1', 'a2')]
          index = 1
      elif receiver_strategy_a1 == 'a2' and receiver_strategy_a2_or_x == 'x':
          # Generate versions replacing 'x' with 'a1' and 'a2'
          replacements = [('a2', 'a1'), ('a2', 'a2')]
          index = 1
      elif receiver_strategy_a1 == 'x'  and receiver_strategy_a2_or_x == 'a1':
          # Generate versions replacing 'x' with 'a1' and 'a2'
          replacements = [('a1', 'a1'), ('a2', 'a1')]
          index = 0
      elif receiver_strategy_a1 == 'x'  and receiver_strategy_a2_or_x == 'a2':
          # Generate versions replacing 'x' with 'a1' and 'a2'
          replacements = [('a1', 'a2'), ('a2', 'a2')]
          index = 0
      elif receiver_strategy_a1 == 'x'  and receiver_strategy_a2_or_x == 'x':
          # Generate versions replacing 'x' with 'a1' and 'a2'
          replacements = [('a1', 'a1'), ('a1', 'a2'), ('a2', 'a1'), ('a2', 'a2')]
          index = 2
      else:
          # If there's no 'x', use the original values
          replacements = [(receiver_strategy_a1, receiver_strategy_a2_or_x)]

      # Print all replaced versions
      for receiver_strategy_a2 in replacements:
          sender_strategy = f"({sender_strategy_m1}, {sender_strategy_m2})"
          # receiver_strategy = f"({receiver_strategy_a1[1]}, {receiver_strategy_a2[1]})"
          receiver_strategy = f"{receiver_strategy_a2}"
          sender_strategy_list.append(sender_strategy)
          receiver_strategy_list.append(receiver_strategy)
          p_list.append(p_value)
          q_list.append(q_value)
          print(f"({sender_strategy}, {receiver_strategy}, p = {p_value}, q = {q_value})")
  else:
    sender_strategy, receiver_strategy, p, q, index = None, None, None, None, None
    print("No PBE found in this expression.")
  return sender_strategy_list, receiver_strategy_list, p_list, q_list, index

def get_evaluation_prompt(strategy, response):
    """Get evaluation prompt"""
    return f"""
You are a mathematics and game theory expert. Please help me determine if there exists a Perfect Bayesian Equilibrium (PBE) in the following game, and if so, what the specific expressions are.

The game involves two players: a signal sender (denoted as S) and a signal receiver (denoted as R). The sender S has two possible types: high type (denoted as t1) and low type (denoted as t2), and this information is known only to S themselves. S can send two possible signals: signal m1 and signal m2. After receiving a signal, the receiver R can react in two possible ways: action a1 and action a2. The sequence and strategies of the game are as follows:

1. Nature acts first, determining S's type through a random process with a known probability distribution. Specifically, nature selects S's type as high type t1 with a 50% probability and as low type t2 with a 50% probability.

2. Knowing their type, S chooses to send signal m1 or m2. S's strategy can involve sending different signals based on their type, or choosing a signal to send regardless of their own type.

3. After observing S's signal, R chooses an action a1 or a2. R's strategy depends on their beliefs about S's type, which are updated based on S's signal.

4. Finally, the payoffs for S and R are determined by S's type, the signal sent by S, and R's action.

From this simple model of the signaling game, it is clear that the signal sender may have two types and can send two signals, hence they have 4 pure strategies:
* Strategy (m1, m1), where the sender sends signal m1 regardless of being high type or low type;
* Strategy (m2, m2), where the sender sends signal m2 regardless of being high type or low type;
* Strategy (m1, m2), where the sender sends signal m1 when they are high type t1 and signal m2 when they are low type t2;
* Strategy (m2, m1), where the sender sends signal m2 when they are high type t1 and signal m1 when they are low type t2.

By similar reasoning, the signal receiver has 4 pure strategies:
* Strategy (a1, a1), where the receiver takes action a1 regardless of whether the signal is m1 or m2;
* Strategy (a2, a2), where the receiver takes action a2 regardless of whether the signal is m1 or m2;
* Strategy (a1, a2), where the receiver takes action a1 when the signal is m1 and action a2 when the signal is m2;
* Strategy (a2, a1), where the receiver takes action a2 when the signal is m1 and action a1 when the signal is m2.

In this signaling game, the payoffs for different strategy combinations are as follows, where the first number in parentheses represents the payoff for the signal sender, and the second number represents the payoff for the signal receiver.
When S is t1, (t1, m1, a1) = (1, 3); (t1, m1, a2) = (4, 0); (t1, m2, a1) = (2, 1); (t1, m2, a2) = (0, 0);
When S is t2, (t2, m1, a1) = (2, 4); (t2, m1, a2) = (0, 1); (t2, m2, a1) = (1, 0); (t2, m2, a2) = (1, 2);

For convenience, let us define the probability that the receiver believes the sender is of type t1 upon receiving signal m1 as p, and the probability of being type t2 as 1-p. Upon receiving signal m2, the probability that the receiver believes the sender is of type t1 is q, and the probability of being type t2 is 1-q.

Now, we come to judge whether the signal sender strategy {strategy} can find a Perfect Bayesian Equilibrium PBE, in other words, whether ({strategy}, (receiver strategy), p = probability, q = probability) exists as a PBE. The strategy is as follows:
1. For a given signal sender strategy, we first determine if values of p and q can be determined. For example, if the sender's strategy is (m1, m1), then the signal does not convey additional information to the receiver, hence their belief p = 1 - p = 0.5; if the sender's strategy is (m1, m2), then receiving signal m1 indicates S's type is t1, and signal m2 indicates S's type is t2.
2. Then, find the optimal response for the receiver under the current scenario. That is, based on the determined values of p and q, choose the receiver's optimal response. For example, when p = 0.5, meaning there's a 50% chance S is t1 and a 50% chance S is t2 upon receiving signal m1, the expected payoff for action a1 is 3p + 4(1-p), and for action a2, the expected payoff is 0*p + 1 * (1-p). When p = 0.5, we find a1 is better than a2, suggesting that when the sender's strategy is against m1, the best strategy for the receiver, if a PBE exists, must be (a1, x).
3. Next, for the receiver strategy identified, we need to verify if our initially given sender strategy remains optimal, in other words, the sender has no incentive to deviate. For example, if the original sender strategy is (m1, m2) and the receiver strategy is (a1, a2), we need to determine among the four strategies (m1, m1), (m1, m2), (m2, m1), and (m2, m2), whether (m1, m2) yields the highest payoff for the sender. If not, then it's not a PBE. If so, then this pair of sender and receiver strategies could be a PBE. At this point, if S is t1, the payoff for (m1, m2) for the sender is 1; if S is t2, the payoff is 1, with an overall expected payoff of 1; while calculating the expected payoff for (m1, m1), we need to update the belief to p = 1 - p = 0.5 given the receiver strategy of (a1, a2), leading to a final action of a1 and an overall expected payoff of 0.5*1+0.5*2, thus not a PBE. If the receiver's strategy is (a1, x), then we need to separately determine (a1, a1) and (a1, a2). Moreover, it's important to note that for each information set in the dynamic game, players update their beliefs based on the observed action history.
4. If at this point a possible PBE receiver strategy has been uniquely identified (see step 2), then we can directly output the ((sender strategy), (receiver strategy), p = probability, q = probability); if not yet determined, further determination is needed. For example, if step two identifies the receiver strategy as (a1, x) and step three finds only (a1, a2) as the optimal strategy, we still need to supplement the range of q values for which the receiver definitely takes action a2 upon receiving signal m2, then output ((sender strategy), (receiver strategy), p = probability, q = probability); if (a1,x) can all be optimal strategies, then output multiple PBEs, (sender strategy), (receiver strategy), p = probability, q = probability).

  Please evaluate the following response based on its alignment with strategic principles and mathematical accuracy. Score range: 0-10 (where 10 indicates complete alignment with strategic principles and high mathematical accuracy, and 0 indicates complete misalignment with strategic principles and mathematical accuracy). If the quantity in numerical form is not obtained, points will be deducted.

  Response: {response}

  Please provide your final score in the following format: 'score': 'xx'.
    """

def calculate_expected_benefit(sender_strategy, receiver_strategy, p, q, payoffs):
    """
    Calculate the expected benefit of a given strategy combination.

    Args:
    - sender_strategy: Sender's strategy, e.g., (m1, m2)
    - receiver_strategy: Receiver's strategy, e.g., (a1, a2)
    - p: Probability that the receiver believes the sender is type t1 upon receiving signal m1
    - q: Probability that the receiver believes the sender is type t1 upon receiving signal m2
    - payoffs: Payoff matrix, a dictionary with keys as strategy combinations and values as payoff pairs (Sender payoff, Receiver payoff)

    Returns:
    - Expected benefit for the sender and the receiver
    """
    # Initialize expected benefits
    expected_benefit_sender = 0
    expected_benefit_receiver = 0

    # Loop through each type of sender
    for type_s in ['t1', 't2']:
        # Determine the sender's signal
        signal = sender_strategy[0] if type_s == 't1' else sender_strategy[1]

        # Determine the receiver's action
        action = receiver_strategy[0] if signal == 'm1' else receiver_strategy[1]

        # Calculate probability
        prob = p if signal == 'm1' else q if type_s == 't1' else 1-p if signal == 'm1' else 1-q

        # Calculate expected benefits
        key = (type_s, signal, action)
        if key in payoffs:
            expected_benefit_sender += prob * payoffs[key][0]
            expected_benefit_receiver += prob * payoffs[key][1]

    return expected_benefit_sender, expected_benefit_receiver

def benefit(sender_strategy, receiver_strategy, p, q):
  payoffs = {
      ('t1', 'm1', 'a1'): (1, 3),
      ('t1', 'm1', 'a2'): (4, 0),
      ('t1', 'm2', 'a1'): (2, 1),
      ('t1', 'm2', 'a2'): (0, 0),
      ('t2', 'm1', 'a1'): (2, 4),
      ('t2', 'm1', 'a2'): (0, 1),
      ('t2', 'm2', 'a1'): (1, 0),
      ('t2', 'm2', 'a2'): (1, 2),
  }

  # Example strategy combination
  sender_strategy = (sender_strategy[1], sender_strategy[-2])  # Assume sender's strategy is t1 sends m1, t2 sends m2
  receiver_strategy = (receiver_strategy[1], receiver_strategy[-2])  # Assume receiver's strategy is to take a1 when seeing m1, and a2 when seeing m2
  try:
    p = float(p)  # Value of p
  except:
    p = 0.8
  try:
    q = float(q)  # Value of q
  except:
    q = 0.8
  if sender_strategy[1] == 'm1' and sender_strategy[-2] == 'm1':
    p = 0.5
  if sender_strategy[1] == 'm2' and sender_strategy[-2] == 'm2':
    q = 0.5
  if sender_strategy[1] == 'm1' and sender_strategy[-2] == 'm2':
    p = 1
    q = 0
  if sender_strategy[1] == 'm2' and sender_strategy[-2] == 'm1':
    p = 0
    q = 1

  # Calculate expected benefit
  # expected_benefit_sender, expected_benefit_receiver = calculate_expected_benefit(sender_strategy_list[0], receiver_strategy_list[0], p_list[0], q_list[0], payoffs)
  expected_benefit_sender, expected_benefit_receiver = calculate_expected_benefit(sender_strategy, receiver_strategy, p, q, payoffs)
  return expected_benefit_sender, expected_benefit_receiver, p, q


def extract_nash_equilibrium(response, pattern_1, pattern_2=None):
    """Extract Nash equilibrium results"""
    try:
        # Try the first pattern first
        match = re.search(pattern_1, response)
        if match:
            return match.group(1)
        # If the second pattern is provided, try the second one
        if pattern_2:
            match = re.search(pattern_2, response)
            if match:
                return match.group(1)
    except:
        return None
    return None


def generate_sr_prompt(strategy, top_responses):
    sr_prompt = f"""
You are a mathematics and game theory expert. Please help me determine if there exists a Perfect Bayesian Equilibrium (PBE) in the following game, and if so, what the specific expressions are.

The game involves two players: a signal sender (denoted as S) and a signal receiver (denoted as R). The sender S has two possible types: high type (denoted as t1) and low type (denoted as t2), and this information is known only to S themselves. S can send two possible signals: signal m1 and signal m2. After receiving a signal, the receiver R can react in two possible ways: action a1 and action a2. The sequence and strategies of the game are as follows:

1. Nature acts first, determining S's type through a random process with a known probability distribution. Specifically, nature selects S's type as high type t1 with a 50% probability and as low type t2 with a 50% probability.

2. Knowing their type, S chooses to send signal m1 or m2. S's strategy can involve sending different signals based on their type, or choosing a signal to send regardless of their own type.

3. After observing S's signal, R chooses an action a1 or a2. R's strategy depends on their beliefs about S's type, which are updated based on S's signal.

4. Finally, the payoffs for S and R are determined by S's type, the signal sent by S, and R's action.

From this simple model of the signaling game, it is clear that the signal sender may have two types and can send two signals, hence they have 4 pure strategies:
* Strategy (m1, m1), where the sender sends signal m1 regardless of being high type or low type;
* Strategy (m2, m2), where the sender sends signal m2 regardless of being high type or low type;
* Strategy (m1, m2), where the sender sends signal m1 when they are high type t1 and signal m2 when they are low type t2;
* Strategy (m2, m1), where the sender sends signal m2 when they are high type t1 and signal m1 when they are low type t2.

By similar reasoning, the signal receiver has 4 pure strategies:
* Strategy (a1, a1), where the receiver takes action a1 regardless of whether the signal is m1 or m2;
* Strategy (a2, a2), where the receiver takes action a2 regardless of whether the signal is m1 or m2;
* Strategy (a1, a2), where the receiver takes action a1 when the signal is m1 and action a2 when the signal is m2;
* Strategy (a2, a1), where the receiver takes action a2 when the signal is m1 and action a1 when the signal is m2.

In this signaling game, the payoffs for different strategy combinations are as follows, where the first number in parentheses represents the payoff for the signal sender, and the second number represents the payoff for the signal receiver.
When S is t1, (t1, m1, a1) = (1, 3); (t1, m1, a2) = (4, 0); (t1, m2, a1) = (2, 1); (t1, m2, a2) = (0, 0);
When S is t2, (t2, m1, a1) = (2, 4); (t2, m1, a2) = (0, 1); (t2, m2, a1) = (1, 0); (t2, m2, a2) = (1, 2);

For convenience, let us define the probability that the receiver believes the sender is of type t1 upon receiving signal m1 as p, and the probability of being type t2 as 1-p. Upon receiving signal m2, the probability that the receiver believes the sender is of type t1 is q, and the probability of being type t2 is 1-q.

Now, we come to judge whether the signal sender strategy {strategy} can find a Perfect Bayesian Equilibrium PBE, in other words, whether ({strategy}, (receiver strategy), p = probability, q = probability) exists as a PBE. The strategy is as follows:
1. For a given signal sender strategy, we first determine if values of p and q can be determined. For example, if the sender's strategy is (m1, m1), then the signal does not convey additional information to the receiver, hence their belief p = 1 - p = 0.5; if the sender's strategy is (m1, m2), then receiving signal m1 indicates S's type is t1, and signal m2 indicates S's type is t2.
2. Then, find the optimal response for the receiver under the current scenario. That is, based on the determined values of p and q, choose the receiver's optimal response. For example, when p = 0.5, meaning there's a 50% chance S is t1 and a 50% chance S is t2 upon receiving signal m1, the expected payoff for action a1 is 3p + 4(1-p), and for action a2, the expected payoff is 0*p + 1 * (1-p). When p = 0.5, we find a1 is better than a2, suggesting that when the sender's strategy is against m1, the best strategy for the receiver, if a PBE exists, must be (a1, x).
3. Next, for the receiver strategy identified, we need to verify if our initially given sender strategy remains optimal, in other words, the sender has no incentive to deviate. For example, if the original sender strategy is (m1, m2) and the receiver strategy is (a1, a2), we need to determine among the four strategies (m1, m1), (m1, m2), (m2, m1), and (m2, m2), whether (m1, m2) yields the highest payoff for the sender. If not, then it's not a PBE. If so, then this pair of sender and receiver strategies could be a PBE. At this point, if S is t1, the payoff for (m1, m2) for the sender is 1; if S is t2, the payoff is 1, with an overall expected payoff of 1; while calculating the expected payoff for (m1, m1), we need to update the belief to p = 1 - p = 0.5 given the receiver strategy of (a1, a2), leading to a final action of a1 and an overall expected payoff of 0.5*1+0.5*2, thus not a PBE. If the receiver's strategy is (a1, x), then we need to separately determine (a1, a1) and (a1, a2). Moreover, it's important to note that for each information set in the dynamic game, players update their beliefs based on the observed action history.
4. If at this point a possible PBE receiver strategy has been uniquely identified (see step 2), then we can directly output the ((sender strategy), (receiver strategy), p = probability, q = probability); if not yet determined, further determination is needed. For example, if step two identifies the receiver strategy as (a1, x) and step three finds only (a1, a2) as the optimal strategy, we still need to supplement the range of q values for which the receiver definitely takes action a2 upon receiving signal m2, then output ((sender strategy), (receiver strategy), p = probability, q = probability); if (a1,x) can all be optimal strategies, then output multiple PBEs, (sender strategy), (receiver strategy), p = probability, q = probability).

Based on the following high-quality responses, summarize the general expression of Perfect Bayesian Equilibrium under the current game settings.
            """

    for idx, row in top_responses.iterrows():
        sr_prompt += f"\n\nResponse {idx + 1}: {row['quantity_prompt']}"

    sr_prompt += f"""
  Based on the above higher-quality response, which is the solution idea.

  Please calculate whether the signal sender strategy {strategy} can find a Perfect Bayesian Equilibrium PBE. If so, output "The PBE strategy is: ((sender strategy), (receiver strategy), p = probability, q = probability)"; if not, output "No PBE exists".
  """ 
    return sr_prompt


def get_verification_prompt(game_setting, sr_response):
    prompt_start = """Assume you are an economist needing to determine whether a solution is a Perfect Bayesian Equilibrium in a dynamic game with incomplete information. Follow these steps:
1. Given that the decisions of other players remain unchanged, determine whether each player's decision is optimal. If it is not optimal, then it must not be PBE.

2. Identify all information sets: Break down the game into all possible information sets for each player.

3. Determine the players' beliefs: For each information set, specify the beliefs players have about the game's history.

4. Construct strategy profiles: Determine the best strategy for each player at every information set, given their beliefs.

5. Check for consistency: Ensure that the strategy profiles and beliefs are consistent throughout the game. The strategies should be optimal given the beliefs, and the beliefs should be updated correctly using Bayes' rule wherever applicable.


By following these steps, you will be able to rigorously determine if the given solution is a Perfect Bayesian Equilibrium in the game.

Game Settings:
"""

    prompt_middle = """
Solution: """

    prompt_end = """
You need to note that the result of the solution may or may not be a Perfect Bayesian Equilibrium. Please calculate strictly and judge based on the calculation results.

When writing code, pay attention to possible floating-point precision issues that may cause errors. Consider using types such as Decimal in Python or Fraction (like Rational). When using ('==', '<=', '>='), considering the accuracy of floating point numbers, you can set a small tolerance value epsilon = 10^-8. e.g. if you want to write, 'number1 <= number2', then use 'number1 - number2 <= epsilon' instead.
Try not to use '==' to compare two expressions. When comparing two mathematical expressions for equality, use methods like simplify and equals, or employ expand, specific simplification functions, and numerical substitution to ensure accurate comparison.
When finding the partial derivative, remember to also substitute the equilibrium solutions of other players and their beliefs.
If you want to write, 'x1 <= x2', then use 'x1 - x2 <= epsilon' instead. If you want to write, 'x1 >= x2', then use 'x2 - x1 <= epsilon' instead. If you want to write, 'x1 == x2', then use 'abs(x1 - x2) <= epsilon' instead. x1, x2 can be a numeric value or an expression.


Write the code to verify the strategies and beliefs at every information set. Use is_perfect_bayesian_equilibrium to store the result. True means it is a Perfect Bayesian Equilibrium, False means it is not. Only output the content that can be input into the py file. Do not output any other content at all."""

    prompt = prompt_start + game_setting + prompt_middle + sr_response + prompt_end

    return prompt


def get_fallback_prompt(game_setting, sr_response, isNE, top_response, idx):
    return f"""You are a mathematics and game theory expert. Please help me determine if there exists a Perfect Bayesian Equilibrium (PBE) in the following game, and if so, what the specific expressions are.

The game involves two players: a signal sender (denoted as S) and a signal receiver (denoted as R). The sender S has two possible types: high type (denoted as t1) and low type (denoted as t2), and this information is known only to S themselves. S can send two possible signals: signal m1 and signal m2. After receiving a signal, the receiver R can react in two possible ways: action a1 and action a2. The sequence and strategies of the game are as follows:

1. Nature acts first, determining S's type through a random process with a known probability distribution. Specifically, nature selects S's type as high type t1 with a 50% probability and as low type t2 with a 50% probability.

2. Knowing their type, S chooses to send signal m1 or m2. S's strategy can involve sending different signals based on their type, or choosing a signal to send regardless of their own type.

3. After observing S's signal, R chooses an action a1 or a2. R's strategy depends on their beliefs about S's type, which are updated based on S's signal.

4. Finally, the payoffs for S and R are determined by S's type, the signal sent by S, and R's action.

From this simple model of the signaling game, it is clear that the signal sender may have two types and can send two signals, hence they have 4 pure strategies:
* Strategy (m1, m1), where the sender sends signal m1 regardless of being high type or low type;
* Strategy (m2, m2), where the sender sends signal m2 regardless of being high type or low type;
* Strategy (m1, m2), where the sender sends signal m1 when they are high type t1 and signal m2 when they are low type t2;
* Strategy (m2, m1), where the sender sends signal m2 when they are high type t1 and signal m1 when they are low type t2.

By similar reasoning, the signal receiver has 4 pure strategies:
* Strategy (a1, a1), where the receiver takes action a1 regardless of whether the signal is m1 or m2;
* Strategy (a2, a2), where the receiver takes action a2 regardless of whether the signal is m1 or m2;
* Strategy (a1, a2), where the receiver takes action a1 when the signal is m1 and action a2 when the signal is m2;
* Strategy (a2, a1), where the receiver takes action a2 when the signal is m1 and action a1 when the signal is m2.

In this signaling game, the payoffs for different strategy combinations are as follows, where the first number in parentheses represents the payoff for the signal sender, and the second number represents the payoff for the signal receiver.
When S is t1, (t1, m1, a1) = (1, 3); (t1, m1, a2) = (4, 0); (t1, m2, a1) = (2, 1); (t1, m2, a2) = (0, 0);
When S is t2, (t2, m1, a1) = (2, 4); (t2, m1, a2) = (0, 1); (t2, m2, a1) = (1, 0); (t2, m2, a2) = (1, 2);

### Equilibrium Candidate:
{sr_response}

### Reason it is not an Equilibrium:
{isNE}  

### High-quality Response {idx+1}:
{top_response.iloc[idx]['quantity_prompt']}

# ### Task:
# - Analyze the **High-quality Response {idx+1}** based on the Equilibrium Candidate and the reason why it is not an equilibrium.
# - Explain **why the current response is not a valid equilibrium**.
# - Provide **specific suggestions** on how to modify the **High-quality Response** so that it can become a valid Bayesian Nash Equilibrium.

### Output Format:
- [Analysis]: Explanation of why the current response is not a valid equilibrium.
- [Suggestions]: Specific steps to adjust the response to become a valid equilibrium.
"""

def get_quantity_prompt(game_setting, top_response, fallback_response, idx, strategy):
    return f"""Assuming you are a player, please find the conditions for a Bayesian Nash Equilibrium. Regardless of cooperation, both parties seek Nash equilibrium.

You are a mathematics and game theory expert. Please help me determine if there exists a Perfect Bayesian Equilibrium (PBE) in the following game, and if so, what the specific expressions are.

The game involves two players: a signal sender (denoted as S) and a signal receiver (denoted as R). The sender S has two possible types: high type (denoted as t1) and low type (denoted as t2), and this information is known only to S themselves. S can send two possible signals: signal m1 and signal m2. After receiving a signal, the receiver R can react in two possible ways: action a1 and action a2. The sequence and strategies of the game are as follows:

1. Nature acts first, determining S's type through a random process with a known probability distribution. Specifically, nature selects S's type as high type t1 with a 50% probability and as low type t2 with a 50% probability.

2. Knowing their type, S chooses to send signal m1 or m2. S's strategy can involve sending different signals based on their type, or choosing a signal to send regardless of their own type.

3. After observing S's signal, R chooses an action a1 or a2. R's strategy depends on their beliefs about S's type, which are updated based on S's signal.

4. Finally, the payoffs for S and R are determined by S's type, the signal sent by S, and R's action.

From this simple model of the signaling game, it is clear that the signal sender may have two types and can send two signals, hence they have 4 pure strategies:
* Strategy (m1, m1), where the sender sends signal m1 regardless of being high type or low type;
* Strategy (m2, m2), where the sender sends signal m2 regardless of being high type or low type;
* Strategy (m1, m2), where the sender sends signal m1 when they are high type t1 and signal m2 when they are low type t2;
* Strategy (m2, m1), where the sender sends signal m2 when they are high type t1 and signal m1 when they are low type t2.

By similar reasoning, the signal receiver has 4 pure strategies:
* Strategy (a1, a1), where the receiver takes action a1 regardless of whether the signal is m1 or m2;
* Strategy (a2, a2), where the receiver takes action a2 regardless of whether the signal is m1 or m2;
* Strategy (a1, a2), where the receiver takes action a1 when the signal is m1 and action a2 when the signal is m2;
* Strategy (a2, a1), where the receiver takes action a2 when the signal is m1 and action a1 when the signal is m2.

In this signaling game, the payoffs for different strategy combinations are as follows, where the first number in parentheses represents the payoff for the signal sender, and the second number represents the payoff for the signal receiver.
When S is t1, (t1, m1, a1) = (1, 3); (t1, m1, a2) = (4, 0); (t1, m2, a1) = (2, 1); (t1, m2, a2) = (0, 0);
When S is t2, (t2, m1, a1) = (2, 4); (t2, m1, a2) = (0, 1); (t2, m2, a1) = (1, 0); (t2, m2, a2) = (1, 2);

### Previous High-Quality Response:
Your previous response was: {top_response.iloc[idx]['quantity_prompt']}

### Feedback on the Previous Response:
The feedback is: {fallback_response}

Based on the above higher quality response, which is the solution idea.


Now, begin calculating whether the signal sender strategy {strategy} can find a Perfect Bayesian Equilibrium PBE. If so, output "The PBE strategy is: ((sender strategy), (receiver strategy), p = probability, q = probability)"; if not, output "No PBE exists".
         """


def signal(model_name, strategy_list):
    """Analyze the Perfect Bayesian Equilibrium of a signaling game"""
    results = []
    for strategy in strategy_list:
        # Generate the signaling game prompt
        signal_prompt = signal_prompt_gen(strategy)
        signal_output = call_model(model_name, signal_prompt)
        
        if not signal_output:
            print(f"Failed to get response for strategy {strategy}")
            continue
            
        final_response = signal_output
        
        # Extract strategy combinations
        sender_strategy_list, receiver_strategy_list, p_list, q_list, index = pbe_re(signal_output)
        sender_strategy_list_NE = sender_strategy_list.copy() if sender_strategy_list else []
        receiver_strategy_list_NE = receiver_strategy_list.copy() if receiver_strategy_list else []
        p_list_NE = p_list.copy() if p_list else []
        q_list_NE = q_list.copy() if q_list else []
        
        test_prompt = ''
        score = 0
        
        # Verify each strategy combination
        if sender_strategy_list:
            for i in range(len(sender_strategy_list)):
                sender_strategy = sender_strategy_list[i]
                receiver_strategy = receiver_strategy_list[i]
                p = p_list[i]
                q = q_list[i]
                
                # Convert strategy format
                if isinstance(sender_strategy, str):
                    sender_strategy = (sender_strategy[2:4], sender_strategy[-4:-2])
                if isinstance(receiver_strategy, str):
                    receiver_strategy = (receiver_strategy[2:4], receiver_strategy[-4:-2])
                
                # Calculate benefits
                expected_benefit_sender, expected_benefit_receiver, p, q = benefit(
                    sender_strategy, receiver_strategy, p, q)
                print(f"PBE: {expected_benefit_sender}, {expected_benefit_receiver}")
                
                # Test other possible sender strategies
                sender_strategy_set = [('m1', 'm1'), ('m1', 'm2'), ('m2', 'm1'), ('m2', 'm2')]
                for sender_strategy_test in sender_strategy_set:
                    if sender_strategy_test != sender_strategy:
                        expected_benefit_sender_test, expected_benefit_receiver_test, p_test, q_test = benefit(
                            sender_strategy_test, receiver_strategy, p, q)
                        print(f"Test: {expected_benefit_sender_test}, {expected_benefit_receiver_test}")
                        
                        # If there is a better strategy, the current strategy is not a PBE
                        if expected_benefit_sender_test > expected_benefit_sender:
                            try:
                                del sender_strategy_list_NE[i]
                                del receiver_strategy_list_NE[i]
                                del p_list_NE[i]
                                del q_list_NE[i]
                                score = score - 1
                            except:
                                pass
                            test_prompt += f"When the sender's strategy is ({sender_strategy_test}, {receiver_strategy}), the updated values of p and q are p = {p_test}, q = {q_test}. The receiver's income is {expected_benefit_sender_test}, so the receiver has motivation to change from {sender_strategy}, thus not a PBE."
        
        # If a problem is found, provide feedback
        feedback_response = ''
        if test_prompt:
            feedback_prompt = (test_prompt + 
                f"\nSo what you gave is not PBE, please check again whether the signal sender strategy {strategy} "
                f"can find a Perfect Bayesian Equilibrium PBE.\n\n"
                f"Original question: {signal_prompt}\n\n"
                f"Original answer: {signal_output}\n\n"
                "Please check again.")
            
            feedback_response = call_model(model_name, feedback_prompt)
            final_response = feedback_response
        
        # Final evaluation
        sender_strategy_list, receiver_strategy_list, p_list, q_list, index = pbe_re(final_response)
        evaluation_prompt = get_evaluation_prompt(strategy, final_response)
        evaluation_output = call_model(model_name, evaluation_prompt)
        
        # Extract the score
        score_match = re.search(r"'score': '([0-9]+)'", evaluation_output)
        if score_match:
            score += float(score_match.group(1))
            print(f"Score: {score}")
        else:
            score = 0
            print("No valid score found in the response.")
        
        # Final verification of whether it is a PBE
        NE = 1
        if sender_strategy_list:
            for i in range(len(sender_strategy_list)):
                sender_strategy = sender_strategy_list[i]
                receiver_strategy = receiver_strategy_list[i]
                p = p_list[i]
                q = q_list[i]
                
                if isinstance(sender_strategy, str):
                    sender_strategy = (sender_strategy[2:4], sender_strategy[-4:-2])
                if isinstance(receiver_strategy, str):
                    receiver_strategy = (receiver_strategy[2:4], receiver_strategy[-4:-2])
                
                expected_benefit_sender, expected_benefit_receiver, p, q = benefit(
                    sender_strategy, receiver_strategy, p, q)
                print(f"Final PBE: {expected_benefit_sender}, {expected_benefit_receiver}")
                
                sender_strategy_set = [('m1', 'm1'), ('m1', 'm2'), ('m2', 'm1'), ('m2', 'm2')]
                for sender_strategy_test in sender_strategy_set:
                    if sender_strategy_test != sender_strategy:
                        expected_benefit_sender_test, expected_benefit_receiver_test, p_test, q_test = benefit(
                            sender_strategy_test, receiver_strategy, p, q)
                        print(f"Final Test: {expected_benefit_sender_test}, {expected_benefit_receiver_test}")
                        if expected_benefit_sender_test > expected_benefit_sender:
                            score -= 2
                            NE = 0
                            print('Final: Not PBE!')
        
        # Save the results
        results.append({
            'strategy': strategy,
            'score': score,
            'quantity_response': final_response,
            'evaluation_response': evaluation_output,
            'sender_strategy_list': sender_strategy_list,
            'receiver_strategy_list': receiver_strategy_list,
            'p_list': p_list,
            'q_list': q_list,
            'signal_output': signal_output,
            'feedback_response': feedback_response,
            'NE': NE
        })
    
    return results