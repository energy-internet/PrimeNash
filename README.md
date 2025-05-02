# PrimeNash

A game theory analysis tool powered by Large Language Models.

## Project Overview

This project is a tool designed to analyze and solve various game theory problems using Large Language Models (LLMs). It supports analysis of multiple types of games, including:

- Complete information static games
- Incomplete information static games
- Complete information dynamic games
- Incomplete information dynamic games

## Supported Game Models

- Cournot Model
- First-Price Sealed-Bid Auction
- Hawk-Dove Game
- New product Release Game
- Stackelberg Model
- Battle of Sex
- Spence Signaling Game
- Carbon Market Game


## Supported LLMs

- GPT-o1-Mini
- GPT-4o
- GPT-4o-Mini
- Claude-Sonnet
- Gemini
- Qwen



## Abstract

This project is inspired by a research study on the automated derivation of closed-form Nash equilibria using AI. The study introduces a framework that combines strategy generation, evaluation, and equilibrium proof modules to iteratively derive and validate solutions for classical game-theoretic problems. While the framework has been successfully applied to seven canonical game scenarios, including a complex carbon market bidding scenario, key details have been abstracted to protect intellectual property and research innovations.


## Project Structure

```
game_theory_llm/
├── config/
│   └── conf.json         # Configuration file
├──  data/      
├── src/
│   ├── prompts/          # Prompts for various game analyses
│   ├── utils/            # Utility functions
│   ├── run_game_models.py # Main executable file
│   └── game_classification.py # Game type classification
├──  tests/                # Test files
└── requirements.txt
```

## Usage Example

```python
from run_game_models import main

# Run analysis for the Cournot game
main(model_name='gemini', game_type='cournot', num_simulations=10)

# Run analysis for the Signaling game
main(model_name='gpt-4', game_type='signaling', num_simulations=10)
```

