#!/bin/bash

# Training

# Inference / Demonstration
metagpt "Write a cli snake game"
metagpt "Write a game with pygame similar to the snake game" --run-tests
metagpt "Write a cli snake game based on pygame" --code-review
python examples/example.py
python examples/debate.py
python examples/oss_management.py
python examples/search_usage.py
python examples/redis_example.py
python examples/travel_planner.py --city Beijing --date 2024-01-01
python examples/agent_creator.py
python examples/invoice_ocr.py
python examples/di/data_interpreter.py
python examples/di/machine_learning.py
python examples/aflow/optimize_llama.py
python examples/werewolf_game/werewolf_game.py

# Testing / Evaluation
pytest tests/
pytest tests/metagpt/actions/test_action.py
```