## SAR Environment Description


## Coding standards
Use static typing, check typing with mypy.
Use pylint to ensure code quality.

You can (and should) use ./run_autoformat.sh to automatically 
run black, docformatter, and isort.


## Installation Instructions
Create a virtual environment: uv venv --python=3.11
uv pip install -e ".[develop]"

# Manual testing
To interact with the environment and manually control a player:
run python interactive_sar.py

How to manually control a player:
- L/R arrow keys to rotate player
- forward arrow key to move
- left tab to pick up a person
- left shift to drop a person
- pick up a person and move to a goal for rewards + rescuing that person
- Note: lava acts as "collapsible floor"