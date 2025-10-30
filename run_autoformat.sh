#!/bin/bash

# Reformat python files with black to ensure PEP 8 compliance
python -m black .

# Reformat docstrings to ensure PEP 257 compliance for consistency and readability of code documentation
docformatter -i -r . --exclude venv

# Sort import statements alphabetically and separated into sections
isort .
