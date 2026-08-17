"""
conftest.py — sys.path fix so pytest can import project modules. 

Per planner §8 and §9: this is NON-NEGOTIABLE. Plain `pytest` must work
without `python -m pytest`. This exact gap cost points on past assignments.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
