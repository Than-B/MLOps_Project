# conftest.py — shared pytest configuration
import sys
from pathlib import Path

# ทำให้ pytest หา module ใน api/ ได้
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
