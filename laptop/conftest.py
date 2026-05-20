"""Put the laptop/ directory on sys.path so `import slam...` and `import
nl_parser` resolve when pytest is run from anywhere in the repo."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
