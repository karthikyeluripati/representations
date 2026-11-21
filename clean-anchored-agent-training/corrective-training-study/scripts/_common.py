import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, ".."))
PROJ = os.path.abspath(os.path.join(STUDY, ".."))
for p in (PROJ, os.path.join(STUDY, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
