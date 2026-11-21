"""Localized corrective training without anchoring: action-only SFT (B5) and verified counterfactual DPO (EXP2).

Example (GPU): python scripts/train_localized.py --arm B5 --model <path> --lora --val --save
Example (CPU smoke): python scripts/train_localized.py --arm B5 --model tiny --limit 8 --steps 2
"""
from _train_common import main

if __name__ == "__main__":
    main(["B5", "EXP2"], __doc__)
