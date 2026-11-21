"""Ordinary SFT arms: stage-0 pi0, success-only, +corrective demos, balanced, PAFT-style, and EXP1 (balanced corrective SFT on the CACRT states).

Example (GPU): python scripts/train_balanced_sft.py --arm STAGE0 --model <path> --lora --val --save
Example (CPU smoke): python scripts/train_balanced_sft.py --arm STAGE0 --model tiny --limit 8 --steps 2
"""
from _train_common import main

if __name__ == "__main__":
    main(["STAGE0", "B1", "B2", "B3", "B4", "EXP1"], __doc__)
