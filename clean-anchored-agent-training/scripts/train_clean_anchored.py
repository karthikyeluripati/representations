"""Clean-anchored arms: CACRT (EXP3) with twin anchors, its ablations, and generic-anchor / KL / PCGrad baselines.

Example (GPU): python scripts/train_clean_anchored.py --arm EXP3 --model <path> --lora --val --save
Example (CPU smoke): python scripts/train_clean_anchored.py --arm EXP3 --model tiny --limit 8 --steps 2
"""
from _train_common import main

if __name__ == "__main__":
    main(["EXP3", "EXP3_agem", "EXP3_unverified", "B7_kl", "B7_replay", "B8_pcgrad"], __doc__)
