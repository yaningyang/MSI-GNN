import argparse
from train import train_cv
from evaluate import evaluate_independent
from config import Config
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='train', choices=['train','eval'])
    p.add_argument('--ckpt', type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == 'train':
        agg, fold_metrics, histories, graph = train_cv(Config)
        print("Cross-validation aggregated metrics:")
        for k,v in agg.items():
            print(f"{k}: {v[0]:.4f} ± {v[1]:.4f}")
    elif args.mode == 'eval':
        if args.ckpt is None:
            raise RuntimeError("Provide --ckpt path for evaluation")
        evaluate_independent(Config, ckpt_path=args.ckpt)

if __name__ == "__main__":
    main()
