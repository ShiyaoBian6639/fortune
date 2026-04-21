"""
Standalone script: predict next-day change for all stocks using a trained model.

Usage:
    python -m dl.predict_all
    python -m dl.predict_all --model_path stock_data/transformer_classifier.pth
    python -m dl.predict_all --top_n 50 --output predictions.csv
"""

import argparse
import os
import torch

from .config import get_config
from .predict import predict_all_stocks


def main():
    parser = argparse.ArgumentParser(description='Predict all stocks with trained model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to .pth checkpoint (default: stock_data/transformer_classifier.pth)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root data directory (default: from config)')
    parser.add_argument('--output', type=str, default='stock_predictions_all.csv',
                        help='Output CSV file (default: stock_predictions_all.csv)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Inference batch size (default: 512)')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Number of top/bottom predictions to print (default: 20)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference (default: use CUDA if available)')
    args = parser.parse_args()

    config     = get_config()
    data_dir   = args.data_dir or config['data_dir']
    model_path = args.model_path or os.path.join(data_dir, 'transformer_classifier.pth')
    device     = 'cpu' if args.cpu else config['device']

    predict_all_stocks(
        model_path = model_path,
        data_dir   = data_dir,
        output_csv = args.output,
        batch_size = args.batch_size,
        device     = device,
        top_n      = args.top_n,
    )


if __name__ == '__main__':
    main()
