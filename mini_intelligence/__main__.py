"""Command-line entry point for Mini Intelligence."""
import argparse
import json
import sys

from .router import MiniIntelligence


def build_parser():
    parser = argparse.ArgumentParser(prog="mini-intelligence",
                                     description="Select a model and show its predicted benchmark success.")
    parser.add_argument("prompt", nargs="?", help="Query text. Omit to read from standard input.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Default: CUDA when available.")
    parser.add_argument("--policy", choices=["raw", "gated"], default="raw",
                        help="Use the highest score (raw, default) or the saved validation margin "
                             "over the fallback model (gated).")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32",
                        help="float32 (default) or bfloat16 autocast on a CUDA GPU.")
    parser.add_argument("--model-dir", help="Local copy of the Hugging Face model folder instead of a download.")
    parser.add_argument("--offline", action="store_true", help="Use a previously cached model download.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    query = args.prompt if args.prompt is not None else sys.stdin.read()
    if not query.strip():
        parser.error("Provide a query as an argument or on standard input")
    router = MiniIntelligence(device=args.device, local_files_only=args.offline, model_dir=args.model_dir,
                              precision=args.precision)
    print(json.dumps(router.predict(query, policy=args.policy), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
