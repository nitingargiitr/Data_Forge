#!/usr/bin/env python3
"""
Main execution script for Contextual Compression System
DataForge E-Summit 2026
"""

import argparse
import sys
import os

# Add project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from hierarchical_compressor import HierarchicalCompressor


def main():
    parser = argparse.ArgumentParser(
        description="Contextual Compression for Extreme Long Inputs"
    )

    parser.add_argument(
        "-i", "--input",
        default=r"D:\DataForge\contextual_compression\data\raw_pdfs\check.pdf",
        help="Path to input PDF"
    )

    parser.add_argument(
        "-o", "--output",
        default="outputs",
        help="Output directory"
    )

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    print("\n🚀 Initializing compression system...\n")

    compressor = HierarchicalCompressor(output_dir=args.output)

    try:
        report = compressor.compress_document(args.input)
        outputs = compressor.export()

        print("\n✅ SUCCESS!")
        print("Generated files:\n")

        for k, v in outputs.items():
            print(f"  [{k}] → {v}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()