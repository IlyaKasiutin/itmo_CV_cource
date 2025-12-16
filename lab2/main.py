#!/usr/bin/env python3
"""
Точка входа приложения Robust Object Tracker
"""

import argparse
from video_utils import process_video
import cv2

def main():
    parser = argparse.ArgumentParser(
        description="Robust object tracker (detection + LK tracking + Kalman)"
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='input video path')
    parser.add_argument('--output', '-o', 
                       help='output video path')
    parser.add_argument('--detector', choices=['sift', 'orb', 'akaze'], 
                       default='sift', help='feature detector')
    parser.add_argument('--show-matches', action='store_true',
                       help='show matches (debug)')
    parser.add_argument('--debug', action='store_true',
                       help='debug prints')
    parser.add_argument('--max-variants', type=int, default=8,
                       help='max template scales/angles variants (speed)')
    
    args = parser.parse_args()
    
    process_video(
        input_video=args.input,
        output_video=args.output,
        detector=args.detector,
        show_matches=args.show_matches,
        debug=args.debug,
        max_variants=args.max_variants
    )


if __name__ == "__main__":
    main()