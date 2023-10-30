#!/usr/bin/env python3

import argparse
from lib.diarizer import Diarizer
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Transcribe and diarize an audio file")
parser.add_argument(
    "audio_file", type=str, help="Path to the audio file to transcribe and diarize"
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="medium.en",
    help="Name of WhisperAI model to use for transcription",
)
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cuda",
    choices=["cpu", "cuda"],
    help='The device to use for inference (e.g., "cpu" or "cuda")',
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="data/output.json",
    help="Path to the output file",
)
parser.add_argument(
    "--print",
    action="store_true",
    help="Print the parsed output to stdout in addition to saving to file",
)
parser.add_argument(
    "--log",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level",
)

args = parser.parse_args()

diarizer = Diarizer(model=args.model, device=args.device, log_level=args.log)
diarizer.run(args.audio_file, output_file=args.output, print_results=args.print)
