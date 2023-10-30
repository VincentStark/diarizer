import os
import json
import logging
from typing import List, Dict, Any
from whisper import load_model
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


class Diarizer:
    def __init__(self, model: str = "medium.en", device: str = "cuda", log_level: str = "INFO"):
        self.huggingface_token = os.environ["HUGGINGFACE_TOKEN"]
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)
        
    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        self.logger.info("Transcribing")
        model = load_model(self.model, device=self.device)
        result = model.transcribe(audio_file, verbose=False)
        return {"segments": result["segments"], "language_code": result["language"]}

    def align_segments(
        self,
        segments: List[Dict[str, Any]],
        language_code: str,
        audio_file: str,
    ) -> Dict[str, Any]:
        self.logger.info("Aligning Segments")
        model, metadata = load_align_model(language_code=language_code, device=self.device)
        return align(segments, model, metadata, audio_file, self.device, print_progress=True, combined_progress=True)

    def diarize(self, audio_file: str) -> Dict[str, Any]:
        self.logger.info("Diarizing")
        pipeline = DiarizationPipeline(use_auth_token=self.huggingface_token, device=self.device)
        return pipeline(audio_file)

    def assign_speakers(
        self,
        diarization_result: Dict[str, Any],
        aligned_segments: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        self.logger.info("Assigning Speakers")
        result_segments = assign_word_speakers(diarization_result, aligned_segments)
        return [
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
                "speaker": segment.get("speaker", "UNKNOWN"),
            }
            for segment in result_segments["segments"]
        ]

    def transcribe_and_diarize(
        self, audio_file: str, print_results: bool = False
    ) -> List[Dict[str, Any]]:
        transcript = self.transcribe(audio_file)
        aligned_segments = self.align_segments(transcript["segments"], transcript["language_code"], audio_file)
        diarization_result = self.diarize(audio_file)
        segments_w_speakers = self.assign_speakers(diarization_result, aligned_segments)

        if print_results:
            for i, segment in enumerate(segments_w_speakers):
                print(f"Segment {i + 1}:")
                print(f"Start time: {segment['start']:.2f}")
                print(f"End time: {segment['end']:.2f}")
                print(f"Speaker: {segment['speaker']}")
                print(f"Transcript: {segment['text']}\n")

        return segments_w_speakers

    def run(self, audio_file: str, output_file: str, print_results: bool = False):
        transcription = self.transcribe_and_diarize(audio_file, print_results)

        with open(output_file, "w") as f:
            json.dump(transcription, f)
