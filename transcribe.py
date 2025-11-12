import argparse
import base64
import json
from pathlib import Path

from openai import OpenAI
import dotenv


def to_data_url(path: Path) -> str:
    """Convert an audio file to a data URL for speaker reference."""
    with open(path, "rb") as fh:
        return "data:audio/wav;base64," + base64.b64encode(fh.read()).decode("utf-8")


def transcribe(
    audio_path: Path,
    output_path: Path,
    enable_diarization: bool = False,
    speaker_names: list[str] | None = None,
    speaker_references: list[Path] | None = None,
) -> tuple[Path, Path]:
    """Send audio to OpenAI, save segmented text and raw JSON outputs."""
    dotenv.load_dotenv()
    client = OpenAI()

    with audio_path.open("rb") as audio_file:
        if enable_diarization:
            # Use diarization model
            extra_body = {}
            if speaker_names and speaker_references:
                if len(speaker_names) != len(speaker_references):
                    raise ValueError(
                        "Number of speaker names must match number of reference files"
                    )
                extra_body["known_speaker_names"] = speaker_names
                extra_body["known_speaker_references"] = [
                    to_data_url(ref) for ref in speaker_references
                ]

            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="gpt-4o-transcribe-diarize",
                response_format="diarized_json",
                chunking_strategy="auto",
                extra_body=extra_body if extra_body else None,
            )
        else:
            # Use standard Whisper model
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

    # Format output based on whether diarization was used
    lines: list[str] = []
    
    if enable_diarization:
        # Diarized output with speaker labels
        segments = getattr(transcription, "segments", []) or []
        if segments:
            for segment in segments:
                speaker = getattr(segment, "speaker", "Unknown")
                text = getattr(segment, "text", "").strip()
                start = getattr(segment, "start", 0)
                end = getattr(segment, "end", 0)
                if text:
                    lines.append(f"[{speaker}] ({start:.2f}s - {end:.2f}s): {text}")
        else:
            full_text = getattr(transcription, "text", "").strip()
            if full_text:
                lines.append(full_text)
    else:
        # Standard output without speaker labels
        segments = getattr(transcription, "segments", []) or []
        if segments:
            for segment in segments:
                text = getattr(segment, "text", "").strip()
                if text:
                    lines.append(text)
        else:
            full_text = getattr(transcription, "text", "").strip()
            if full_text:
                lines.append(full_text)

    output_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        transcription_payload = transcription.model_dump()
    except AttributeError:
        try:
            transcription_payload = transcription.dict()
        except AttributeError:
            transcription_payload = json.loads(transcription.json())

    json_output_path = (
        output_path.with_suffix(".json")
        if output_path.suffix
        else output_path.with_name(output_path.name + ".json")
    )
    json_output_path.write_text(
        json.dumps(transcription_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path, json_output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file via OpenAI Whisper with optional speaker diarization"
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default="recording_sayaka.m4a",
        help="Path to the audio file (default: recording_sayaka.m4a)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the text file to write (default: same name as audio with .txt)",
    )
    parser.add_argument(
        "-d",
        "--diarize",
        action="store_true",
        help="Enable speaker diarization using gpt-4o-transcribe-diarize",
    )
    parser.add_argument(
        "-n",
        "--speaker-names",
        nargs="+",
        help="Known speaker names (requires --speaker-references)",
    )
    parser.add_argument(
        "-r",
        "--speaker-references",
        nargs="+",
        help="Paths to speaker reference audio files (2-10 seconds each)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else audio_path.with_suffix(".txt")
    )

    speaker_names = args.speaker_names
    speaker_references = (
        [Path(ref).expanduser().resolve() for ref in args.speaker_references]
        if args.speaker_references
        else None
    )

    # Validate speaker references exist
    if speaker_references:
        for ref in speaker_references:
            if not ref.exists():
                raise FileNotFoundError(f"Speaker reference file not found: {ref}")

    text_path, json_path = transcribe(
        audio_path,
        output_path,
        enable_diarization=args.diarize,
        speaker_names=speaker_names,
        speaker_references=speaker_references,
    )
    
    print(f"Transcript written to {text_path}")
    print(f"Raw JSON written to {json_path}")
    if args.diarize:
        print("Speaker diarization was enabled")


if __name__ == "__main__":
    main()