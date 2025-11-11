import argparse
import json
from pathlib import Path

from openai import OpenAI
import dotenv


def transcribe(audio_path: Path, output_path: Path) -> tuple[Path, Path]:
    """Send audio to OpenAI, save segmented text and raw JSON outputs."""
    dotenv.load_dotenv()
    client = OpenAI()

    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = getattr(transcription, "segments", []) or []
    lines: list[str] = []
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
        description="Transcribe an .m4a file via OpenAI Whisper"
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default="recording_sayaka.m4a",
        help="Path to the .m4a audio file (default: recording_sayaka.m4a)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the text file to write (default: same name as audio with .txt)",
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

    text_path, json_path = transcribe(audio_path, output_path)
    print(f"Transcript written to {text_path}")
    print(f"Raw JSON written to {json_path}")


if __name__ == "__main__":
    main()
