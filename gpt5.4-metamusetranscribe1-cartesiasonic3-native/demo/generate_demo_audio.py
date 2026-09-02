"""Render the Babel demo conversation to audio with distinct voices per speaker.

This is a demo asset generator, not part of the agent. It uses Kokoro (ONNX)
so the whole conversation can be rendered offline: Babel gets one voice, and
each caller at the table gets their own. Hindi/English code-switched lines are
split by script so each run is phonemised in the right language.

Usage (one-off, no need to add anything to the example's dependencies):

    uv run --with kokoro-onnx --with soundfile --with scipy --with numpy \
        python demo/generate_demo_audio.py --model kokoro-v1.0.onnx --voices voices-v1.0.bin

Model files: https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0).
espeak-ng must be installed (apt install espeak-ng) and, if the bundled loader
cannot find its data, export ESPEAK_DATA_PATH=/usr/share/espeak-ng-data.

Outputs (in --out-dir, default: this folder):
    demo_conversation.wav / .mp3   full call, 24kHz mono
    conversation_script.md         the script with timestamps and pipeline notes
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
GAP_S = 0.45  # silence between turns
PHONE_BAND = (300.0, 3400.0)  # Babel is heard through the phone speaker

# Kokoro voice per speaker. The agent keeps ONE voice across languages; only the
# phonemiser language changes, which is how a single Cartesia voice behaves too.
VOICES = {
    "Babel": "af_heart",
    "Sam": "am_michael",
    "Carlos": "em_alex",
    "Priya": "hf_alpha",
}
KOKORO_LANG = {"en": "en-us", "es": "es", "hi": "hi"}


@dataclass
class Line:
    speaker: str
    lang: str  # dominant language: en | es | hi (agent lines carry the <xx> tag)
    text: str
    note: str = ""  # what the pipeline does on this line (shown in the script)
    cut_after_s: float | None = None  # agent line interrupted by barge-in
    barge_in_at_s: float | None = None  # caller starts talking this far into previous line
    extra: dict = field(default_factory=dict)


SCRIPT: list[Line] = [
    Line(
        "Babel",
        "en",
        "Hi, welcome to Bistro Mundo, I'm Babel. Put me on speaker. Everyone can "
        "order in any language you like, and I'll remember who said what.",
        note="Turn 1 greeting. Plivo `start` -> Muse session opened (g711_ulaw, diarization on, "
        "keyword biasing: menu names) -> GPT-5.4 mini streams `<en> ...` -> Cartesia Sonic-3 "
        "context, sentence by sentence -> Plivo playAudio -> checkpoint -> playedStream.",
    ),
    Line(
        "Sam",
        "en",
        "Hey Babel, it's Sam. There are three of us. I'll start with two tacos al pastor "
        "and a sparkling water.",
        note="Muse final transcript tagged `[Speaker 1]`, endpointed 0.4s after Sam stops. "
        "Tools: identify_speaker(1, Sam) -> Muse keyword biasing updated with 'Sam'; "
        "add_order_item(1, tacos al pastor, 2); add_order_item(1, sparkling water, 1).",
    ),
    Line(
        "Babel",
        "en",
        "Got it, Sam. Two tacos al pastor and a sparkling water for you. Who's next?",
        note="Reply tagged `<en>`; Cartesia language = en.",
    ),
    Line(
        "Carlos",
        "es",
        "Hola Babel, soy Carlos. Para mí la paella de mariscos, por favor. "
        "¿Es suficiente para dos personas?",
        note="New voice: Muse diarization assigns `[Speaker 2]` (event speaker_detected). "
        "Spanish, no language hint needed. Tools: identify_speaker(2, Carlos, es); "
        "add_order_item(2, seafood paella, 1).",
    ),
    Line(
        "Babel",
        "es",
        "Claro, Carlos. La paella de mariscos rinde para dos personas, y ya está anotada a tu nombre.",
        note="Reply tagged `<es>`; same Cartesia voice, Spanish synthesis.",
    ),
    Line(
        "Priya",
        "hi",
        "Hi, मैं Priya हूँ। मेरे लिए एक paneer tikka, extra spicy, और एक mango lassi please.",
        note="`[Speaker 3]`. Mid-sentence code-switching Hindi/English in ONE utterance, "
        "transcribed as one segment. Tools: identify_speaker(3, Priya, hi); "
        "add_order_item(3, paneer tikka, 1, 'extra spicy'); add_order_item(3, mango lassi, 1).",
    ),
    Line(
        "Babel",
        "hi",
        "ज़रूर Priya। एक पनीर टिक्का, एक्स्ट्रा स्पाइसी, और एक मैंगो लस्सी आपके नाम पर लिख दिया।",
        note="Reply tagged `<hi>`.",
    ),
    Line(
        "Babel",
        "en",
        "By the way, tonight we also have a chef's special, a saffron seafood stew with "
        "grilled bread and a little bit of",
        note="Babel keeps talking...",
        cut_after_s=3.2,
    ),
    Line(
        "Sam",
        "en",
        "Sorry Babel, quick change. Make mine three tacos, not two.",
        note="BARGE-IN. Sam talks over Babel: Silero VAD speech START (prob > 0.85, echo of "
        "Babel's own audio stays < 0.75) -> in-flight turn task cancelled, Cartesia context "
        "cancelled, send queue drained, `clearAudio` sent to Plivo, turn_complete(barge_in=true). "
        "Muse still endpoints Sam's sentence as `[Speaker 1 (Sam)]`.",
        barge_in_at_s=2.6,
    ),
    Line(
        "Babel",
        "en",
        "No problem, Sam. Three tacos al pastor for you now.",
        note="Tools: add_order_item(1, tacos al pastor, 3, 'replaces the earlier two').",
    ),
    Line(
        "Carlos",
        "es",
        "Babel, ¿me puedes repetir quién pidió qué?",
        note="`[Speaker 2 (Carlos)]`. Tool: get_order_summary() -> grouped by speaker label.",
    ),
    Line(
        "Babel",
        "es",
        "Por supuesto. Sam: tres tacos al pastor y un agua con gas. Carlos: la paella de "
        "mariscos. Priya: un paneer tikka extra picante y un mango lassi.",
        note="Read-back grouped by diarized speaker, in Carlos's language.",
    ),
    Line(
        "Priya",
        "hi",
        "Perfect. Babel, ticket text कर दो, and we're done.",
        note="`[Speaker 3 (Priya)]`, code-switched again. Tools: send_sms(caller number, ticket); "
        "end_call('order confirmed').",
    ),
    Line(
        "Babel",
        "hi",
        "हो गया Priya, टिकट आपके फ़ोन पर भेज दिया। Sam, Carlos, Priya, thank you, enjoy your dinner!",
        note="Reply tagged `<hi>`. end_call waits for Plivo `playedStream` so the goodbye "
        "finishes before the session ends. call_summary: 8 turns, 3 speakers, 1 barge-in.",
    ),
]

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_RUN_SPLIT = re.compile(r"([ऀ-ॿ][ऀ-ॿ\s।॥,.!?]*)")


def split_runs(text: str, dominant: str) -> list[tuple[str, str]]:
    """Split text into (lang, run) pairs by script so each run is phonemised correctly."""
    if not _DEVANAGARI.search(text):
        return [(dominant, text)]
    runs: list[tuple[str, str]] = []
    for part in _RUN_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        if _DEVANAGARI.search(part):
            runs.append(("hi", part))
        else:
            runs.append(("en", part))
    return runs


def bandpass(audio: np.ndarray, low: float, high: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [low, high], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def synth_line(kokoro, line: Line) -> np.ndarray:
    voice = VOICES[line.speaker]
    pieces: list[np.ndarray] = []
    for lang, run in split_runs(line.text, line.lang):
        audio, sr = kokoro.create(run, voice=voice, speed=1.0, lang=KOKORO_LANG[lang])
        assert sr == SR
        pieces.append(audio.astype(np.float32))
        pieces.append(np.zeros(int(0.06 * SR), dtype=np.float32))
    audio = np.concatenate(pieces)
    if line.cut_after_s is not None:
        n = int(line.cut_after_s * SR)
        fade = np.linspace(1.0, 0.0, int(0.08 * SR), dtype=np.float32)
        audio = audio[:n].copy()
        audio[-len(fade) :] *= fade
    if line.speaker == "Babel":
        audio = bandpass(audio, *PHONE_BAND) * 1.15
    peak = float(np.abs(audio).max()) or 1.0
    return (audio / peak * 0.8).astype(np.float32)


def render(kokoro, out_dir: Path) -> None:
    timeline: list[tuple[float, float, Line]] = []
    mix = np.zeros(0, dtype=np.float32)
    cursor = 0.0
    prev_start = 0.0
    for line in SCRIPT:
        audio = synth_line(kokoro, line)
        if line.barge_in_at_s is not None:
            start = prev_start + line.barge_in_at_s
        else:
            start = cursor + GAP_S
        start_i = int(start * SR)
        end_i = start_i + len(audio)
        if end_i > len(mix):
            mix = np.concatenate([mix, np.zeros(end_i - len(mix), dtype=np.float32)])
        mix[start_i:end_i] += audio
        timeline.append((start, end_i / SR, line))
        prev_start = start
        cursor = end_i / SR
        print(f"{start:6.2f}s  {line.speaker:<6} {line.text[:60]}")

    mix = np.clip(mix, -1.0, 1.0)
    wav = out_dir / "demo_conversation.wav"
    sf.write(wav, mix, SR)
    mp3 = out_dir / "demo_conversation.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "96k",
            str(mp3),
        ],
        check=True,
    )
    write_script(out_dir / "conversation_script.md", timeline, len(mix) / SR)
    print(f"\nwrote {wav} ({len(mix) / SR:.1f}s), {mp3}")


def write_script(path: Path, timeline: list[tuple[float, float, Line]], total_s: float) -> None:
    def ts(t: float) -> str:
        return f"{int(t // 60):02d}:{t % 60:05.2f}"

    lines = [
        "# Babel demo call: one phone, three people, three languages",
        "",
        f"Rendered conversation: `demo_conversation.mp3` ({total_s:.0f}s). "
        "Voices: Babel = Kokoro `af_heart` (one voice, three languages, heard through the "
        "phone band-pass), Sam = `am_michael`, Carlos = `em_alex`, Priya = `hf_alpha`.",
        "",
        "Scenario: three friends call Bistro Mundo, put the phone in the middle of the table "
        "and order in English, Spanish and Hindi. Meta Muse Voice Transcribe tags every "
        "utterance with a stable speaker label and endpoints each turn; GPT-5.4 mini answers "
        "each person in their own language; Cartesia Sonic-3 speaks it with one voice.",
        "",
        "| Time | Speaker | Line |",
        "|---|---|---|",
    ]
    for start, _end, line in timeline:
        tag = f"`<{line.lang}>` " if line.speaker == "Babel" else ""
        who = "**Babel**" if line.speaker == "Babel" else line.speaker
        cut = " *(interrupted)*" if line.cut_after_s else ""
        lines.append(f"| {ts(start)} | {who} | {tag}{line.text}{cut} |")
    lines += ["", "## What the pipeline does, line by line", ""]
    for start, _end, line in timeline:
        who = "Babel" if line.speaker == "Babel" else f"{line.speaker}"
        lines.append(f"- **{ts(start)} {who}**: {line.note}")
    lines += [
        "",
        "## Structured events emitted during this call",
        "",
        "| Event | Count | Notes |",
        "|---|---|---|",
        "| `call_answered` | 1 | session start, SIP headers |",
        "| `speaker_detected` | 3 | first time Muse labels Speaker 1, 2, 3 |",
        "| `speaker_identified` | 3 | names learned, Muse keyword biasing updated |",
        "| `user_text` | 7 | one per committed turn, speaker-tagged text |",
        "| `agent_text` | 8 | fired as soon as the LLM reply is known, before TTS |",
        "| `turn_complete` | 8 | per-turn llm_ms / tts_ttfb_ms / playback_ms, one with barge_in=true |",
        "| `call_summary` | 1 | turns, speakers, barge-ins, TTFS avg, errors |",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kokoro-v1.0.onnx")
    ap.add_argument("--voices", default="voices-v1.0.bin")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    from kokoro_onnx import Kokoro

    kokoro = Kokoro(args.model, args.voices)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render(kokoro, out_dir)


if __name__ == "__main__":
    main()
