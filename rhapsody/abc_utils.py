"""ABC notation ↔ MIDI conversion utilities for Rhapsody symbolic music generation."""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def validate_abc(text: str) -> bool:
    """Check whether text parses as valid ABC notation (basic syntax check)."""
    required_header = False
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("X:") or line.startswith("T:"):
            required_header = True
        if line.startswith("K:") or line.startswith("k:"):
            return required_header
    return False


def extract_abc_from_generated(text: str) -> str:
    """Strip control tokens and extract the ABC portion from model output."""
    cleaned = text
    for tok in ("<|music|>", "<|abc_start|>", "<|abc_end|>", "<|midi_start|>", "<|midi_end|>"):
        cleaned = cleaned.replace(tok, "")
    lines = cleaned.strip().splitlines()
    abc_lines: list[str] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if not capture:
            if stripped and (stripped.startswith("X:") or stripped.startswith("K:") or stripped.startswith("T:")):
                capture = True
                abc_lines.append(stripped)
        else:
            abc_lines.append(stripped)
    return "\n".join(abc_lines).strip()


def abc_to_midi(abc_text: str, output_path: Optional[str] = None) -> Optional[bytes]:
    """Convert ABC notation string to MIDI using abc2midi (abc2midi package).

    Returns MIDI bytes if successful, None otherwise.
    If output_path is given, writes the file to disk.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".abc", delete=False) as f:
        f.write(abc_text)
        abc_path = Path(f.name)

    midi_path = abc_path.with_suffix(".mid")
    try:
        result = subprocess.run(
            ["abc2midi", str(abc_path), "-o", str(midi_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        midi_bytes = midi_path.read_bytes()
        if output_path is not None:
            Path(output_path).write_bytes(midi_bytes)
        return midi_bytes
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        abc_path.unlink(missing_ok=True)
        midi_path.unlink(missing_ok=True)


def midi_to_abc(midi_path: str) -> Optional[str]:
    """Convert a MIDI file to ABC notation using midi2abc (abc2midi package).

    Returns the ABC string if successful, None otherwise.
    """
    try:
        result = subprocess.run(
            ["midi2abc", str(midi_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def render_abc_to_audio(
    abc_text: str,
    output_path: str,
    audio_format: str = "wav",
) -> bool:
    """Render ABC notation to audio via abc2midi + fluidsynth.

    Requires abc2midi and fluidsynth to be installed, plus a SoundFont file.

    Returns True on success.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".abc", delete=False) as f:
        f.write(abc_text)
        abc_path = Path(f.name)

    midi_path = abc_path.with_suffix(".mid")
    try:
        midi_result = subprocess.run(
            ["abc2midi", str(abc_path), "-o", str(midi_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if midi_result.returncode != 0:
            return False

        soundfont_candidates = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default-GM.sf2",
            "/usr/share/sounds/sf2/GeneralUser.sf2",
        ]
        soundfont = next((s for s in soundfont_candidates if Path(s).exists()), None)
        if soundfont is None:
            print("[Rhapsody] WARNING: No SoundFont found; cannot render audio via fluidsynth.")
            return False

        audio_result = subprocess.run(
            [
                "fluidsynth",
                "-ni",
                soundfont,
                str(midi_path),
                "-F", str(output_path),
                "-r", "44100",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return audio_result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        abc_path.unlink(missing_ok=True)
        midi_path.unlink(missing_ok=True)


def format_abc_with_header(
    body: str,
    title: str = "",
    composer: str = "",
    key: str = "",
    meter: str = "",
    tempo: int = 0,
) -> str:
    """Wrap an ABC note body with header fields."""
    lines: list[str] = []
    lines.append("X:1")
    if title:
        lines.append(f"T:{title}")
    if composer:
        lines.append(f"C:{composer}")
    if meter:
        lines.append(f"M:{meter}")
    if key:
        lines.append(f"K:{key}")
    else:
        lines.append("K:C")  # default C major
    if tempo:
        lines.append(f"Q:1/4={tempo}")
    lines.append(body.strip())
    return "\n".join(lines)
