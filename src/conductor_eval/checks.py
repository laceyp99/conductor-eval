from conductor_core.music import (
    DURATION_BEATS,
    SCALE_INTERVALS,
    note_name_to_pitch_class,
)

from conductor_eval.midi import (
    beats_to_ticks,
    calculate_polyphony_profile,
    extract_note_intervals,
    ticks_to_beats,
)

_ROMAN_DEGREES = {
    "I": 0,
    "IV": 3,
    "V": 4,
    "VI": 5,
}


def scale_test(midi, root, scale):
    """
    Test whether all note events in a MIDI file belong to the specified scale.

    Args:
        midi (MidiFile): The MIDI file to check.
        root (str): The root note (e.g., "C", "D#", etc.).
        scale (str): The scale mode ("Major" or "minor").

    Returns:
        dict: A dictionary containing the total number of notes, number of correct notes,
              number of incorrect notes, and lists of correct and incorrect pitch classes.

    Raises:
        ValueError: If the provided root note or scale mode is invalid.
    """
    # Validate root and scale.
    try:
        root_pc = note_name_to_pitch_class(root)
    except ValueError:
        raise ValueError(f"Invalid root note: {root}")
    if scale.lower() not in SCALE_INTERVALS:
        raise ValueError(f"Invalid scale mode: {scale.lower()}")

    # Determine the acceptable pitch classes for the given scale.
    acceptable_pcs = [(root_pc + interval) % 12 for interval in SCALE_INTERVALS[scale.lower()]]
    # print(f"Root Note: {root}, Scale Mode: {scale}, Acceptable Pitch Classes: {acceptable_pcs}")

    correct = 0
    incorrect = 0
    total = 0
    correct_pitches = set()
    incorrect_pitches = set()

    # Iterate through all messages in the MIDI file.
    for msg in midi:
        if msg.type == "note_on" and msg.velocity > 0:
            # print(f"Checking note: {msg.note}, Pitch Class: {msg.note % 12}")
            total += 1
            if (msg.note % 12) in acceptable_pcs:
                correct += 1
                correct_pitches.add(msg.note % 12)
            else:
                incorrect += 1
                incorrect_pitches.add(msg.note % 12)

    results = {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "pitches": {
            "correct": list(correct_pitches),
            "incorrect": list(incorrect_pitches),
        },
    }
    return results


def duration_test(midi, duration):
    """Test whether all note events in a MIDI file have the specified duration.

    Args:
        midi (MidiFile): The MIDI file to check.
        duration (str): The expected duration of each note event.

    Returns:
        dict: A dictionary containing the total number of notes, number of correct notes,
              number of incorrect notes, and a dictionary of incorrect note lengths.
    """
    if duration not in DURATION_BEATS:
        raise ValueError(f"Invalid duration: {duration}")

    ticks_per_beat = midi.ticks_per_beat
    expected_ticks = beats_to_ticks(DURATION_BEATS[duration], ticks_per_beat, "duration")
    # print(f"Expected duration in ticks: {expected_ticks}")

    total = 0
    correct = 0
    incorrect = 0
    incorrect_lengths = {}

    for track in midi.tracks:
        active_notes = {}
        current_time_ticks = 0
        for msg in track:
            current_time_ticks += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active_notes[msg.note] = current_time_ticks
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active_notes:
                    total += 1
                    start_time = active_notes.pop(msg.note)
                    note_duration = current_time_ticks - start_time
                    # print(f"Note {msg.note} duration: {note_duration} ticks")

                    if note_duration != expected_ticks:
                        incorrect += 1
                        ratio = ticks_to_beats(note_duration, ticks_per_beat, "note_duration")
                        if ratio not in incorrect_lengths.keys():
                            incorrect_lengths[ratio] = 1
                        else:
                            incorrect_lengths[ratio] += 1
                    else:
                        correct += 1
    results = {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "lengths": incorrect_lengths,
    }
    return results


def monophony_test(midi):
    """Test whether no completed notes overlap anywhere in the MIDI file."""
    profile = calculate_polyphony_profile(midi)
    passed = profile["max_polyphony"] <= 1
    return {
        "passed": passed,
        **profile,
    }


def polyphony_test(midi, min_voices=2):
    """Test whether the MIDI reaches a requested number of simultaneous voices."""
    if not isinstance(min_voices, int) or isinstance(min_voices, bool) or min_voices < 2:
        raise ValueError("min_voices must be an integer greater than or equal to 2")

    profile = calculate_polyphony_profile(midi)
    passed = profile["max_polyphony"] >= min_voices
    return {
        "passed": passed,
        "min_voices": min_voices,
        **profile,
    }


def _resolve_diatonic_triads(root, scale, progression):
    """Resolve supported scale degrees to pitch-class triads."""
    try:
        root_pc = note_name_to_pitch_class(root)
    except ValueError:
        raise ValueError(f"Invalid root note: {root}") from None

    scale_name = scale.lower()
    if scale_name not in {"major", "minor"}:
        raise ValueError("Chord progression checks support major and minor scales")
    if not isinstance(progression, list) or not progression:
        raise ValueError("progression must be a non-empty list of Roman numerals")

    scale_pcs = [(root_pc + interval) % 12 for interval in SCALE_INTERVALS[scale_name]]
    resolved = []
    for numeral in progression:
        if not isinstance(numeral, str) or numeral.upper() not in _ROMAN_DEGREES:
            supported = ", ".join(_ROMAN_DEGREES)
            raise ValueError(f"Unsupported Roman numeral {numeral!r}; expected one of: {supported}")
        degree = _ROMAN_DEGREES[numeral.upper()]
        pitch_classes = {
            scale_pcs[degree],
            scale_pcs[(degree + 2) % 7],
            scale_pcs[(degree + 4) % 7],
        }
        resolved.append((numeral, pitch_classes))
    return resolved


def chord_progression_test(
    midi,
    root,
    scale,
    progression,
    beats_per_chord=4,
    strict=True,
):
    """Test diatonic chord pitch classes at fixed harmonic boundaries.

    Roman-numeral case is treated as a scale-degree label. Chord quality is
    derived from the selected scale so all expected tones remain diatonic.
    """
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    chord_ticks = beats_to_ticks(beats_per_chord, midi.ticks_per_beat, "beats_per_chord")
    if chord_ticks == 0:
        raise ValueError("beats_per_chord must be greater than zero")

    expected_chords = _resolve_diatonic_triads(root, scale, progression)
    intervals = extract_note_intervals(midi)
    bar_results = []

    for index, (numeral, expected_pcs) in enumerate(expected_chords):
        onset_tick = index * chord_ticks
        actual_pcs = {
            note.pitch % 12 for note in intervals if note.start_tick <= onset_tick < note.end_tick
        }
        missing = expected_pcs - actual_pcs
        extra = actual_pcs - expected_pcs
        passed = not missing and (not strict or not extra)
        bar_results.append(
            {
                "bar": index + 1,
                "numeral": numeral,
                "onset_beat": ticks_to_beats(onset_tick, midi.ticks_per_beat),
                "expected_pitch_classes": sorted(expected_pcs),
                "actual_pitch_classes": sorted(actual_pcs),
                "missing_pitch_classes": sorted(missing),
                "extra_pitch_classes": sorted(extra),
                "passed": passed,
            }
        )

    return {
        "passed": all(bar["passed"] for bar in bar_results),
        "strict": strict,
        "bars": bar_results,
    }


def harmonic_rhythm_test(midi, expected_onsets):
    """Test that completed notes begin only at the expected beat positions."""
    if not isinstance(expected_onsets, list) or not expected_onsets:
        raise ValueError("expected_onsets must be a non-empty list")
    expected_ticks = {
        beats_to_ticks(beat, midi.ticks_per_beat, "expected_onsets") for beat in expected_onsets
    }
    if len(expected_ticks) != len(expected_onsets):
        raise ValueError("expected_onsets must contain unique beat positions")

    actual_ticks = {note.start_tick for note in extract_note_intervals(midi)}
    missing_ticks = expected_ticks - actual_ticks
    unexpected_ticks = actual_ticks - expected_ticks

    def to_beats(ticks):
        return [ticks_to_beats(tick, midi.ticks_per_beat) for tick in sorted(ticks)]

    return {
        "passed": not missing_ticks and not unexpected_ticks,
        "expected_onsets": to_beats(expected_ticks),
        "actual_onsets": to_beats(actual_ticks),
        "missing_onsets": to_beats(missing_ticks),
        "unexpected_onsets": to_beats(unexpected_ticks),
    }


def chord_event_positions_test(midi, expected_starts, expected_ends):
    """Test that every completed note uses an expected start/end beat pair."""
    if not isinstance(expected_starts, list) or not isinstance(expected_ends, list):
        raise ValueError("expected_starts and expected_ends must be lists")
    if not expected_starts or len(expected_starts) != len(expected_ends):
        raise ValueError("expected_starts and expected_ends must have the same non-zero length")

    expected_pairs = set()
    for start, end in zip(expected_starts, expected_ends):
        start_tick = beats_to_ticks(start, midi.ticks_per_beat, "expected_starts")
        end_tick = beats_to_ticks(end, midi.ticks_per_beat, "expected_ends")
        if end_tick <= start_tick:
            raise ValueError("Every expected end must be later than its start")
        expected_pairs.add((start_tick, end_tick))
    if len(expected_pairs) != len(expected_starts):
        raise ValueError("Expected start/end pairs must be unique")

    intervals = extract_note_intervals(midi)
    actual_pairs = {(note.start_tick, note.end_tick) for note in intervals}
    missing_pairs = expected_pairs - actual_pairs
    unexpected_pairs = actual_pairs - expected_pairs

    def serialize(pairs):
        return [
            {
                "start_beat": ticks_to_beats(start, midi.ticks_per_beat),
                "end_beat": ticks_to_beats(end, midi.ticks_per_beat),
            }
            for start, end in sorted(pairs)
        ]

    return {
        "passed": bool(intervals) and not missing_pairs and not unexpected_pairs,
        "expected_positions": serialize(expected_pairs),
        "actual_positions": serialize(actual_pairs),
        "missing_positions": serialize(missing_pairs),
        "unexpected_positions": serialize(unexpected_pairs),
    }
