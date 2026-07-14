from conductor_core.music import (
    DURATION_BEATS,
    SCALE_INTERVALS,
    note_name_to_pitch_class,
)

from conductor_eval.midi import calculate_polyphony_profile


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

    duration_ticks = DURATION_BEATS[duration]
    ticks_per_beat = midi.ticks_per_beat
    expected_ticks = duration_ticks * ticks_per_beat
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
                        ratio = note_duration / ticks_per_beat
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
