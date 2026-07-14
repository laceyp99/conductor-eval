import pytest
from mido import Message, MidiFile, MidiTrack

from conductor_eval.checks import (
    chord_event_positions_test,
    chord_progression_test,
    harmonic_rhythm_test,
    monophony_test,
    polyphony_test,
    scale_test,
)

PROGRESSION = ["I", "V", "vi", "IV"]
EXPECTED_ONSETS = [0, 4, 8, 12]
EXPECTED_ENDS = [4, 8, 12, 16]


def make_midi(*tracks):
    midi = MidiFile(ticks_per_beat=480)
    for messages in tracks:
        track = MidiTrack(messages)
        midi.tracks.append(track)
    return midi


def make_timed_midi(notes):
    """Build one MIDI track from (pitch, start beat, end beat) tuples."""
    midi = MidiFile(ticks_per_beat=480)
    events = []
    for pitch, start_beat, end_beat in notes:
        events.append((int(start_beat * 480), 1, "note_on", pitch))
        events.append((int(end_beat * 480), 0, "note_off", pitch))
    events.sort()

    track = MidiTrack()
    previous_tick = 0
    for tick, _, event_type, pitch in events:
        velocity = 80 if event_type == "note_on" else 0
        track.append(
            Message(
                event_type,
                note=pitch,
                velocity=velocity,
                time=tick - previous_tick,
            )
        )
        previous_tick = tick
    midi.tracks.append(track)
    return midi


def block_chord_notes(chords):
    return [(pitch, bar * 4, (bar + 1) * 4) for bar, chord in enumerate(chords) for pitch in chord]


def run_harmonic_checks(midi, root, scale):
    return {
        "scale": scale_test(midi, root, scale),
        "chords": chord_progression_test(
            midi,
            root=root,
            scale=scale,
            progression=PROGRESSION,
        ),
        "rhythm": harmonic_rhythm_test(midi, expected_onsets=EXPECTED_ONSETS),
        "positions": chord_event_positions_test(
            midi,
            expected_starts=EXPECTED_ONSETS,
            expected_ends=EXPECTED_ENDS,
        ),
        "polyphony": polyphony_test(midi, min_voices=3),
    }


def test_monophony_test_passes_for_sequential_notes():
    midi = make_midi(
        [
            Message("note_on", note=60, velocity=80, time=0),
            Message("note_off", note=60, velocity=0, time=480),
            Message("note_on", note=62, velocity=80, time=0),
            Message("note_off", note=62, velocity=0, time=480),
        ]
    )

    result = monophony_test(midi)

    assert result["passed"] is True
    assert result["max_polyphony"] == 1


def test_monophony_test_detects_overlap_across_tracks():
    midi = make_midi(
        [
            Message("note_on", note=60, velocity=80, time=0),
            Message("note_off", note=60, velocity=0, time=480),
        ],
        [
            Message("note_on", note=67, velocity=80, time=240),
            Message("note_off", note=67, velocity=0, time=480),
        ],
    )

    result = monophony_test(midi)

    assert result["passed"] is False
    assert result["max_polyphony"] == 2


def test_polyphony_test_passes_for_requested_voice_count():
    midi = make_midi(
        [
            Message("note_on", note=60, velocity=80, time=0),
            Message("note_on", note=64, velocity=80, time=0),
            Message("note_on", note=67, velocity=80, time=0),
            Message("note_off", note=60, velocity=0, time=480),
            Message("note_off", note=64, velocity=0, time=0),
            Message("note_off", note=67, velocity=0, time=0),
        ]
    )

    result = polyphony_test(midi, min_voices=3)

    assert result["passed"] is True
    assert result["max_polyphony"] == 3
    assert result["min_voices"] == 3


def test_polyphony_test_fails_below_requested_voice_count():
    midi = make_midi(
        [
            Message("note_on", note=60, velocity=80, time=0),
            Message("note_on", note=67, velocity=80, time=0),
            Message("note_off", note=60, velocity=0, time=480),
            Message("note_off", note=67, velocity=0, time=0),
        ]
    )

    result = polyphony_test(midi, min_voices=3)

    assert result["passed"] is False
    assert result["max_polyphony"] == 2


@pytest.mark.parametrize("min_voices", [True, 1, 2.5])
def test_polyphony_test_rejects_invalid_minimum(min_voices):
    with pytest.raises(ValueError, match="min_voices"):
        polyphony_test(make_midi(), min_voices=min_voices)


def test_major_block_chord_progression_passes_all_harmonic_checks():
    midi = make_timed_midi(
        block_chord_notes(
            [
                [60, 64, 67],  # C
                [55, 59, 62],  # G
                [57, 60, 64],  # Am
                [53, 57, 60],  # F
            ]
        )
    )

    results = run_harmonic_checks(midi, root="C", scale="major")

    assert results["scale"]["incorrect"] == 0
    assert results["chords"]["passed"] is True
    assert results["rhythm"]["passed"] is True
    assert results["positions"]["passed"] is True
    assert results["polyphony"]["passed"] is True


def test_minor_block_chord_progression_derives_diatonic_qualities():
    midi = make_timed_midi(
        block_chord_notes(
            [
                [57, 60, 64],  # Am
                [52, 55, 59],  # Em
                [53, 57, 60],  # F
                [50, 53, 57],  # Dm
            ]
        )
    )

    results = run_harmonic_checks(midi, root="A", scale="minor")

    assert results["scale"]["incorrect"] == 0
    assert results["chords"]["passed"] is True
    assert results["rhythm"]["passed"] is True
    assert results["positions"]["passed"] is True
    assert results["polyphony"]["passed"] is True


def test_chord_progression_reports_wrong_chord_by_bar():
    midi = make_timed_midi(
        block_chord_notes(
            [
                [60, 64, 67],
                [55, 59, 62],
                [50, 53, 57],  # Dm instead of Am
                [53, 57, 60],
            ]
        )
    )

    result = chord_progression_test(
        midi,
        root="C",
        scale="major",
        progression=PROGRESSION,
    )

    assert result["passed"] is False
    assert result["bars"][2] == {
        "bar": 3,
        "numeral": "vi",
        "onset_beat": 8.0,
        "expected_pitch_classes": [0, 4, 9],
        "actual_pitch_classes": [2, 5, 9],
        "missing_pitch_classes": [0, 4],
        "extra_pitch_classes": [2, 5],
        "passed": False,
    }


def test_harmonic_rhythm_rejects_midbar_onset():
    notes = block_chord_notes([[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]])
    notes.append((62, 2, 3))
    midi = make_timed_midi(notes)

    result = harmonic_rhythm_test(midi, expected_onsets=EXPECTED_ONSETS)

    assert result["passed"] is False
    assert result["unexpected_onsets"] == [2.0]


def test_chord_event_positions_rejects_short_note():
    notes = block_chord_notes([[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]])
    notes.remove((60, 0, 4))
    notes.append((60, 0, 3))
    midi = make_timed_midi(notes)

    result = chord_event_positions_test(
        midi,
        expected_starts=EXPECTED_ONSETS,
        expected_ends=EXPECTED_ENDS,
    )

    assert result["passed"] is False
    assert result["unexpected_positions"] == [{"start_beat": 0.0, "end_beat": 3.0}]


def test_chord_progression_strict_mode_rejects_extension():
    notes = block_chord_notes([[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]])
    notes.append((71, 0, 4))  # B extends C major to Cmaj7.
    midi = make_timed_midi(notes)

    result = chord_progression_test(
        midi,
        root="C",
        scale="major",
        progression=PROGRESSION,
    )

    assert result["passed"] is False
    assert result["bars"][0]["extra_pitch_classes"] == [11]
