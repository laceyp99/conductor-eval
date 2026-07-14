import pytest
from mido import Message, MidiFile, MidiTrack

from conductor_eval.checks import monophony_test, polyphony_test


def make_midi(*tracks):
    midi = MidiFile(ticks_per_beat=480)
    for messages in tracks:
        track = MidiTrack(messages)
        midi.tracks.append(track)
    return midi


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
