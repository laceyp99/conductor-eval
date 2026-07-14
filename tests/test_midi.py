from mido import Message, MidiFile, MidiTrack

from conductor_eval.midi import calculate_polyphony_profile, extract_note_intervals


def test_extract_note_intervals_uses_absolute_time_across_tracks():
    midi = MidiFile(ticks_per_beat=480)

    first_track = MidiTrack()
    first_track.append(Message("note_on", note=60, velocity=90, time=480))
    first_track.append(Message("note_off", note=60, velocity=0, time=240))
    midi.tracks.append(first_track)

    second_track = MidiTrack()
    second_track.append(Message("note_on", note=67, velocity=80, time=480))
    second_track.append(Message("note_off", note=67, velocity=0, time=480))
    midi.tracks.append(second_track)

    intervals = extract_note_intervals(midi)

    assert [(note.pitch, note.track, note.start_tick) for note in intervals] == [
        (60, 0, 480),
        (67, 1, 480),
    ]
    assert [note.duration_ticks for note in intervals] == [240, 480]


def test_extract_note_intervals_treats_zero_velocity_note_on_as_note_off():
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    track.append(Message("note_on", note=64, velocity=72, channel=3, time=120))
    track.append(Message("note_on", note=64, velocity=0, channel=3, time=360))
    midi.tracks.append(track)

    intervals = extract_note_intervals(midi)

    assert len(intervals) == 1
    assert intervals[0].pitch == 64
    assert intervals[0].velocity == 72
    assert intervals[0].channel == 3
    assert intervals[0].start_tick == 120
    assert intervals[0].end_tick == 480
    assert intervals[0].duration_ticks == 360


def test_calculate_polyphony_profile_merges_tracks():
    midi = MidiFile(ticks_per_beat=480)

    first_track = MidiTrack()
    first_track.append(Message("note_on", note=60, velocity=90, time=0))
    first_track.append(Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(first_track)

    second_track = MidiTrack()
    second_track.append(Message("note_on", note=67, velocity=80, time=240))
    second_track.append(Message("note_off", note=67, velocity=0, time=480))
    midi.tracks.append(second_track)

    profile = calculate_polyphony_profile(midi)

    assert profile["max_polyphony"] == 2
    assert profile["polyphony_distribution"] == {1: 1.0, 2: 0.5}
    assert profile["polyphony_percentages"] == {1: 66.67, 2: 33.33}
    assert profile["total_duration"] == 1.5
