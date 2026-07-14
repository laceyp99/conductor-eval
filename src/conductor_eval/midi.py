"""Utilities for inspecting MIDI files on a shared absolute timeline."""

from collections import deque
from dataclasses import dataclass

from mido import MidiFile


@dataclass(frozen=True, slots=True)
class NoteInterval:
    """One completed MIDI note represented on an absolute-tick timeline."""

    pitch: int
    velocity: int
    channel: int
    track: int
    start_tick: int
    end_tick: int

    @property
    def duration_ticks(self) -> int:
        """Return the note length in MIDI ticks."""
        return self.end_tick - self.start_tick


def extract_note_intervals(midi: MidiFile) -> list[NoteInterval]:
    """Return completed notes from every track on one absolute-tick timeline.

    Delta times are accumulated independently for each MIDI track. Repeated
    instances of the same pitch and channel are paired first-in, first-out,
    matching Conductor Core's MIDI parser. Unmatched note-off messages and
    notes left active at the end of a track are ignored.
    """
    intervals = []

    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        active_notes = {}

        for msg in track:
            absolute_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active_notes.setdefault(key, deque()).append((absolute_tick, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                starts = active_notes.get(key)
                if not starts:
                    continue

                start_tick, velocity = starts.popleft()
                intervals.append(
                    NoteInterval(
                        pitch=msg.note,
                        velocity=velocity,
                        channel=msg.channel,
                        track=track_index,
                        start_tick=start_tick,
                        end_tick=absolute_tick,
                    )
                )
                if not starts:
                    del active_notes[key]

    return sorted(
        intervals,
        key=lambda note: (
            note.start_tick,
            note.end_tick,
            note.track,
            note.channel,
            note.pitch,
        ),
    )


def calculate_polyphony_profile(midi: MidiFile) -> dict:
    """Measure simultaneous completed notes across every MIDI track."""
    intervals = extract_note_intervals(midi)
    events = {}
    for note in intervals:
        events[note.start_tick] = events.get(note.start_tick, 0) + 1
        events[note.end_tick] = events.get(note.end_tick, 0) - 1

    active_notes = 0
    max_polyphony = 0
    previous_tick = 0
    distribution_ticks = {}

    for tick in sorted(events):
        elapsed = tick - previous_tick
        if elapsed:
            distribution_ticks[active_notes] = distribution_ticks.get(active_notes, 0) + elapsed
        active_notes += events[tick]
        max_polyphony = max(max_polyphony, active_notes)
        previous_tick = tick

    total_duration_ticks = max((note.end_tick for note in intervals), default=0)
    distribution = {
        level: round(ticks / midi.ticks_per_beat, 4) for level, ticks in distribution_ticks.items()
    }
    percentages = {
        level: round(ticks / total_duration_ticks * 100, 2)
        for level, ticks in distribution_ticks.items()
        if total_duration_ticks > 0
    }
    return {
        "polyphony_distribution": distribution,
        "polyphony_percentages": percentages,
        "max_polyphony": max_polyphony,
        "total_duration": round(total_duration_ticks / midi.ticks_per_beat, 4),
        "ticks_per_beat": midi.ticks_per_beat,
    }
