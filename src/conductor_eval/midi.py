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
