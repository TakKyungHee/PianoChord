from music21 import stream, note, chord, duration, instrument, clef, meter
import sys
import subprocess


class PianoPlayer:
    def __init__(self, midifilename='untitled.mid'):
        self.midifilename = midifilename
        self.reset()

    def reset(self):
        self.midi = stream.Score()
        self.melody = stream.Part()
        self.melody.insert(0, instrument.Vocalist())
        self.treble = stream.Part()
        self.treble.insert(0, instrument.Piano())
        self.treble.insert(0, clef.TrebleClef())
        self.bass = stream.Part()
        self.bass.insert(0, instrument.Piano())
        self.bass.insert(0, clef.BassClef())

    def record(self, sequences, time_signatures, original_interval, measure_number=0, clef='melody'):
        if clef == 'treble':
            part = self.treble
        elif clef == 'bass':
            part = self.bass
        elif clef == 'melody':
            part = self.melody
        for i, seq in enumerate(list(sequences)):
            measure = stream.Measure(number=i+1)
            rhythms = 0
            for n in seq:
                if n <= 0:
                    continue
                rhythm = duration.Duration(n % 1000 / 100)
                if rhythms + (n % 1000 / 100) > time_signatures[measure_number+i]:
                    break
                rhythms += n % 1000 / 100
                n = n // 1000
                if n >= 100:
                    melodies = []
                    while n >= 1:
                        melodies.append(n % 100)
                        n = n//100
                    streamnote = chord.Chord(melodies)
                elif n >= 1:
                    streamnote = note.Note(n)
                else:
                    streamnote = note.Rest()
                streamnote.duration = rhythm
                measure.append(streamnote)
            if rhythms < time_signatures[measure_number+i]:
                measure.append(
                    note.Rest(quarterLength=time_signatures[measure_number+i]-rhythms))
            measure.append(meter.TimeSignature(
                f'{time_signatures[measure_number+i]}/4'))
            measure.transpose(
                original_interval[measure_number+i], inPlace=True)
            part.append(measure)

    def save(self, midifilename=None):
        self.midi.append(self.melody)
        self.midi.append(self.treble)
        self.midi.append(self.bass)
        if midifilename is None:
            midifilename = self.midifilename
        self.midi.write('midi', fp=midifilename)
        try:
            __import__('mido')
        except ImportError:
            print(f"{'mido'} not found, installing...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", 'mido'])
        import mido
        # MIDI 파일 열기
        mf = mido.MidiFile(midifilename)
        # 모든 트랙의 프로그램 변경 메시지를 피아노로 변경
        for track in mf.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    msg.program = 0  # Acoustic Grand Piano (Program Number 0)
        mf.save(midifilename)

    def play(self, midifilename=None):
        import pygame
        pygame.init()

        # 미디 파일 재생
        if midifilename is not None:
            pygame.mixer.music.load(midifilename)
        else:
            pygame.mixer.music.load(self.midifilename)
        pygame.mixer.music.play()

        # 재생이 끝날 때까지 대기
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
