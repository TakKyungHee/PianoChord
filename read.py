import warnings
from music21 import instrument, converter, note, chord, stream, interval, pitch
import os
import pickle

warnings.filterwarnings("ignore", category=UserWarning, module='music21')


class note_token:
    def __init__(self):
        self.vocab = [0]

    def __len__(self):
        return len(self.vocab)

    def add(self, note):
        if note not in self.vocab:
            self.vocab.append(note)
            self.vocab.sort()

    def tonote(self, sequence):
        return [self.vocab[i] for i in sequence]


def get_mxl(path):
    return [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.mxl')]


def load_mxl(files):
    scores = []
    for f in files:
        scores.append(converter.parse(f))
    return scores


def transpose(score):
    measures = score.parts[0].getElementsByClass(stream.Measure)
    key_signature = measures[0].keySignature
    if key_signature is None:
        for i in range(len(measures)):
            original_interval.append(0)
    else:
        if key_signature.asKey().mode == 'major':
            interv = pitch.Pitch('C')
        else:
            interv = pitch.Pitch('A')
        interv = interval.Interval(key_signature.asKey().tonic, interv)
        for i in range(len(measures)):
            original_interval.append(-interv.semitones)
        score = score.transpose(interv)
    return score


def extract_melody_and_piano(scores):
    melody_sequences = []
    treble_sequences = []
    bass_sequences = []
    all_sequences = [melody_sequences, treble_sequences, bass_sequences]
    score_length = []
    for score in scores:
        score = transpose(score)
        for i, part in enumerate(score.parts):
            if i >= 3:
                break
            if len(time_signatures) > 0:
                time_signature = time_signatures[-1]
            else:
                time_signature = 4
            measures = part.getElementsByClass(stream.Measure)
            if i == 0:
                accompany_measures = []
                for j, measure in enumerate(measures):
                    if not any(isinstance(element, note.Note) for element in measure.flat.notesAndRests):
                        accompany_measures.append(j)
                if len(score_length) > 0:
                    score_length.append(
                        (len(measures)+score_length[-1][0], accompany_measures))
                else:
                    score_length.append(
                        (len(measures), accompany_measures))
            # instr = part.getInstrument()
            target_sequences = all_sequences[i]
            # if isinstance(instr, instrument.Vocalist) or i == 0:
            #     target_sequences = melody_sequences
            # elif isinstance(instr, instrument.Piano) and i == 1:
            #     target_sequences = treble_sequences
            # elif isinstance(instr, instrument.Piano) and i == 2:
            #     target_sequences = bass_sequences
            for measure in measures:
                if i == 0:
                    if measure.getElementsByClass('TimeSignature'):
                        time_signature = int(4*measure.getElementsByClass('TimeSignature')[
                            0].numerator/measure.getElementsByClass('TimeSignature')[
                                0].denominator)
                    time_signatures.append(
                        time_signature)
                target_sequence = []
                for element in measure.flat.notesAndRests:
                    target_sequence.append(convert_element_to_token(element))
                target_sequences.append(target_sequence)
    for j, target_sequences in enumerate(all_sequences):
        if len(target_sequences) == 0:
            del all_sequences[j:]
        for i, target_sequence in enumerate(target_sequences):
            target_sequences[i] = list(map(
                lambda x: token.vocab.index(x), target_sequence))
    return tuple(all_sequences), tuple(score_length), tuple(time_signatures), tuple(original_interval)


def convert_element_to_token(element):
    if isinstance(element, note.Note):
        element = 1000*element.pitch.midi+100*element.quarterLength
    elif isinstance(element, note.Rest):
        element = 100*element.quarterLength
    elif isinstance(element, chord.Chord):
        element = sum([1000*a*(10**(2*n)) for n, a in enumerate(
            [p.midi for p in element.pitches])])+100*element.quarterLength
    token.add(int(element))
    return int(element)


def read_files(files):
    scores = load_mxl(files)
    return extract_melody_and_piano(scores)


original_interval = []
time_signatures = []
token = note_token()


# 파일에서 리스트 불러오기
if os.path.exists('token.pkl'):
    with open('token.pkl', 'rb') as f:
        token = pickle.load(f)
