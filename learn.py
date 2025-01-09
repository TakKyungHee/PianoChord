from torch.utils.data import DataLoader, TensorDataset
import read
import os
from read import token
import pickle
from play import PianoPlayer
import math
import matplotlib
import matplotlib.pyplot as plt
import warnings
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore", category=UserWarning,
                        message="enable_nested_tensor is True, but self.use_nested_tensor is False because")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="Initializing zero-element tensors is a no-op")


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, time_signatures, indexs, accompany=False) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_key_padding_mask: Tensor, shape ``[batch_size, seq_len]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_mask).transpose(0, 1)
        # (batch_size, seq_len, d_model)
        output = self.decoder(output)
        if accompany == False:
            output = postprocess(output, time_signatures, indexs)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class IndexDataset(TensorDataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return tuple([data[index] for data in self.data]+[index])


def create_src_mask(src):
    return (src == 0).float()


def postprocess(output, time_signatures, indexs):
    for j, seq in enumerate(torch.argmax(output, dim=-1)):
        notes = token.tonote(seq)
        rhythm = 0
        for i, note in enumerate(notes):
            rhythm += note % 1000 / 100
            if rhythm >= time_signatures[indexs[j]] and i < len(notes)-1:
                output[j][i+1:] = output[j][i+1:]/1.1
                output[j][i+1:][0] += 1
                break
    return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
lr = 0.001
ntokens = len(token)  # 단어 사전(어휘집)의 크기
print(f'ntokens : {ntokens}')  # 4096
emsize = 128  # 임베딩 차원
# ``nn.TransformerEncoder`` 에서 피드포워드 네트워크(feedforward network) 모델의 차원
d_hid = 256
nlayers = 6  # ``nn.TransformerEncoder`` 내부의 nn.TransformerEncoderLayer 개수
nhead = 8  # ``nn.MultiheadAttention`` 의 헤드 개수
dropout = 0.1  # 드랍아웃(dropout) 확률

# 모델
treble_model = TransformerModel(ntokens, emsize, nhead, d_hid,
                                nlayers, dropout).to(device)
bass_model = TransformerModel(ntokens, emsize, nhead, d_hid,
                              nlayers, dropout).to(device)
# 모델의 state_dict 불러오기
if os.path.exists('./treble_model.pth'):
    treble_model.load_state_dict(torch.load(
        './treble_model.pth', map_location=device, weights_only=True))
    print('Treble Model loaded')
if os.path.exists('./bass_model.pth'):
    bass_model.load_state_dict(torch.load(
        './bass_model.pth', map_location=device, weights_only=True))
    print('Bass Model loaded')


# 손실 함수
criterion = nn.CrossEntropyLoss()  # 필요에 따라 손실 함수 변경
treble_optimizer = optim.Adam(treble_model.parameters(), lr=lr)
bass_optimizer = optim.Adam(bass_model.parameters(), lr=lr)


# 예시 데이터 로더 설정 (실제 데이터에 맞게 조정 필요)
def preprocess(sequences):
    sequence = list(sequences)
    # min_len은 melody, treble, bass의 각 마디개수이므로, 다 똑같음.
    min_len = min([len(seq) for seq in sequence])
    for i, seq in enumerate(sequence):
        sequence[i] = pad_sequence([torch.tensor(s, device=device)
                                    for s in seq][:min_len], padding_value=0)
    # max_len = 19
    max_len = max([seq.size(0) for seq in sequence])
    for i, seq in enumerate(sequence):
        sequence[i] = torch.cat([seq, torch.zeros(
            max_len - seq.size(0), seq.size(1), dtype=torch.long, device=device)], dim=0).transpose(0, 1)
    return tuple(sequence), max_len


def combine_sequences_to_scores(*sequences):
    sequences = list(sequences)
    score_length = sequences.pop()
    all_scores = []
    for sequence in sequences:
        scores = []
        for i in range(len(score_length)):
            if i > 0:
                scores.append(
                    list(sequence[score_length[i-1][0]:score_length[i][0]].reshape(-1)))
            else:
                scores.append(
                    list(sequence[:score_length[i][0]].reshape(-1)))
        all_scores.append(scores)
    all_scores, _ = preprocess(all_scores)
    return tuple(all_scores)


def devide_scores_to_sequences(scores, score_length, measure_length, indexs):
    sequences = []
    for j in range(len(scores)):
        if len(score_length) <= indexs[j]:
            break
        score_len = score_length[indexs[j]][0]-score_length[indexs[j]-1
                                                            ][0] if indexs[j] > 0 else score_length[indexs[j]][0]
        for i in range(score_len):
            sequences.append(
                list(scores[j][i*measure_length:(i+1)*measure_length]))
    return tuple(sequences)


def train(model, train_loader, optimizer, time_signatures, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets, indexs in train_loader:
            inputs, targets = inputs.long(), targets.long()

            src_mask = create_src_mask(inputs)

            optimizer.zero_grad()

            outputs = model(inputs.transpose(0, 1),
                            src_mask, time_signatures, indexs)

            loss = criterion(
                outputs.view(-1, ntokens), targets.view(-1))
            # 패딩 마스크 생성
            mask = (targets != 0).float().view(-1)

            # 패딩된 부분 무시하고 손실 계산
            loss = loss * mask
            loss = loss.sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss/len(train_loader))
        plot_recording(losses)


def valid(model, valid_loader, time_signatures, original_interval, score_length, measure_length, player, clef, melodyrecord=False):
    model.eval()
    Inputs = []
    recordbatch = 3
    with torch.no_grad():
        total_loss = 0
        batch = 0
        for inputs, targets, indexs in valid_loader:
            inputs, targets = inputs.long(), targets.long()
            Inputs.extend(inputs)

            src_mask = create_src_mask(inputs)

            outputs = model(inputs.transpose(0, 1), src_mask, time_signatures,
                            range(len(time_signatures)))
            loss = criterion(
                outputs.view(-1, ntokens), targets.view(-1))
            # 패딩 마스크 생성
            mask = (targets != 0).float().view(-1)

            # 패딩된 부분 무시하고 손실 계산
            loss = loss * mask
            loss = loss.sum() / mask.sum()
            total_loss += loss.item()

            # 악보를 다시 마디별로 쪼개고, 전처리 과정에서 생긴 패딩 없애기
            outputs = devide_scores_to_sequences(
                torch.argmax(outputs, dim=-1), score_length, measure_length, indexs)

            # print([token.tonote(seq) for seq in targets], '\n', [token.tonote(seq)
            #       for seq in outputs])
            if batch == recordbatch:
                if melodyrecord == True:
                    player.record([token.tonote(seq)
                                   for seq in Inputs[:len(outputs)]], time_signatures, original_interval, measure_number=batch*valid_loader.batch_size)
                    melodyrecord = False
                player.record([token.tonote(seq)
                               for seq in outputs], time_signatures, original_interval, measure_number=batch*valid_loader.batch_size, clef=clef)
            batch += 1
    print(f'Average Loss: {total_loss/len(valid_loader):.4f}' +
          '\n'+f'Average Accompany Loss: {total_loss/len(valid_loader):.4f}')


# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_recording(losses):
    plt.figure(1)
    plt.clf()
    plt.xlabel('Epoch')
    plt.title('Training Loss')
    plt.plot(losses, label='Train Loss')
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)


if __name__ == '__main__':

    file_path = './xml'
    player = PianoPlayer()

    if os.path.exists('sequences.pkl') and os.path.exists('len.pkl') and os.path.exists('time.pkl') and os.path.exists('interval.pkl'):
        with open('sequences.pkl', 'rb') as f:
            sequences = pickle.load(f)
        with open('len.pkl', 'rb') as f:
            score_length = pickle.load(f)
        with open('time.pkl', 'rb') as f:
            time_signatures = pickle.load(f)
        with open('interval.pkl', 'rb') as f:
            original_interval = pickle.load(f)
    else:
        sequences, score_length, time_signatures, original_interval = read.read_files(
            read.get_mxl(file_path))
        # print('Melody Sequence:', melody)
        # print('Treble Clef Sequence:', treble)
        # print('Bass Clef Sequence:', bass)
        # 파일에 저장
        with open('token.pkl', 'wb') as f:
            pickle.dump(token, f)
        with open('sequences.pkl', 'wb') as f:
            pickle.dump(sequences, f)
        with open('len.pkl', 'wb') as f:
            pickle.dump(score_length, f)
        with open('interval.pkl', 'wb') as f:
            pickle.dump(original_interval, f)
        with open('time.pkl', 'wb') as f:
            pickle.dump(time_signatures, f)

    empty_measures = [measure+score_length[j-1][0] if j > 0 else measure for j in range(
        len(score_length)) for measure in score_length[j][1]]
    sequences, measure_length = preprocess(sequences)
    melody, treble, bass = sequences
    treble_scores, bass_scores = combine_sequences_to_scores(
        treble, bass, score_length)
    print(melody.shape, treble.shape, bass.shape)
    print(treble_scores.shape, bass_scores.shape)
    treble[empty_measures], bass[empty_measures] = 0, 0
    treble_train_data = IndexDataset(melody.to(device), treble.to(device))
    treble_train_loader = DataLoader(
        treble_train_data, batch_size=BATCH_SIZE, shuffle=True)
    treble_valid_loader = DataLoader(
        treble_train_data, batch_size=BATCH_SIZE)
    bass_train_data = IndexDataset(melody.to(device), bass.to(device))
    bass_train_loader = DataLoader(
        bass_train_data, batch_size=BATCH_SIZE, shuffle=True)
    bass_valid_loader = DataLoader(
        bass_train_data, batch_size=BATCH_SIZE)

    # 모델 학습
    num_epochs = 1
    train(treble_model, treble_train_loader,
          treble_optimizer, time_signatures, num_epochs)
    valid(treble_model, treble_valid_loader, time_signatures,  original_interval, score_length, measure_length,
          player, 'treble')
    train(bass_model, bass_train_loader,
          bass_optimizer, time_signatures, num_epochs)
    valid(bass_model, bass_valid_loader, time_signatures, original_interval, score_length, measure_length,
          player, 'bass', melodyrecord=True)

    # 모델의 state_dict를 저장
    torch.save(treble_model.state_dict(), './treble_model.pth')
    torch.save(bass_model.state_dict(), './bass_model.pth')
    print('Model saved')

    # 테스트용 미디파일 저장
    player.save()
