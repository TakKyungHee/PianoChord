from learn import *

# 모델
treble_accompany_model = TransformerModel(ntokens, emsize, nhead, d_hid,
                                          nlayers, dropout).to(device)
bass_accompany_model = TransformerModel(ntokens, emsize, nhead, d_hid,
                                        nlayers, dropout).to(device)
# 모델의 state_dict 불러오기
if os.path.exists('./treble_ac_model.pth'):
    treble_accompany_model.load_state_dict(torch.load(
        './treble_ac_model.pth', map_location=device, weights_only=True))
    print('Treble accompany Model loaded')
if os.path.exists('./bass_ac_model.pth'):
    bass_accompany_model.load_state_dict(torch.load(
        './bass_ac_model.pth', map_location=device, weights_only=True))
    print('Bass accompany Model loaded')


# 손실 함수
criterion = nn.CrossEntropyLoss()  # 필요에 따라 손실 함수 변경
treble_optimizer = optim.Adam(treble_model.parameters(), lr=lr)
treble_accompany_optimizer = optim.Adam(
    treble_accompany_model.parameters(), lr=lr)
bass_optimizer = optim.Adam(bass_model.parameters(), lr=lr)
bass_accompany_optimizer = optim.Adam(bass_accompany_model.parameters(), lr=lr)

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
    original_treble, original_bass = treble.clone(), bass.clone()
    treble[empty_measures], bass[empty_measures] = 0, 0
    treble_accompany_train_data = IndexDataset(
        treble.to(device), original_treble.to(device))
    treble_accompany_train_loader = DataLoader(
        treble_accompany_train_data, batch_size=BATCH_SIZE, shuffle=True)
    treble_accompany_valid_loader = DataLoader(
        treble_accompany_train_data, batch_size=BATCH_SIZE)
    bass_accompany_train_data = IndexDataset(
        bass.to(device), original_bass.to(device))
    bass_accompany_train_loader = DataLoader(
        bass_accompany_train_data, batch_size=BATCH_SIZE, shuffle=True)
    bass_accompany_valid_loader = DataLoader(
        bass_accompany_train_data, batch_size=BATCH_SIZE)

    # 모델 학습
    num_epochs = 10
    train(treble_accompany_model, treble_accompany_train_loader,
          treble_accompany_optimizer, time_signatures, num_epochs)
    valid(treble_accompany_model, treble_accompany_valid_loader,
          time_signatures, original_interval, score_length, measure_length, player, 'treble')
    train(bass_accompany_model, bass_accompany_train_loader,
          bass_accompany_optimizer, time_signatures, num_epochs)
    valid(bass_accompany_model, bass_accompany_valid_loader,
          time_signatures, original_interval, score_length, measure_length, player, 'bass', melodyrecord=True)

    # 모델의 state_dict를 저장
    torch.save(treble_accompany_model.state_dict(), './treble_ac_model.pth')
    torch.save(bass_accompany_model.state_dict(), './bass_ac_model.pth')
    print('Model saved')

    # 테스트용 미디파일 저장
    player.save()
