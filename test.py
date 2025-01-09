from ac_learn import *
from toxml import convert_pdf_to_images, image_upscaling, convert_image_to_musicxml, cleaning


def test(model, accompany_model, test_loader, clef, melodyrecord=False):
    model.eval()
    Inputs = []
    Outputs = [None]*len(test_loader.dataset)
    with torch.no_grad():
        for inputs, indexs in test_loader:
            inputs = inputs.long()
            Inputs.extend(inputs)

            src_mask = create_src_mask(inputs)

            outputs = model(inputs.transpose(0, 1), src_mask, time_signatures,
                            range(len(time_signatures)))
            for i in range(len(indexs)):
                Outputs[indexs[i]] = torch.argmax(outputs[i], dim=-1)
            # print([token.tonote(seq) for seq in inputs], '\n', [token.tonote(seq)
            #       for seq in torch.argmax(outputs, dim=-1)])

        # 생성된 악보를 다시 인풋으로 받는 처리
        scores = combine_sequences_to_scores(torch.stack(Outputs), score_length)[
            0]  # 가변인자 있으면 반드시 [0] 씀. 기억!
        accompany_valid_data = IndexDataset(
            scores.to(device))
        accompany_valid_loader = DataLoader(
            accompany_valid_data, batch_size=BATCH_SIZE//2)

        for inputs, indexs in accompany_valid_loader:
            inputs = inputs.long()

            src_mask = create_src_mask(inputs)

            outputs = accompany_model(inputs.transpose(0, 1),
                                      src_mask, time_signatures, indexs, accompany=True)

            # 악보를 다시 마디별로 쪼개고, 전처리 과정에서 생긴 패딩 없애기
            outputs = devide_scores_to_sequences(
                torch.argmax(outputs, dim=-1), score_length, measure_length, indexs)

            if melodyrecord == True:
                player.record([token.tonote(seq)
                               for seq in Inputs[:len(outputs)]], time_signatures, original_interval)
                melodyrecord = False
            player.record([token.tonote(seq)
                           for seq in outputs], time_signatures, original_interval, clef=clef)


if __name__ == '__main__':

    file_path = './test'
    player = PianoPlayer()

    if len([True for file in os.listdir(file_path) if file.endswith('.png') or file.endswith('.jpg')]) == 0:
        for file in os.listdir(file_path):
            if file.endswith('.pdf'):
                input_name = os.path.join(file_path, file)
                convert_pdf_to_images(file_path, input_name)

    if len([True for file in os.listdir(file_path) if file.endswith('.mxl')]) == 0:
        for file in os.listdir(file_path):
            if file.endswith('.png') or file.endswith('.jpg'):
                input_name = os.path.join(file_path, file)
                image_upscaling(input_name)
                convert_image_to_musicxml(input_name, file_path)
                cleaning(file_path, file_path)

    sequences, score_length, time_signatures, original_interval = read.read_files(
        read.get_mxl(file_path))
    sequences, measure_length = preprocess(sequences)
    melody = sequences[0]
    print(melody.shape)

    print('Original Interval:', original_interval)
    print('Time Signatures:', time_signatures)

    test_data = IndexDataset(melody)
    treble_test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE)
    bass_test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE)
    test(treble_model, treble_accompany_model,
         treble_test_loader, 'treble', melodyrecord=True)
    test(bass_model, bass_accompany_model, bass_test_loader, 'bass')
    player.save()
    player.play()
