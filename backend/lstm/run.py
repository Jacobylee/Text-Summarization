import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn

from backend.lstm.bilstm import EncoderLSTM, DecoderLSTM, HIDDEN_DIM
from backend.lstm.data_loading import build_index, build_data_vectors, build_vocab, articles_dir, summaries_dir, EOS, \
    BOS, Article_TRUNCATE


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=Article_TRUNCATE, show_dev=False):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        print('@@@', input_tensor[ei].unsqueeze(0))
        encoder_output, (encoder_h, encoder_h_c) = encoder(input_tensor[ei].unsqueeze(0))
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([tgt_stoi[BOS]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == tgt_stoi[EOS]:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def dev(encoder, decoder, x, y):
    encoder_hidden = encoder.init_hidden()

    input_length = x.size(0)
    target_length = y.size(0)

    encoder_outputs = torch.zeros(Article_TRUNCATE, encoder.hidden_size)

    for ei in range(input_length):
        encoder_output, _ = encoder(x[ei].unsqueeze(0))
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([tgt_stoi[BOS]])
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        # print(decoder_output)
        # print(torch.argmax(decoder_output))
        if decoder_input.item() == tgt_stoi[EOS]:
            break


if __name__ == '__main__':
    a_filename_tokens, s_filename_tokens, src_itos, tgt_itos = build_vocab(articles_dir, summaries_dir)
    src_stoi = build_index(src_itos)
    tgt_stoi = build_index(tgt_itos)
    src_vector, tgt_vector = build_data_vectors(articles_dir, summaries_dir, src_stoi, tgt_stoi)
    data = pd.DataFrame({'Articles': a_filename_tokens, 'Summary': s_filename_tokens, 'A_vectors': src_vector,
                         'S_vectors': tgt_vector})
    input_dim = len(src_itos)
    output_dim = len(tgt_itos)
    X_train, X_val, Y_train, Y_val = train_test_split(data['A_vectors'], data['S_vectors'], test_size=0.3,
                                                      random_state=29)
    X_train = torch.tensor(X_train, dtype=torch.long)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    Y_val = torch.tensor(Y_val, dtype=torch.long)

    n_iters = 500
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0   # Reset every plot_every

    encoder = EncoderLSTM(input_dim, HIDDEN_DIM)
    decoder = DecoderLSTM(HIDDEN_DIM, output_dim)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

    criterion = nn.NLLLoss()

    for i in range(len(X_train)):
        input_tensor = X_train[i]
        target_tensor = Y_train[i]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, show_dev=True)  # todo
        print_loss_total += loss
        plot_loss_total += loss
        print(loss)
    # --------------------------------
    # dev(encoder, decoder, X_val[0], Y_val[0])
