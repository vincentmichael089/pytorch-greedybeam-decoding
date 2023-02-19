# Greedy-Beam Decoding for PyTorch's Encoder-Decoder model
Scratch implementation of greedy and beam search decoding for PyTorch's encoder-decoder model.

### Information about the code:

1.  Every decoding can be called by calling the `translate()` function.
2. `input_sequence` is a tensor with a size of `BATCH x SEQ_LEN`.
3. `model.encode()` and `model.decode()`, both are the model encoder and decoder respectively, should yield a tensor with a size of `BATCH x SEQ_LEN x HIDDEN_SIZE`.
4. `model.linear()` is a linear layer as follows: `nn.Linear(emb_size, trg_vocab_size)`, basicaly transforming the decoder output to fit the target vocabulary.
