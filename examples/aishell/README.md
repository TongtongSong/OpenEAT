# Aishell

## common config
- Vocab info: 3243 Characters and Extra nuits (blank, unk, sos/eos)
- Feature info: 80 fbank, no cmvn, no speech perturb, SpecAugment 3 (50) + 2 (10).
- Model info: base 256+1024, 4 heads, kernel size 15 (conformer).
- Architecture info: AED 12+3+3.
- Training info: lr 0.001, dropout 0.1, ctc_weight 0.3, reverse_weight 0.3, 1 GPU, max_frames_in_batch 10000, 50 epochs, warmup 25K.
- Decoding info: average last 5, ctc weight 0.5, reverse_weight 0.3.

## Reslut

|Model|ctc greedy search|ctc prefix beam search|attention|attention rescoring|
|:---:|:---:|:---:|:---:|:---:|
|Transformer|7.32|7.32|7.23|6.63|
|Conformer|6.73|6.73|6.88|6.22|











