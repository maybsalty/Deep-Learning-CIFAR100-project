# Task 3: Headline Generation — Internal Guide

## Task and dataset

- [ ] Describe headline generation and explain why it is a text-generation task.
- [ ] Introduce Gigaword and cite its source:
  https://www.tensorflow.org/datasets/catalog/gigaword
- [ ] Record the actual train, validation and test counts and how the splits
  were selected. Training used 100,000 examples.

## Preprocessing

- [ ] Explain the cleaning steps actually used.
- [ ] Describe the separate article and headline WordPiece vocabularies and
  confirm which data was used to build them.
- [ ] Record vocabulary sizes, sequence lengths, padding and truncation.
- [ ] Explain `[CLS]`, `[SEP]` and the target shift used for teacher forcing.
- [ ] Document the cached files:
  - `tokenized_data.npz`
  - `wordpiece_vocab_x.npy`
  - `wordpiece_vocab_y.npy`

## Model and training

- [ ] Describe the embedding, Conv1D and bidirectional LSTM encoder.
- [ ] Describe the LSTM decoder, dot-product attention, dropout and output layer.
- [ ] Explain why attention and padding masks are used.
- [ ] Record the loss, optimizer, batch size, early stopping and checkpoint settings.
- [ ] Include the training output and learning curves.
- [ ] Discuss why validation loss bottomed out around epoch 25 while training
  loss continued to fall.

## Evaluation

- [ ] Explain the difference between teacher-forced accuracy and independently
  generated headlines.
- [ ] Describe ROUGE-L F1 and rescaled BERTScore F1 and their settings.
- [ ] Report the results:
  - Validation: ROUGE-L 0.231; rescaled BERTScore 0.167.
  - Test: ROUGE-L 0.183; rescaled BERTScore 0.023.
- [ ] Record the number of examples and source of references for each run.
  Validation references were decoded from padded targets and may be truncated.
- [ ] Check package versions and reference processing before comparing the runs.
- [ ] Show several articles, reference headlines and generated headlines.

## Limitations and conclusion

- [ ] Discuss empty, incoherent or factually incomplete outputs.
- [ ] Explain the dataset’s literal `UNK` placeholders and their effect.
- [ ] Explain that ROUGE and BERTScore do not guarantee factual correctness.
- [ ] Summarize findings and discuss possible improvements without claiming
  they have already been tested.