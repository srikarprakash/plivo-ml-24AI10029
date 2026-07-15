## Run 2: character-level tokenizer (replacing byte-level)
Hypothesis: byte-level tokenizer forces 3 tokens per Devanagari character, wasting context and steps on the Hindi half of the corpus; a char-level vocab trained on the corpus should let the model see more real content per step.
Changed: tokenizer.py -> CharTokenizer, vocab built from all 913 unique chars in train_corpus.txt, byte fallback for anything unseen (keeps lossless round-trip). Same LR schedule as Run 1.
Dev bpb: 2.3593 -> 2.298
Tokens in training corpus: 7,318,592 -> 5,703,936 (~22% fewer tokens for the same text)
Conclusion: tokenizer was in fact the bigger lever, as expected. Params went 1,339,840 -> 1,550,080 (vocab grew), still well under the 2M cap, so there's ~450k of param budget left to spend.
