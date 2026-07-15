## Run 2: character-level tokenizer (replacing byte-level)
Hypothesis: byte-level tokenizer forces 3 tokens per Devanagari character, wasting context and steps on the Hindi half of the corpus; a char-level vocab trained on the corpus should let the model see more real content per step.
Changed: tokenizer.py -> CharTokenizer, vocab built from all 913 unique chars in train_corpus.txt, byte fallback for anything unseen (keeps lossless round-trip). Same LR schedule as Run 1.
Dev bpb: 2.3593 -> 2.298
Tokens in training corpus: 7,318,592 -> 5,703,936 (~22% fewer tokens for the same text)
Conclusion: tokenizer was in fact the bigger lever, as expected. Params went 1,339,840 -> 1,550,080 (vocab grew), still well under the 2M cap, so there's ~450k of param budget left to spend.
## Run 3: wider model + weight tying
Hypothesis: char tokenizer freed us from being context-bottlenecked; with tokens now more information-dense, more model capacity should help, and tying weights buys headroom to afford it under the 2M param cap.
Changed: n_embd 160 -> 176, tie_weights True (was False). Same LR schedule/clip as Run 1, same tokenizer as Run 2.
Dev bpb: 2.298 -> 2.2763
Params: 1,550,080 -> 1,840,256 (still under 2,000,000 cap)
Conclusion: extra width + tying helped further. ~160k param headroom left; diminishing returns expected from width alone at this point.
## Run 4: higher peak LR
Hypothesis: loss curve in Run 3 was still dropping steeply with no instability at lr=6e-4, suggesting we were under-shooting the optimal LR rather than at risk of overshooting it.
Changed: peak lr 6e-4 -> 9e-4. Everything else identical to Run 3 (n_embd=176, tied weights, char tokenizer, AdamW+warmup/cosine+clip).
Dev bpb: 2.2763 -> 2.1901
Conclusion: biggest single-run improvement so far. Loss curve stayed smooth (no spikes), confirming 9e-4 wasn't past the stability boundary. This is the best config found; used for the final run.
