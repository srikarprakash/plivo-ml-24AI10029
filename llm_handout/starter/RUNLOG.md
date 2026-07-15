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
## Run 5: BPE tokenizer (replacing char-level) + fixed weight tying
Hypothesis: char-level tokenizer was better than byte-level but still spends one token per character; BPE merges frequent subword/conjunct sequences into single tokens, letting each 128-token window cover more real text. Also discovered tie_weights wasn't actually engaging in Runs 3-4 (observed params matched the untied formula) — fixing it is required to afford BPE's larger vocab under the 2M cap.
Changed: tokenizer.py -> byte-level BPE, vocab 2048, merges learned on a 1MB sample of train_corpus.txt, byte fallback preserved (verified lossless on corpus + unseen emoji/CJK/control chars). Fixed tie_weights to actually apply (confirmed via params: 2,239,776 untied -> 1,879,328 tied). Same n_embd=176, lr=9e-4, AdamW+warmup/cosine+clip as Run 4.
Dev bpb: 2.1901 -> 2.077
Tokens in training corpus: 5,703,936 (char) -> 2,449,817 (BPE) - each token now covers ~3x more bytes than char-level.
Note: raw train loss appears higher (4.17 vs 1.99) but is NOT comparable across tokenizers - larger vocab (2048 vs 913) means harder per-token softmax classification even as bpb (the byte-normalized, tokenizer-agnostic metric) improved.
Conclusion: best result of the hour. BPE + correctly-applied weight tying together were the two biggest wins.
## Run 6: batch size 8 -> 16 (data coverage)
Hypothesis: steps x batch x block_size = 2000x8x128 = 2,048,000 tokens processed total, which is LESS than the 2,449,817-token BPE corpus - the model wasn't even completing one pass over its own training data. Increasing batch (allowed to change, doesn't touch the 2000-step cap) should let it see meaningfully more real content.
Changed: --batch 8 -> 16. Everything else identical to Run 5 (BPE vocab 2048, tied weights, n_embd=176, lr=9e-4, AdamW+warmup/cosine+clip).
Dev bpb: 2.077 -> 1.9502
Conclusion: biggest single win of the hour. Confirms the model was data-starved, not capacity- or optimizer-starved. Cost: wall-clock only (~180ms/step vs ~103ms/step), no params, no step-cap risk.
## Run 7: depth-scaled residual init
Hypothesis: starter model.py used one flat std=0.05 for every weight (flagged as questionable in initial read). Weights that write INTO the residual stream (attn.proj, mlp's 2nd linear) compound variance across n_layer additions if not scaled down - standard GPT-2 fix is std/sqrt(2*n_layer) for exactly those weights.
Changed: added depth-scaled init for attn.proj.weight and mlp.2.weight only (std=0.05/sqrt(2*4)=0.0177). Everything else identical to Run 6 (batch=16, BPE vocab 2048, tied weights, n_embd=176, lr=9e-4).
Dev bpb: 1.9502 -> 1.9076
Conclusion: clean isolated win - only variable changed was init. Train loss at step 2000 also lower than Run 6 (3.66 vs 3.74) at identical config otherwise, consistent with better-conditioned optimization from the start, not just noise.
## Run 8: batch size 16 -> 32
Hypothesis: Run 6 showed data coverage was the single biggest lever; scaling further to ~3.3 full passes over the corpus (2000x32x128=8,192,000 tokens vs a 2,449,817-token corpus) should continue to help.
Changed: --batch 16 -> 32. Everything else identical to Run 7 (BPE vocab 2048, tied weights, n_embd=176, lr=9e-4, depth-scaled init, AdamW+warmup/cosine+clip).
Dev bpb: 1.9076 -> 1.7898
Conclusion: largest single-run win of the session. Confirms data coverage remained the dominant lever even after 3+ effective passes. Cost was ~2x wall-clock (767s vs 367s), no params, no step-cap risk. Stopping experimentation here to leave time for deliverables — trend suggests batch=64 might help further, untested due to time budget.
