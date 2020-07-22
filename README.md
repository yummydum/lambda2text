# lambda2text

Translate lambda expressions to natural language.

# Overview

The logical formula in ccg2lambda format is obtained by depccg (see Makefile for command).

Example logical formula (input to the model)
```
exists x01.(_amrozi(x01) & exists x02.(_brother(x02) & exists x03.((x03 = _he) & exists x04.(_witness(x04) & _"(x04) & exists e05.(_call(e05) & (Subj(e05) = x03) & (Acc(e05) = x04) & (Dat(e05) = x02)))) & exists e06.(_accuse(e06) & (Subj(e06) = x01) & (Acc(e06) = x02) & exists x07 x08.(_evidence(x08) & exists e09.(_distort(e09) & (Subj(e09) = x07) & (Acc(e09) = x08) & _deliberately(e09) & (Subj(e09) = x07) & _of(e06,x07))))))
```

The target of the model is to translate this logical formula to the natural language with the equivalent semantics;
```
Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .
```

Currently a simple transformer based seq2seq is implemented (see `src/model/seq2seq.py`).
Training is handeled by `src/trainer/train_seq2seq.py`.

TODO: 
- Execute ccg2lambda at scale
- Implement encoders which takes tree structure in the logical formula in account
