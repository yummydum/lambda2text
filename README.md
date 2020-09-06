# lambda2text

Translate lambda expressions to natural language.

# Overview

Example logical formula (input to the model)

```
exists x01.(_amrozi(x01) & exists x02.(_brother(x02) & exists x03.((x03 = _he) & exists x04.(_witness(x04) & _"(x04) & exists e05.(_call(e05) & (Subj(e05) = x03) & (Acc(e05) = x04) & (Dat(e05) = x02)))) & exists e06.(_accuse(e06) & (Subj(e06) = x01) & (Acc(e06) = x02) & exists x07 x08.(_evidence(x08) & exists e09.(_distort(e09) & (Subj(e09) = x07) & (Acc(e09) = x08) & _deliberately(e09) & (Subj(e09) = x07) & _of(e06,x07))))))
```

The target of the model is to translate this logical formula to the natural language with the equivalent semantics;

```
Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .
```

# TODO:

- Apply ccg2lambda to SICK and extract unproven subgoals (9/10)
- Implement baseline method (9/10)
- Complete Error analysis and summarize the result in a notebook (9/17)
- Complete the first attempt to solve the error (10/1)


# Preprocess

1. download_mnli_split.py (get mnli data from remote & split)
2. run_ccg2lambda (for each split data apply ccg2lambda parallely)
3. create_formal_cleansed.py (filter & aggregate split files)
4. dataset.py (build vocab & return bucket Iterator)
5. train_tokenizers.py

# Model

- Transformer based seq2seq (baseline)

# Train

```
cd src
make sweep
make run gpu=0,1 id=<sweep_id>
```

# Reference
- https://github.com/bentrevett/pytorch-seq2seq

