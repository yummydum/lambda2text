# Problems of current NLI systems (3 min)

- What is NLI
  - Example instance fron SNLI or MNLI
- Clever Hans problem
  - Right for the wrong reason
  - カバーできてない言語現象をやまほど揚げる
  - Happens due to both complexity of inference phenomena and black box nature of deep learning

# Idea for interpretable NLI system with generation and coq (4 min)

- Introduction to formal semantics
- Brief intro for coq
- ccg2lambda + generation + coq
  - Interpretable
  - Interactive prooving

# Current status (3 min)

## BLEU score for LSTM and transformer

- token

  - lstm 42.3
  - transformer 45.5

- graph
  - lstm 23.2
  - transformer

## Next steps

- why score for graph so low?
- How to improve accuracy for the generation (what's the new point?)
  - tree transformer
  - other stuff
