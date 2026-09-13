# NLP Through the Ages

The plan is to build a project tracing the evolutions in the field of Natural Language Processing over the years with small, real implementations of each era, to show how the field has evolved from simple hand-written rules to models capable of understanding and generating human-like text. I'm planning to implement each era's model from scratch, train them on the same corpus (Tiny Shakespeare), where possible, with the goal of learning and understanding the underlying principles of each approach.


| # | Era | Year(s) | What it is | Status
|---|-----|---------|------------|---------
| 01 | Rule-based | 1966 | ELIZA -- regex pattern matching | &#x2713;
| 02 | Statistical ML | 1980s-90s | Character n-gram model | &#x2713;
| 03 | Feedforward NN | 2003 | Bengio-style neural language model |
| 04 | Recurrent NN | 1990s/2010s | Vanilla RNN language model |
| 05 | Gated RNN | 1997/2014 | LSTM language model |
| 06 | Attention | 2014-15 | Seq2seq + Bahdanau attention |
| 07 | Transformer | 2017 | Decoder-only Transformer (MiniGPT) |

## Repo structure

```
NLP-Evolution/
  common/        shared char tokenizer, batching, metrics, etc.
  data/          Tiny Shakespeare corpus (~1.1MB, public domain)
  models/        one folder per era: model.py, train.py, infer.py
  docs/          reading material for each era
```

## Models

### 1. Rule-based: ELIZA

Please check [this page](docs/01_eliza/01_eliza.md) for more details on the ELIZA model, its implementation, and how it works.

To run the ELIZA model, navigate to the `models/01_eliza` directory and run the following command:

```bash
python eliza.py
```

**Example conversation:**

```
ELIZA: Hello. How are you feeling today? (Type 'quit' to exit)
YOU: I feel sad
ELIZA: Tell me more about feeling sad.
YOU: I can't tell
ELIZA: What is stopping you from telling?
YOU: because I need help   
ELIZA: Is that the real reason?
YOU: yes, my mother would have helped me
ELIZA: Tell me more about your family.
YOU: I really love them
ELIZA: Let's explore that further.
YOU: Why don't you tell me something
ELIZA: Perhaps I will tell you something eventually.
YOU: ok
ELIZA: Can you elaborate on that?
YOU: quit
ELIZA: Goodbye! Take care.
```

### 2. Statistical ML: N-gram model

Please check [this page](docs/02_ngram/02_ngram.md) for more details on the N-gram model, its implementation, and how it works.

To run the N-gram model, navigate to the `models/02_ngram` directory and run the following command:

```bash
python train.py
``` 
and to generate text using the trained model, run the following command:
```bash
python infer.py
```

**Example output** of the model with `seed` as "ROMEO:" and `length` as 400:

```
ROMEO:
Will ming impering the of man hom outh.

Nurset
To thou or thind weedio as I shallow.

MENENIUS:
Looking to your miss that armfu;;?QoKBx-nx,cve,
Let masters: thined, comfor with that suppetM

SLY:
Whate, you, so I acting wake talk of your not in head;
Was rospected mous tell, your you fleepinal at onescestice you.

MOND:
Madam, and ened, fords, they
The do to
To riceasary's till'd beam, wher;
And
```

The **perplexity score** of the default 4-gram model is **5.55**.