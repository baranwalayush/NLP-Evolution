## Eliza 

Eliza is a computer program developed by Joseph Weizenbaum in the 1960s at MIT. It was one of the first programs capable of attempting the Turing Test, simulating conversation with a human user. It is a simple **rule-based chatbot** that uses **pattern matching and substitution** to give the illusion of understanding. It was originally designed to mimic a psychotherapist, by responding to user inputs with questions and statements that encourage further conversation and gives the user the impression of being listened to and understood. 

### How it works

Eliza operates by recognizing keywords in the user's input and applying pre-defined rules to generate responses. It uses a set of scripts that contain patterns and corresponding responses. When a user inputs a statement, Eliza searches for keywords and matches them against its patterns. If a match is found, it selects an appropriate response from its script, often rephrasing the user's input or asking follow-up questions. If no keywords are found, Eliza defaults to generic responses to keep the conversation going such as "Tell me more about that" or "Can you elaborate on that?". 

### How to run Eliza

To run the ELIZA model, navigate to the `models/01_eliza` directory and run the following command:

```bash
python eliza.py
```

### Example conversation

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

### Additional notes

1.  **Eliza effect**

    In plain English, the ELIZA effect happens when a machine sounds socially convincing enough that people begin to treat it as though there is a real mind behind the words. The machine does not need to be sentient, emotional, or even especially advanced. It only needs to produce responses that feel coherent, attentive, or personal.

2. You can check [this really interesting page](https://elizaemulator.com/influence) to know more about the influence, ethics, and impact Eliza had on the field of AI and human-computer interaction.