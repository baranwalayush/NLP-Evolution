"""
ELIZA (Weizenbaum, 1966) -- The DOCTOR script.

No learning, No memory, No context. A simple rule-based chatbot doing regex pattern-matching against a
scripted list of rules with pronoun reflection to simulate conversation with a human user. 

It's the zero point of this whole timeline, everything else in this repo will be an attempt to replace this hand-written rule list with actual learning from data.
"""

import random
import re

RULES = [
    (r"i feel (.*)", ["Tell me more about feeling {0}.", 
                      "Do you often feel {0}?"]),
    (r"i need (.*)", ["Why do you need {0}?", 
                      "Would it really help you to get {0}?"]),
    (r"i am (.*)|i'm (.*)", ["How long have you been {0}?", 
                             "Why do you say you are {0}?"]),
    (r"i can't (.*)|i cannot (.*)", ["What is stopping you from {0}ing?", 
                                     "Have you tried to {0}?", 
                                     "Do you think you should be able to {0}?"]),
    (r"why don't you (.*)", ["Do you really think I don't {0}?", 
                             "Perhaps I will {0} eventually."]),
    (r"are you (.*)", ["Why does it matter whether I am {0}?", 
                       "Would you prefer if I weren't {0}?"]),
    (r"because (.*)", ["Is that the real reason?", 
                       "What other reason might there be?"]),
    (r"(.*)(?:mother|father|family)(.*)", ["Tell me more about your family.", 
                                        "How do you feel about your family?"]),
    (r"(.*)\bsorry\b(.*)", ["Please don't apologize.", 
                            "No need to apologize."]),
    (r"hello|hi|hey", ["Hello. How are you feeling today?"]),
    (r"(.*)\?$", ["Why do you ask that?", 
                  "What do you think?"]),
    (r"(.*)", [  # fallback, always matches
        "Please tell me more.", 
        "Can you elaborate on that?", 
        "Let's explore that further.",
    ]),
]


REFLECTIONS = {
    "am": "are", "was": "were", "i": "you", "i'd": "you would",
    "i've": "you have", "i'll": "you will", "my": "your", 
    "are": "am", "you've": "I have", "you'll": "I will", 
    "your": "my", "yours": "mine", "you": "I", "me": "you",
}


def _reflect(fragment: str) -> str:
    """
    Reflects pronouns in the input fragment to simulate conversation.
    """
    words = fragment.lower().split()
    return ' '.join(REFLECTIONS.get(w, w) for w in words)


def respond(user_input: str) -> str:
    """
    Generates a response based on the user's input by matching it against the predefined rules.
    """
    text = user_input.strip().lower()
    for pattern, responses in RULES:
        match = re.match(pattern, text)
        if match:
            template = random.choice(responses)
            groups = [_reflect(g) for g in match.groups() if g]
            return template.format(*groups)
    return "Please tell me more."


def chat():
    print("ELIZA: Hello. How are you feeling today? (Type 'quit' to exit)")
    while True:
        user_input = input("YOU: ")
        if user_input.strip().lower() in ("quit"):
            print("ELIZA: Goodbye! Take care.")
            break
        print("ELIZA:", respond(user_input))


if __name__ == "__main__":
    chat()