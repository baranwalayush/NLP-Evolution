## Artificial Neural Networks (ANNs) for Natural Language Processing (NLP)

Artificial Neural Networks (ANNs) are a class of machine learning models inspired by the structure and function of the human brain. They consist of interconnected nodes, or "perceptrons", organized into layers that process input data and learn to make predictions or classifications. 

ANNs have been widely used in various NLP tasks, such as text classification, sentiment analysis, machine translation, and language modeling. They were the first type of neural networks to be applied to NLP tasks and have paved the way for more advanced architectures like Recurrent Neural Networks (RNNs) and Transformers.

Bengio et al. (2003) demonstrated the effectiveness of ANNs in NLP by introducing a neural probabilistic language model that outperformed traditional n-gram models. This work laid the foundation for subsequent research in neural network-based NLP. Here are some key aspects of ANNs in the context of NLP:
1. **Architecture**: ANNs typically consist of an input layer, one or more hidden layers, and an output layer. Each layer contains multiple neurons that apply activation functions to the weighted sum of their inputs, allowing the network to learn complex patterns in the data.
2. **Training**: ANNs are trained using backpropagation, a supervised learning algorithm that adjusts the weights of the connections between neurons based on the error of the network's predictions. This process allows the network to learn from labeled data and improve its performance over time.
3. **Applications**: ANNs have been applied to a wide range of NLP tasks, including:
    - Text classification, Sentiment analysis, and Named Entity Recognition (NER).

### How it works in our case
Here, we have implemented a simple feedforward neural network consisting of an embedding layer, a hidden layer, and an output layer. The embedding layer converts input words into dense vector representations, capturing semantic information about the words. The hidden layer processes these embeddings and learns to identify patterns in the data, while the output layer produces predictions based on the learned patterns. The most notable idea here is the word embedding layer, which allows the model to learn distributed representations of words, capturing their meanings and relationships in a continuous vector space. This enables the model to generalize better to unseen words and phrases, improving its performance on NLP tasks unlike traditional n-gram models that rely on discrete representations of words. 

Now, two words with similar meanings, but no common characters, will have similar embeddings, allowing the model to recognize their semantic similarity. For example, the words "king" and "queen" may have similar embeddings, while the words "king" and "car" would have very different embeddings. This allows the model to capture the relationships between words and improve its performance on NLP tasks.

The model is provided input in a similar way to the n-gram model, but instead of counting the occurrences of n-grams, it maps each word to its corresponding embedding vector and processes the sequence of embeddings through the hidden layer to produce predictions. Unlike n-gram models, which rely on a fixed-size context window, ANNs can process variable-length input sequences and capture long-range dependencies in the data. This allows the model to learn more complex patterns and relationships between words, improving its performance on NLP tasks.

However, a fixed-context window of size 8 is baked into this architecture on purpose to limit the amount of information the model needs to process at each step. This is a limitation of this approach, which can be addressed by using the next generation neural nets like RNNs.

### Example
For example, given the input sequence "I love natural language processing", the model would first convert each word into its corresponding embedding vector, resulting in a sequence of vectors. These vectors would then be processed through the hidden layer, which would learn to identify patterns in the data and produce a prediction for the next word in the sequence. The model would then output a probability distribution over the vocabulary, indicating the likelihood of each word being the next word in the sequence. The model would then select the most likely word based on the predicted probabilities and use it as the next input in the sequence, repeating this process until the desired length of the output sequence is reached. This allows the model to generate coherent and contextually relevant text based on the input sequence, demonstrating its ability to learn complex patterns and relationships between words in the data.

### How to run the ANN model
To run the ANN model, navigate to the `models/03_ann` directory and run the following command:
```bash
python train.py
```
The model will be trained on the [`tinyshakespeare.txt`](../../data/tinyshakespeare.txt) dataset and saved as `ann_lm.pt`. You can also change the dataset by modifying the `DATA_PATH` variable in `train.py`.

To generate text using the trained model, run the following command:
```bash
python infer.py
```
Default `seed` is "ROMEO:" and `length` is 400. You can change these values by passing arguments to the script. For example, `python infer.py "JULIET:" 500` will generate 500 characters of text starting with "JULIET:". The `seed` is the initial string to start the generation and `length` is the desired length of the generated text.

### Results
The validation perplexity score of the model is **6.73**. Your output may vary slightly due to the stochastic nature of the training process. The training loss and val perplexity decreases over time, indicating that the model is learning to make better predictions as it is exposed to more data.

Example output of the model:
```
ROMEO:
Onot farm, ochergin faslld,
Of have in miphan
so, that bagrie surspiod feel pliningly, woth had ceabinin? whise wind bery the ppain, stars?
Oll kneet, my hamts o spot yuis de ware.

MENRUCIO:
Weruun it peirigh.

Narsing cours sidvor sice ambrase forited; it gaire a pined
his, lor!

Satp hands idn h
```