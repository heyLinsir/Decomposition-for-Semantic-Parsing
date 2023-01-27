# Decomposition-for-Semantic-Parsing
The code of our TACL paper [*Bridging the Gap between Synthetic and Natural Questions via Sentence Decomposition for Semantic Parsing*]().



## Dependencies

- Python 3.7.1
- nltk==3.4
- numpy==1.18.5
- torch==1.12.0
- transformers==4.15.0
- openai==0.18.1



## Resources

- [Processed Data](https://cloud.tsinghua.edu.cn/f/e2da422c41ed49d7b915/?dl=1)
  - KQA
  - ComplexWebQuestions
- [OpenAI api_key](https://beta.openai.com/)



## Question Decomposition

You can directly use our [decomposed results](https://cloud.tsinghua.edu.cn/f/e2da422c41ed49d7b915/?dl=1), or run the following scripts to decompose questions.

Complete prompts for ComplexWebQuestions and KQA are shown in ./prompt/[ComplexWebQuestions/KQA].

```bash
# For ComplexWebQuestions, use Codex to decompose 5,000 questions from training data, which are used to train a T5-decomposer.
# input_file: ./data/ComplexWebQuestions/ComplexWebQuestions_train.json.pkl.T5-paraphrase
# output_file: ./data/ComplexWebQuestions/ComplexWebQuestions_train.json.pkl.T5-paraphrase.codex-decompose
python ./src/decomposing/ComplexWebQuestions/codex-decompose-for-trainingData.py
```

```bash
# For ComplexWebQuestions, use the 5,000 decomposed questions to train a T5-decomposer.
# And then, decompose all the training questions with the T5-decomposer.
# input_file: ./data/ComplexWebQuestions/ComplexWebQuestions_train.json.pkl.T5-paraphrase.codex-decompose
# output_file: ./data/ComplexWebQuestions/ComplexWebQuestions_train.json.pkl.T5-paraphrase.T5-decompose
bash ./scripts/decomposing/run_T5_Question2Decom.sh
```

```bash
# For ComplexWebQuestions, use Codex to decompose the questions from development/testing sets, which are used as the input of semantic parser.
# input_file: ./data/ComplexWebQuestions/ComplexWebQuestions_[dev/test].json.pkl
# output_file: ./data/ComplexWebQuestions/ComplexWebQuestions_[dev/test].json.pkl.codex-decompose
python ./src/decomposing/ComplexWebQuestions/codex-decompose-for-validationData.py
```

The processing pipeline of KQA is similar to that of ComplexWebQuestions.


## Semantic Parsing

```bash
# For ComplexWebQuestions, use Codex to decompose the questions from development/testing sets, which are used as the input of semantic parser.
# training set: synthetic questions from ./data/ComplexWebQuestions/ComplexWebQuestions_train.json.pkl.T5-paraphrase.T5-decompose
# development set: synthetic questions from ./data/ComplexWebQuestions/ComplexWebQuestions_dev.json.pkl.codex-decompose
# testing set: natural questions from ./data/ComplexWebQuestions/ComplexWebQuestions_test.json.pkl.codex-decompose
bash ./scripts/semantic-parsing/run_T5_Decom2Logic.sh
```
