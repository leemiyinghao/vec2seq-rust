# vec2seq

vec2seq is a Chatbot Repling Core project, act as Question-Answering of presets, split from [WakeUpScrew](https://github.com/leemiyinghao/wakeupscrew-service).

## Mechanism

Reply of vec2seq is based on _Question_ and _Dataset_. Dataset can be added from various sources, mostly articles on PTT bulletboard in current WakeUpScrew Line-bot.

### Embedding of Sentence

We use [finalfusion](https://github.com/finalfusion/finalfusion-rust), which based on FastText and Word2Vec, to provide word embeddings. Train data for word embedding can be but not limit to Wikipedia pages.

In order to address word in chinese, we use a [rust implementation of jieba](https://github.com/messense/jieba-rs) to provide chinese word segementation.

After word embeddings are extracted, all word embeddings in same sentence will be combined into one single sentence embedding, with the help of TF-IDF algorithm.

### Vector Space Search

To achieve real-time-search over more than 2M of articles, we use [granne*](https://github.com/granne/granne), a Rust library for approximate nearest neighbor search based on [Hierarchical Navigable Small World (HNSW) graphs](https://arxiv.org/abs/1603.09320).

### Generate Replies

While question from user can be matched to question in database on semantics, all replies can be seem as a proper reply for question. vec2seq will randomly choice one as final answer.
