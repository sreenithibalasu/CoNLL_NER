# Named Entity Recognition

- **Named Entity Recognition (NER)** aims to categorize unstructured text into pre-defined categories such as person name, organization name, location, phone number, email etc. The idea of doing NER came to me because I got a taste of it during an internship I did in India. 
- NER can be an important task because it can help identify key elements in unstructured data such as text. 
  - *For example*: if we had an image detection and recognition system that extracts text from business cards, categorizing that text automatically into pre-defined categories such as name, organization, email address etc. can help us build structured datasets to store that useful information in.

![image](https://user-images.githubusercontent.com/40226554/164024847-f6fe55ea-1af1-4080-9137-ea5f91631988.png)

- **Usage Notes** can be found [here](https://github.com/sreenithibalasu/CoNLL_NER/blob/main/data/README.md)


## Dataset
- I've chosen the CoNLL 2003 [1] dataset. The task for this "competition" was to do language independent NER. The entire dataset consisted of English and German text. For the purpose of this project, I've chosen to include only the english language text.

| English Text | Tokens | LOC | PER | ORG | MISC |
|--------------|--------|-----|-----|-----|------|
|  Training | 203,621 | 8297 | 11128 | 10025 | 4593 |
|  Validation | 51,362 | 2094 | 3149 | 2092 | 1268 |
|  Testing | 46,435 | 1925 | 2773 | 2496 | 918 |

- This dataset contains tokenized text of **5 categories**:
  - LOC: location
  - PER: person name
  - ORG: organization name
  - MISC: miscellaneous 
  - O: objects

- The dataset was split into train, validation and test sets, with 200k + tokens in the training set and 46k tokens in the test set

<img width="446" alt="Screen Shot 2022-04-18 at 9 15 16 AM" src="https://user-images.githubusercontent.com/40226554/163820892-fb69c3a9-5608-49bb-9878-758e498ab998.png">

<img width="460" alt="Screen Shot 2022-04-18 at 9 15 32 AM" src="https://user-images.githubusercontent.com/40226554/163820948-a53573d0-b145-409a-9b6e-cb3b9f1743a8.png">

<img width="476" alt="Screen Shot 2022-04-18 at 9 15 44 AM" src="https://user-images.githubusercontent.com/40226554/163820983-09bb899e-7e97-4d81-b92e-bb1312d5b0d9.png">

## Preprocessing

### Step 1: Convert Text File to CSV Files
- **Corresponding File**: `txt_to_csv.py`
- The original dataset files were of `.txt` format. These text files started with a `-DOCSTART -X-X-X` and had multiple blank lines in between each text sample. 
- As a preprocessing step, I converted the `txt` files to `csv` files, by extracting lines with text which weren't `-DOCSTART`, had `length > 0` and didn't start with a newline character `\n`.
- I put these entries in a `list` first - separately as `X` and `y` -> `X` has the tokens and `y` has the ground-truth tags.
- I zipped the two lists into one dataframe for training, validation and test sets. 

### Step 2: Text Preprocessing
- **Corresponding File**: `preprocessing.py`
- Drop any missing values from the `csv` files
- Convert string categories to numerical values
  - Class Encoding: `{"PER": 1, "LOC": 2, "MISC": 3, "ORG": 4, "O":0}`
  - The original tags included IOB tags, which were removed
- Text Preprocessing
  - Remove unwanted spaces with `lstrip` and `rstrip`
  - Remove white space characters `['\n', '\t', '\r']`
- Store processed text in new `csv` files

### Step 3: Vectorization
- **Corresponding File**: `training.py` -> `format_word2vec` and `convert_word2vec` function
- What is Word2Vec?
  - `Word2Vec` is an approach used to learn word embeddings for large text datasets
  - `Word embeddings` are nothing but numerical representation of text in the form of vectors
  - Each word is mapped to a corresponding vector which maps characteristics of a word with respect to a document
  - The vector representation helps capture similarities and dissimilarities between words in the document
  - For this project, I built a custom `Word2Vec` model using `gensim`.

- Word2Vec has two approaches
  - Continuous Bag of Words (CBOW)
  - Continuous Skip-Gram
  - For this project, I've cosen the Skip-Gram model over the CBOW model as the skip-gram model had better quality of prediction. Although, the computational complexity is higher.

- With the data I have- a `csv` file of tokens, my first task was to make a corpus of sentences. I grouped words as sentences - each sentence ending with a period `.`
- The corpus is now a list of lists - a list of tokenized sentences
- The corpus contained information from all 3 datasets
- The `Word2Vec` model was then trained on this corpus and the original tokens were then transformed and representec as vectors

### Step 4: Model Training
- **Corresponding File**: `training.py` --> `train_model` function
- The models I've trained include `Random Forest`, `Naive Bayes` and a `Gradient Boosting Machine`
- The models were trained on the vector representation of tokens and the ground truth labels
- I've chosen to work with two ensemble models - Random Forest and Gradient Boosting Machine
- A classification report was constructed for each of the models

### Results

- Model training on 5 classes: `{"PER": 1, "LOC": 2, "MISC": 3, "ORG": 4, "O":0}`

| Model | Training Accuracy | Test Accuracy | 
|-------|-------------------|---------------|
| Random Forest | 98.32% | 93.08% |
| Naive Bayes | 62.33% | 62.78% |
| Gradient Boosting | 87.26% | 86.22% |

- From the table above, we can see that the Random Forest classifer had maximum accuracy of 93% on the test dataset compared to the Naive Bayes and Gradient Boosting models.
- We can observe there is some overfitting in the Naive Bayes model due to the test accuracy being slightly higher than the training accuracy
- The Gradient Boosting model performs well and we can see there is no overfitting. However, it doesn't capture patterns as well as the Random Forest model.


### Sample Outputs
- Sentence: My name is Ashley and I work at Microsoft in America
- Tags for the sentence above was generated using the Random Forest Model

| TOKEN | TAG |
|-------|-----|
| My | O |
| name | O |
| is | O |
| **Ashley** | **PER** |
| and | O |
| I | O | 
| work | O | 
| at | O |
| **Microsoft** | **ORG** | 
| in |  O  |
| **America** | **LOC** | 
| . | O | 

### Future Scope

- Make a web application where users will be able to input a sentence and tagging will be done automatically
- Include more samples and categories
- Train neural networks for better performance

## References

[1] Tjong Kim Sang, E. F., &amp; De Meulder, F. (2003). Introduction to the CONLL-2003 shared task: Language-Independent Named Entity Recognition. Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003  -. https://doi.org/10.3115/1119176.1119195 

[2] Rawlence, L. (n.d.). AIIMI labs on... named-entity recognition. Aiimi. Retrieved April 19, 2022, from https://www.aiimi.com/insights/aiimi-labs-on-named-entity-recognition 

[3] Mikolov, T., Chen, K., Corrado, G., &amp; Dean, J. (2013, September 7). Efficient estimation of word representations in vector space. arXiv.org. Retrieved April 19, 2022, from https://arxiv.org/abs/1301.3781 
