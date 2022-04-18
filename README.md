# Named Entity Recognition

- **Named Entity Recognition (NER)** aims to categorize unstructured text into pre-defined categories such as person name, organization name, location, phone number, email etc. The idea of doing NER came to me because I got a taste of it during an internship I did in India. 
- NER can be an important task because it can help identify key elements in unstructured data such as text. For example: if we had an image detection and recognition system that extracts text from business cards, categorizing that text automatically into pre-defined categories such as name, organization, email address etc. can help us build structured datasets to store that useful information in.

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
