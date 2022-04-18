# Named Entity Recognition

- **Named Entity Recognition (NER)** aims to categorize unstructured text into pre-defined categories such as person name, organization name, location, phone number, email etc. The idea of doing NER came to me because I got a taste of it during an internship I did in India. 
- NER can be an important task because it can help identify key elements in unstructured data such as text. For example: if we had an image detection and recognition system that extracts text from business cards, categorizing that text automatically into pre-defined categories such as name, organization, email address etc. can help us build structured datasets to store that useful information in.

## Dataset
- I've chosen the CoNLL 2003 [1] dataset. The task for this "competition" was to do language independent NER. The entire dataset consisted of English and German text. For the purpose of this project, I've chosen to include only the english language text.

| English Text | Tokens | LOC | PER | ORG | MISC |
|---|---|---|---|---|---|---|
|  Training | 203,621 | 7140 | 6600 | 6321 | 3438 |
|  Validation | 51,362 | 1837 | 1842 | 1341 | 922 |
|  Testing | 46,435 | 1668 | 1617 | 1661 | 702 |
