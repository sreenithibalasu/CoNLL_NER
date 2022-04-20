# Usage Notes and Additional Information

- Runs ONLY on Python 3
- Create a virtual environment (recommended but optional)
- Clone repository contents
- Install necessary packages under `requirements.txt`
  - *Command*: `pip install -r requirements.txt`

- Changes to make in `configs.json`
  - `data_path`: path to where the `train, validation and test` sets are stored.
  - `results_path`: path to where `classification reports` are stored
  - *OPTIONAL* changes to `custom_inputs`: add one item to the list
 
NOTE: the folder to `results_path` must be created beforehand by you.

- **First run** `txt_to_csv.py` - this converts the original text files to CSV and stores it in `data_path`
- **Then run** `preprocessing.py` - this refers to the CSV files and does necessary preprocessing to the text and tags
- **Finally run** `training.py` - this takes care of converting text to vectors, model training and testing, and testing out with `custom_inputs`.

## Code Files and Functions
