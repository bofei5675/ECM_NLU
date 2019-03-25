# ECM_NLU

## Traditional Seq2Seq
* get the pre-trained model from google drive, then run below command to chat with traditional Seq2Seq under directory `Baseline/baseline.py`
```
python baseline.py 
```

## Seq2Seq + Emotional Embedding (Trained on daily dialogue)

* Open jupyter notebook `Baseline/jupyter/dialogue_preprocessing.ipynb`, get the dialogue data from google drive.


## Official Implementation of ECM 

* Open directory `ecm/`, follow the readme, to run training on the dataset `daily dialogue`, change emotion category size to 7.
