# Mario
An implementation of the TOSEM paper titled "Pre-Implementation Method Name Prediction for Object-Oriented Programming" in PyTorch.

The link to the Zenodo repository is 10.5281/zenodo.5241894.

# Requirements
* nltk
* numpy
* tokenizers
* torch
* tqdm
* dill
* optim

# Dataset
```bash
wget https://s3.amazonaws.com/code2vec/data/java-large_data.tar.gz
tar -axf java-large_data.tar.gz -C JavaRepos_all
```


# Baselines
## Baseline#1
This baseline recommends a setter and a getter method names for each involved field.
```bash
cd src
python baseline#1.py
```

## Baseline#2
This baseline always recommends all the method names from the class that is the most similar one to it according to the similarity between their semantic class names
```bash
cd src
python baseline#2.py
```

## Baseline#3
Given a specific class, this baseline recommends the top nine method names that occur the most frequently in its proximate classes.
```bash
cd src
python baseline#3.py
```

# Training
```bash
cd src
python train_shuffled.py \
    -method_data [path/to/training/data].json \
    -test_data [path/to/valid/data].json \
```

# Evaluation
Evalaute the performance on field-relevant methods. The [threshold] means the threshold mentioned in paper.
```bash
cd src
python Evaluate_FR.py \
    -threshold [threshold]
```

Evaluate the performance of Mario, which includes both field-relevant and field-irrelevant predictions. The [FR_PredRes.json] refers to the results generated by Evaluate_FR.py.

```bash
cd src
python Evaluate_FR_FIR.py \
    -test_data [path/to/test/data].json \
    -FR_data [FR_PredRes.json]
```
