![PyPI - Downloads](https://img.shields.io/pypi/dw/pandas-ml-utils)



## TODO
* allow multiple class for classification 
* replace hard coded summary objects by a summary provider function 
* add more tests
* add Proximity https://stats.stackexchange.com/questions/270201/pooling-levels-of-categorical-variables-for-regression-trees/275867#275867

## Wanna help?
* currently I only need binary classification
    * maybe you want to add a feature for multiple classes
* for non classification problems you might want to augment the `Summary` 
* write some tests
* add different more charts for a better understanding/interpretation of the models
* implement hyper parameter tuning
* add feature importance 

## Change Log
### 0.0.12
* added sphinx documentation
* added multi model as regular model which has quite a big impact
  * features and labels signature changed
  * multiple targets has now the consequence that a lot of things a returning a dict now
  * everything is using now DataFrames instead of arrays after plain model invoke
* added some tests
* fixed some bugs a long the way

### 0.0.11
* Added Hyper parameter tuning 
```python
from hyperopt import hp

fit = df.fit_classifier(
            pdu.SkitModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                          pdu.FeaturesAndLabels(features=['vix_Close'], labels=['label'],
                                                targets=("vix_Open", "spy_Volume"))),
            test_size=0.4,
            test_validate_split_seed=42,
            hyper_parameter_space={'alpha': hp.choice('alpha', [0.001, 0.1]), 'early_stopping': True, 'max_iter': 50,
                                   '__max_evals': 4, '__rstate': np.random.RandomState(42)})
```
NOTE there is currently a bug in hyperot [module bson has no attribute BSON](https://github.com/hyperopt/hyperopt/issues/547)
! However there is a workaround:
```bash
sudo pip uninstall bson
pip install pymongo
``` 

### 0.0.10
* Added support for rescaling features within the auto regressive lags. The following example
re-scales the domain of min/max(featureA and featureB) to the range of -1 and 1. 
```python
FeaturesAndLabels(["featureA", "featureB", "featureC"],
                  ["labelA"],
                  feature_rescaling={("featureA", "featureC"): (-1, 1)})
```
* added a feature selection functionality. When starting from scratch this just helps
to analyze the data to find feature importance and feature (auto) correlation.
I.e. `df.filtration(label_column='delta')` takes all columns as features exept for the
delta column (which is the label) and reduces the feature space by some heuristics.
