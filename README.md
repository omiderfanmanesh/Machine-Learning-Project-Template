# bank-marketing

Bank marketing campaign is a method for increasing the number of clients and in this paper, I am going to analyse the
data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based
on phone calls.This current analysis consider the impact of different classifier models and pca component analysis on
this dataset. Moreover, all these model will be evaluated with metric of f1 score. As the result, I obtained the score
of 0.90 from Random forest classifier model.

# How to use this template

## Template structure:

```
.
├── build
├── configs
│   ├── defaults.py -it has some specific configs for models or dataset.
│   ├── __init__.py
├── data
│   ├── bank.py - Bank is a sub-class of BasedDataset and it contains all configs to load the dataset from files
│   ├── based
│   │   ├── based_dataset.py -a super class for all dataset, it contains data manipulations methods
│   │   ├── encoder_enum.py -contains constants of encoders
│   │   ├── file_types.py -contains constants of file type e.g. CSV, TSV, TXT,...
│   │   ├── __init__.py
│   │   ├── sampling_types.py -contains constants of sampling types e.g. SMOTE,...
│   │   ├── scale_types.py -contains constants of scalers e.g. MinMax Scaler
│   │   └── transformers_enums.py -contains types of data transformers e.g. Log Transformer
│   ├── dataset -include all datasets files
│   │   ├── bank-additional
│   │   │   ├── bank-additional.csv
│   │   │   ├── bank-additional-full.csv
│   │   │   └── bank-additional-names.txt
│   │   ├── bank.csv
│   │   ├── bank-full.csv
│   │   ├── bank-names.txt
│   │   ├── description.txt
│   ├── __init__.py
│   ├── loader.py - it loads the dataset that we need
│   ├── preprocessing
│   │   ├── encoders.py - Encoder helps to encode data and has all codes needed for this purpose
│   │   ├── __init__.py
│   │   ├── pca.py - PCA is for applying pca analysis and drawing plots
│   │   └── scalers.py - Scaler is for changing the scale of values in a dataset
├── eda -ontcains files and folders that are needed for analysing and retrieving information about the dataset, features, categories, ...
│   ├── bank_analyser.py -contains codes for retrieving information from a dataset
│   ├── bank_plot.py -contains codes related to draw plots from a dataset
│   ├── based
│   │   ├── based_analyzer.py -BankAnalyzer is a sub-class of BasedAnalyzer and it is help to analyse features
│   │   ├── based_plots.py -BankPlots is a sub-class of BasedPlot that helps to plot features' values
│   │   ├── __init__.py
│   ├── __init__.py
├── engine
│   ├── analyser.py -this file runs BankAnalyzer
│   ├── __init__.py
│   └── trainer.py -this file is for training, hyperparameters tuning and doing cross-validation
├── LICENSE
├── model
│   ├── based
│   │   ├── based_model.py -this is a super class for all models that contains methods for training, evaluation, plotting, ...
│   │   ├── __init__.py
│   │   ├── metric_types.py -contains constants of metrics
│   │   ├── model_selection.py -contains the name of models that you can run for your dataset
│   │   ├── task_mode.py -it helps you to select what task you are going to do e.g. classification or regression
│   │   └── tuning_mode.py -contains the types of tuning algorithms e.g GridSearch
│   ├── decision_tree.py -contains all configuration needed to run Decision Tree algorithms
│   ├── __init__.py
│   ├── logistic_regression.py -contains all configuration needed to run Logistic Regression algorithm
│   ├── random_forest.py -contains all configuration needed to run Random forest algorithms
│   └── svm.py - contains all configuration needed to run Support vector machine algorithms
├── notebooks - contains all notebooks that is needed when you want to run this project by jupiter notebook
│   ├── data_exploration.ipynb
│   └── trainer.ipynb
├── README.md
├── report
│   └── report.pdf
├── test
│   └── test.py
├── tools
│   ├── analysis.py -contains main() for running the analyser.py
│   └── train.py -contains main() for running the trainer.py
└── utils
    ├── __init__.py
    └── runtime_mode.py -all types of running modes for this project e.g. training, tuning or cross-validation

```

Just add your dataset file to *data/dataset* and put your all configuration at *config/defaults.py*. I will explain
configuration settings below:


## BASIC CONFIGS

Initialize the BASIC configs for our project:

```python
_C.BASIC = CN()
```

set the Random SEED number to use as random_state, ...

```python
_C.BASIC.SEED = 2021
```

pca = True will apply principal component analysis to data.

```python
_C.BASIC.PCA = False 
```

select training model e.g. SVM, RandomForest, ...

```python
_C.BASIC.MODEL = Model.SVM 
```

runtime mode means how you are going to run this code. if you need to train model use *RuntimeMode.TRAIN* or if you need
to do cross validation use *RuntimeMode.CROSS_VAL*
and finaly if you need to do hyperparameter tuning use *RuntimeMode.TUNING*

```python
_C.BASIC.RUNTIME_MODE = RuntimeMode.TRAIN  
```

task mode means this is a classification task or regression

```python
_C.BASIC.TASK_MODE = TaskMode.CLASSIFICATION  
```

data resampling is a technique for handling imbalance dataset. if you set SAMPLING_STRATEGY to *None*, it means don't
use resampling. if you set SAMPLING_STRATEGY to one of (Sampling.SMOTE, Sampling.RANDOM_UNDER_SAMPLING), it will be
applied to train set of your dataset. **note that the order is important** e.g. if you want to do first over sampling
then under sampling, you can write
*(Sampling.SMOTE, Sampling.RANDOM_UNDER_SAMPLING)*. It will apply first SMOTE then RANDOM_UNDER_SAMPLING.

```python
_C.BASIC.SAMPLING_STRATEGY = None # don't use any sampling strategy
# _C.BASIC.SAMPLING_STRATEGY = (Sampling.SMOTE) # use just SMOTE
# _C.BASIC.SAMPLING_STRATEGY = (Sampling.SMOTE, Sampling.RANDOM_UNDER_SAMPLING) # use first SMOTE then RANDOM_UNDER_SAMPLING
```

## MODEL CONFIGS

Initialize the MODEL configs for our project:

```python
_C.MODEL = CN()
```

Number of target classes for classification task. In this our case, we have a binary classification model.

```python
_C.MODEL.NUM_CLASSES = 2
```

if you set *_C.BASIC.RUNTIME_MODE = RuntimeMode.CROSS_VAL*, the value of K will be used for KFold cross-validation.

```python
_C.MODEL.K_FOLD = 5  
```

If you want to shuffle the data set it to True otherwise False

```python
_C.MODEL.SHUFFLE = True  
```

## Dataset

Initialize the DATASET configs for our project:

```python
_C.DATASET = CN()
```

write the address of dataset file

```python
_C.DATASET.DATASET_ADDRESS = '../data/dataset/bank.csv'  
```

if you have a brief description file about your dataset, write the address here

```python
_C.DATASET.DATASET_BRIEF_DESCRIPTION = '../data/dataset/description.txt'  

```

this is name of target column of our dataset

```python
_C.DATASET.TARGET = 'y' 

```

if you have a categorical targets,set True otherwise False

```python
_C.DATASET.HAS_CATEGORICAL_TARGETS = True
```

if you want to drop specific columns from dataframe, write their names in DROP_COLS

```python

_C.DATASET.DROP_COLS = (
    # 'duration', # it will not drop
    'day', # it will drop from df
    'balance',
    'month',
    'job',
    'previous',
    'campaign',
    'education',
    # 'poutcome',
    # 'age',
    'pdays',
    'marital',
    'contact',
    'housing',
    'loan',
    # 'default'
)
```

## METRIC


Initialize the EVALUATION configs for our project:

```python
_C.EVALUATION = CN()
```

select your metric for your model. Supported metrics are
'accuracy', 'balanced_accuracy',  'top_k_accuracy',
'average_precision',  'neg_brier_score', 'f1',
'f1_micro', 'f1_macro',  'f1_weighted',
'f1_samples',  'neg_log_loss', 'precision',
'recall',  'jaccard', 'roc_auc',
'roc_auc_ovr', 'roc_auc_ovo',  'roc_auc_ovr_weighted',
'roc_auc_ovo_weighted'

```python
_C.EVALUATION.METRIC = MetricTypes.F1_SCORE_MICRO  

```

set True if you need to plot the confusion matrix at the end of evaluation process

```python
_C.EVALUATION.CONFUSION_MATRIX = False  
```

## CATEGORICAL FEATURES ENCODER CONFIG

Initialize the ENCODER configs for our project:

```python
_C.ENCODER = CN()
```

if you have categorical column, write its name in CAPITAL letter and follow this structure:

*_C.ENCODER.{COLUMN NAME} = TYPE OF ENCODER*

```python

_C.ENCODER.JOB = EncoderTypes.BINARY
_C.ENCODER.MARITAL = EncoderTypes.BINARY
_C.ENCODER.EDUCATION = EncoderTypes.ORDINAL
_C.ENCODER.DEFAULT = EncoderTypes.BINARY
_C.ENCODER.HOUSING = EncoderTypes.BINARY
_C.ENCODER.LOAN = EncoderTypes.BINARY
_C.ENCODER.CONTACT = EncoderTypes.BINARY
_C.ENCODER.MONTH = EncoderTypes.ORDINAL
_C.ENCODER.POUTCOME = EncoderTypes.BINARY
_C.ENCODER.Y = EncoderTypes.LABEL  # if your target is categorical
```

if you want to do use a custom encoders set for example:

```python
_C.ENCODER.Y = EncoderTypes.CUSTOM
```

then you should write a method inside your dataset class and call it before encoding phase. CUSTOM means you don't want
to use encoders of *EncoderTypes*.


## SCALER

Initialize the SCALER configs for our project:

```python
_C.SCALER = CN()
```

select the type of scaler (STANDARD SCALER, MINMAX SCALER, ...) that you want to apply to your data

```python
_C.SCALER = ScaleTypes.STANDARD 
```

## TRANSFORMATION

Initialize the TRANSFORMATION configs for our project:

```python
_C.TRANSFORMATION = CN()
```

If you need to transform a specific column, write transformation type here

```python
_C.TRANSFORMATION.AGE = TransformersType.BOX_COX  #
_C.TRANSFORMATION.BALANCE = TransformersType.BOX_COX  #
_C.TRANSFORMATION.DAY = TransformersType.NONE  #
_C.TRANSFORMATION.DURATION = TransformersType.BOX_COX  #
_C.TRANSFORMATION.CAMPAIGN = TransformersType.BOX_COX  #
_C.TRANSFORMATION.PDAYS = TransformersType.LOG  #
_C.TRANSFORMATION.PREVIOUS = TransformersType.BOX_COX  #
```

## DECOMPOSITION

if you set ```_C.BASIC.PCA = True```, the PCA configuarion will be loaded from this section:

Initialize the SCALER configs for our project:

```python
_C.PCA = CN()
```

number of components

```python
_C.PCA.N_COMPONENTS = 0.7  
```

set True if you want to plot pca components

```python
_C.PCA.PLOT = False   
```

## SAMPLING

All configuration that we need for data resampling is in this section. For example,  
if you set *_C.BASIC.SAMPLING_STRATEGY = (Sampling.SMOTE)*, The configs of SMOTE will be loaded from this part.

```python
_C.RANDOM_UNDER_SAMPLER = CN()
_C.RANDOM_UNDER_SAMPLER.SAMPLING_STRATEGY = 'auto'  # float, str, dict, callable, default=’auto’
_C.RANDOM_UNDER_SAMPLER.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.RANDOM_UNDER_SAMPLER.REPLACEMENT = False  # bool, default=False

_C.RANDOM_OVER_SAMPLER = CN()
_C.RANDOM_OVER_SAMPLER.SAMPLING_STRATEGY = 'minority'  # float, str, dict or callable, default=’auto’
_C.RANDOM_OVER_SAMPLER.RANDOM_STATE = 2021  # int, RandomState instance, default=None
# _C.RANDOM_OVER_SAMPLER.SHRINKAGE = 0  # float or dict, default=None

_C.SMOTE = CN()
_C.SMOTE.SAMPLING_STRATEGY = 'auto'  # float, str, dict or callable, default=’auto’ {'minority'}
_C.SMOTE.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SMOTE.K_NEIGHBORS = 5  # int or object, default=5
_C.SMOTE.N_JOBS = -1  # int, default=None

_C.SMOTENC = CN()
_C.SMOTENC.CATEGORICAL_FEATURES = ('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                                   'month')  # ndarray of shape (n_cat_features,) or (n_features,)
_C.SMOTENC.SAMPLING_STRATEGY = 'minority'  # float, str, dict or callable, default=’auto’
_C.SMOTENC.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SMOTENC.K_NEIGHBORS = 5  # int or object, default=5
_C.SMOTENC.N_JOBS = -1  # int, default=None

_C.SVMSMOTE = CN()
_C.SVMSMOTE.SAMPLING_STRATEGY = 'auto'  # float, str, dict or callable, default=’auto’ {'minority'}
_C.SVMSMOTE.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SVMSMOTE.K_NEIGHBORS = 3  # int or object, default=5
_C.SVMSMOTE.N_JOBS = -1  # int, default=None
_C.SVMSMOTE.M_NEIGHBORS = 10  # int or object, default=10
# _C.SVMSMOTE.SVM_ESTIMATOR = 5  # estimator object, default=SVC()
_C.SVMSMOTE.OUT_STEP = 0.5  # float, default=0.5
```

## Models


when you select ```_C.BASIC.MODEL = Model.SVM ```, the configuration of SVM model will be loaded from here:

support vector machine for classification task

```python
_C.SVM = CN()
_C.SVM.NAME = 'SVM'

_C.SVM.C = 10  # float, default=1.0
_C.SVM.KERNEL = 'rbf'  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
_C.SVM.DEGREE = 1  # int, default=3
_C.SVM.GAMMA = 'scale'  # {'scale', 'auto'} or float, default='scale'
_C.SVM.COEF0 = 0.0  # float, default=0.0
_C.SVM.SHRINKING = True  # bool, default=True
_C.SVM.PROBABILITY = False  # bool, default=False
_C.SVM.TOL = 1e-3  # float, default=1e-3
_C.SVM.CACHE_SIZE = 200  # float, default=200
_C.SVM.CLASS_WEIGHT = None  # dict or 'balanced', default=None
_C.SVM.VERBOSE = True  # bool, default=False
_C.SVM.MAX_ITER = -1  # int, default=-1
_C.SVM.DECISION_FUNCTION_SHAPE = 'ovr'  # {'ovo', 'ovr'}, default='ovr'
_C.SVM.BREAK_TIES = False  # bool, default=False
_C.SVM.RANDOM_STATE = _C.BASIC.RAND_STATE  # int or RandomState instance, default=None

```

support vector machine for regression task

```python

_C.SVR = CN()
_C.SVR.NAME = 'SVM'

_C.SVR.KERNEL = 'rbf'  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
_C.SVR.DEGREE = 3  # int, default=3
_C.SVR.GAMMA = 'scale'  # {'scale', 'auto'} or float, default='scale'
_C.SVR.COEF0 = 0.0  # float, default=0.0
_C.SVR.TOL = 1e-3  # float, default=1e-3
_C.SVR.C = 1.0  # float, default=1.0
_C.SVR.EPSILON = 0.1  # float, default=0.1
_C.SVR.SHRINKING = True  # bool, default=True
_C.SVR.CACHE_SIZE = 200  # float, default=200
_C.SVR.VERBOSE = True  # bool, default=False
_C.SVR.MAX_ITER = -1  # int, default=-1
```

when you set ```_C.BASIC.RUNTIME_MODE = RuntimeMode.TUNING   ```, configurations for hyperparameter tuning of selected
model (e.g. SVM) will be loaded from:

```python
_C.SVM.HYPER_PARAM_TUNING = CN()
_C.SVM.HYPER_PARAM_TUNING.KERNEL = ('linear', 'poly', 'rbf')
_C.SVM.HYPER_PARAM_TUNING.C = (0.1, 1, 10)
_C.SVM.HYPER_PARAM_TUNING.DEGREE = (1, 2, 3)
_C.SVM.HYPER_PARAM_TUNING.GAMMA = ('scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001)
_C.SVM.HYPER_PARAM_TUNING.COEF0 = None
_C.SVM.HYPER_PARAM_TUNING.SHRINKING = None
_C.SVM.HYPER_PARAM_TUNING.PROBABILITY = None
_C.SVM.HYPER_PARAM_TUNING.TOL = None
_C.SVM.HYPER_PARAM_TUNING.CACHE_SIZE = None
_C.SVM.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.SVM.HYPER_PARAM_TUNING.MAX_ITER = None
_C.SVM.HYPER_PARAM_TUNING.DECISION_FUNCTION_SHAPE = None
_C.SVM.HYPER_PARAM_TUNING.BREAK_TIES = None

```

