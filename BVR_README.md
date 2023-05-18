CREATE ENVIRONMENT 

```bash
conda create -n wineq python=3.8 -y
```

ACTIVATE ENVIRONMENT 

```bash
conda activate wineq
```


CREATE REQUIREMENTS.TXT

```bash
dvc
dvc[gdrive]
scikit-learn
```

THEN RUN BELOW COMMAND 

```bash
pip install -r requirements.txt
```

CREATE TEMPLATE.PY TO CREATE NECESSARY DIRECTORY STRUCTURES AND HELPING FILES 

```bash 

import os

dirs = [
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "notebooks",
    "saved_models",
    "src"
]

for dir_ in dirs :
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f :
        pass


files = [
    "dvc.yaml",
    "params.yaml",
    ".gitignore",
    os.path.join("src", "__init__.py")
]

for file_ in files :
    with open(file_, "w") as f :
        pass


```


RUN TEMPLATE.PY 

```bash 

python template.py

```

CREATE data_given folder 

```bash 
mkdir data_given

```




COPY DATA FILE FROM GDRIVE TO data_given folder 

```BASH 

https://drive.google.com/drive/folders/1xw0XX-WK74uxtFFLySbtnX-ODdmdK5Ec

```

TRY BELOW COMMANDS 

```BASH
git init
dvc init
dvc add data_given/winequality.csv
git add .
git commit -m "first commit"

```


MULTIPLE COMMANDS IN SINGLE LINE 

```bash
git add . &&  git commit -m "bvr_readme.md file got updated"
```


COMMIT CHANGES TO REMOTE REPO

```BASH

git add .
git commit -m "first commit"
git remote add origin https://github.com/vishymails/sample_app.git
git branch -M main
git push -u origin main

```


CREATE PARAMS.YAML 

```BASH

base :
  project : winequality-project
  random_state : 42
  target_col : TARGET

data_source : 
  s3_source : data_given/winequality.csv

load_data :
  raw_dataset_csv : data/raw/winequality.csv

split_data :
  train_path : data/processed/train_winequality.csv
  test_path : data/processed/test_winequality.csv
  test_size : 0.2

estimators :
  ElasticNet :
    params :
      # alpha : 0.88
      # l1_ratio : 0.89
      alpha : 0.9
      l1_ratio : 0.4

model_dir : saved_models

reports :
  params : report/params.json
  scores : report/scores.json

webapp_model_dir : prediction_service/model/model.joblib



```


```bash
git add . &&  git commit -m "param.yaml file added "
```


```bash

git push -u origin main
```



UPDATE REQUIREMENTS.TXT

```BASH

dvc
dvc[gdrive]
scikit-learn
pandas
pytest
tox
flake8
flask
gunicorn


```


```BASH

pip install -r requirements.txt

```


CREATE src/get_data.py

```bash
## 1. read parameters
## 2. process
## 3. return dataframe 


import os
import yaml 
import pandas as pd
import argparse 

def read_params(config_path) :
    with open(config_path) as yaml_file :
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path) :
    config = read_params(config_path)
   
   # reading param file as an argument 
   # print(config)

    data_path = config["data_source"]["s3_source"]

    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    print(df)
    return df

if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)



```

EXEUTE BELOW COMMANDS 


```bash

python src/get_data.py
git add . &&  git commit -m "params.yaml added"
git push -u origin main

```


WRITE LOAD_DATA.PY 

```bash

# Read the data from datasource save it in the data/raw for further process

import os
from get_data import read_params, get_data
import argparse

def load_and_save(config_path) :
    config = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ", "_") for col in df.columns]
    print(new_cols)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, sep=",", index=False, header=new_cols)


if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)


```

EXECUTE BELOW COMMANDS 


```bash

python src/load_data.py
git add . &&  git commit -m "get_data.py and load_data.py got added"
git push -u origin main
```


UPDATE DVC.YAML 

```BASH 

stages :
  load_data :
    cmd : python src/load_data.py --config=params.yaml
    deps :
      - src/get_data.py
      - src/load_data.py
      - data_given/winequality.csv
    outs :
      - data/raw/winequality.csv

```

RUN 

```BASH

dvc repro

git add . &&  git commit -m "dvc.yaml LOAD DATA SECTION GOT updated"
git push -u origin main
```


UPDATE SPLIT_DATA SECTION IN DVC.YAML FILE

```BASH

stages :
  load_data :
    cmd : python src/load_data.py --config=params.yaml
    deps :
      - src/get_data.py
      - src/load_data.py
      - data_given/winequality.csv
    outs :
      - data/raw/winequality.csv

  split_data :
    cmd : python src/split_data.py --config=params.yaml
    deps :
      - src/split_data.py
      - data/raw/winequality.csv
    outs :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv



```

RUN 

```BASH

dvc repro

git add . &&  git commit -m "dvc.yaml SPLIT DATA SECTION GOT updated"
git push -u origin main
```



CREATE TRAIN_AND_EVALUATE.PY 

```bash

#load the train and test
# train algorithm
#save the metrics and params

import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred) :
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = ElasticNet(
        alpha=alpha, 
        l1_ratio=l1_ratio, 
        random_state=random_state)
    lr.fit(train_x, train_y)


    predicted_qualitites = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualitites)

    print("ElasticNet model (alpha=%f, l1_ratio=%f) : " % (alpha, l1_ratio))

    print("RMSE : %s" % rmse)
    print("MAE : %s" % mae)
    print("R2 : %s" % r2)
    


    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    
    joblib.dump(lr, model_path)




if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)



```

UPDATE TRAIN_AND_UPDATE STAGE IN DVC.YAML (FULL CODE HAS BEEN SHARED )

```bash 

stages :
  load_data :
    cmd : python src/load_data.py --config=params.yaml
    deps :
      - src/get_data.py
      - src/load_data.py
      - data_given/winequality.csv
    outs :
      - data/raw/winequality.csv


  split_data :
    cmd : python src/split_data.py --config=params.yaml
    deps :
      - src/split_data.py
      - data/raw/winequality.csv
    outs :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv


  train_and_evaluate :
    cmd : python src/train_and_evaluate.py --config=params.yaml
    deps :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv
      - src/train_and_evaluate.py
    params :
      - estimators.ElasticNet.params.alpha
      - estimators.ElasticNet.params.l1_ratio
   # metrics :
   #   - report/scores.json :
   #       cache : false
   #   - report/params.json :
   #      cache : false
    outs :
      - saved_models/model.joblib

```



RUN 

```BASH

dvc repro

git add . &&  git commit -m "dvc.yaml SPLIT DATA SECTION GOT updated"
git push -u origin main
```


CREATE REPORT FOLDERS AND FILES 

```BASH 

mkdir report
touch report/params.json
touch report/scores.json

```


UPDATE TRAIN_AND_EVALUATE.PY (UPDATE ONLY METRICS SECTION )

```BASH 

#load the train and test
# train algorithm
#save the metrics and params

import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred) :
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = ElasticNet(
        alpha=alpha, 
        l1_ratio=l1_ratio, 
        random_state=random_state)
    lr.fit(train_x, train_y)


    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("ElasticNet model (alpha=%f, l1_ratio=%f) : " % (alpha, l1_ratio))

    print("RMSE : %s" % rmse)
    print("MAE : %s" % mae)
    print("R2 : %s" % r2)
    

    #***************************************METRICS REPORT DATA ************************************************************

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f :
        scores = {
            "rmse" : rmse,
            "mae" : mae,
            "r2" : r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f :
        params = {
            "alpha" : alpha,
            "l1_ratio" : l1_ratio
        }
        json.dump(params, f, indent=4)





    ######################################################################################################################
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    
    joblib.dump(lr, model_path)




if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)


```

UNCOMMENT OR ADD METRICS SECTION IN DVC.YAML FILE 

```BASH 

train_and_evaluate :
    cmd : python src/train_and_evaluate.py --config=params.yaml
    deps :
      - data/processed/train_winequality.csv
      - data/processed/test_winequality.csv
      - src/train_and_evaluate.py
    params :
      - estimators.ElasticNet.params.alpha
      - estimators.ElasticNet.params.l1_ratio
    metrics :
      - report/scores.json :
          cache : false
      - report/params.json :
         cache : false
    outs :
      - saved_models/model.joblib

```


RUN 

```BASH

dvc repro

git add . &&  git commit -m "METRICS AND JSON FILE CREATED AND updated"
git push -u origin main
```


RUN TO SEE METRICS DIFFERENCE 


```BASH 

git add . && git commit -m "tracker added"
git push origin main
dvc params diff
dvc metrics show
dvc metrics diff
dvc repro
dvc metrics diff
git add . && git commit -m "tracker added"
git push origin main
dvc metrics diff

DO NOT COMMIT TRY BELOW COMMANDS 


dvc metrics show
dvc metrics diff
git add . && git commit -m "tracker added 2"
git push origin main
doskey /history

```

ENABLE PYTEST AND TOX FRAMEWORKS FOR TESTING PURPOSE - CREATE tox.ini FILE IN SAMPLE_APP FOLDER 

```BASH 

[tox]
envlist = py38
; skipsdist = True

[testenv]
deps = -rrequirements.txt
commands =
    pytest -v

```


CREATE NECESSARY FOLDERS AND FILES FOR SAMPLE TESTING 


```BASH 

mkdir tests

 16  cd tests 
   17  cd ..
   18  touch tests/conftest.py
   19  touch tests/test_config.py
   20  touch tests/__init__.py

```


