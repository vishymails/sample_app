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


```