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
envlist = py39
; skipsdist = True

[testenv]
deps = -rrequirements.txt
commands = 

    # stop the build if there are Python syntax errors or undefined names
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    pytest -v

```

RUN 

```BASH

tox 

OR 

tox -R (this command will delete all libs and reload it once again)

```


IF DISTRIBUTION IS SUPPORTED THEN IT EXPECTS setup.py FILE 

```BASH 

under sample_app folder 

touch setup.py

```


SETUP.PY 

```bash 


from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="Case study project for Oracle Blr",
    author="BVR",
    packages=find_packages(),
    license="MIT"
)


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


UPDATE test_config.py


```bash 

def test_generic() :
    a = 30
    b = 40

    assert a == 30

```


RUN 

```bash 

tox 

```


TO CREATE FINAL DISTRIBUTION FOR SHARING THE PROJECT WE CAN USE WHEEL 

```BASH 

tox
pip install -e .
pip freeze
python setup.py sdist bdist_wheel

LOOK IN TO dist FOLDER - YOU CAN FIND ZIP FILE + WHL FILE (SHARE THE SAME TO ALL WHO LIKES TO USE OR ENHANCE THE PROJECT)
```



CREATE NOTEBOOKS FOR TESTING SOME ROUGH NOTE CODE BEFORE USING

```BASH

pip install jupyterlab
jupyter-lab notebooks/


```


```BASH

import os
import pandas as pd




df = pd.read_csv("../data_given/winequality.csv")

df.columns





df.describe().T




df.columns = [ '_'.join(col.split()) for col in df.columns]




df.describe().T




overview = df.describe()

overview.loc[["min", "max" ]]





class NotInRange(Exception) :
    def __init__(self, message="value not in given range - by Oracle") :
        self.message = message
        super().__init__(self.message)
        




raise NotInRange





overview.loc[["min", "max" ]].to_dict()




overview.loc[["min", "max" ]].to_json()




overview.loc[["min", "max" ]].to_json("schema_in.json")

# CREATES schema_in.json for further use
```

```BASH 

git add . && git commit -m "NOTEBOOK FILE ADDED  "
git push origin main

```


UPDATE test_config.py 

```bash 

import pytest

class NotInRange(Exception) :
    def __init__(self, message="value not in given range - by Oracle") :
        self.message = message
        super().__init__(self.message)
        




def test_generic() :
    a = 30
    b = 40

    assert a == 30


def test_generic1() :
    a = 500
    with pytest.raises(NotInRange) :
        if a not in range(10, 200) :
            raise NotInRange

```

CREATE NECESSARY FOLDER AND FILES FOR CONSUMING ML OPS GENERATED MODEL AND BE PART OF MLOPS

```BASH

24  mkdir -p prediction_service/model
   25  mkdir webapp
   26  touch app.py
   27  touch prediction_service/__init__.py
   28  touch prediction_service/prediction.py
   29  mkdir -p webapp/static/css
   30  mkdir -p webapp/static/script
   31  touch webapp/static/css/main.css
   32  touch webapp/static/script/index.js
   33  mkdir webapp/templates
   34  touch webapp/templates/index.html
   35  touch webapp/templates/404.html
   36  touch webapp/templates/base.html


```


UPDATE MAIN.CSS

```BASH

body {
    background-color: #eff8ff;
}


```


UPDATE INDEX.HTML

```BASH

{% extends 'base.html' %}

{% block title %}
WineQuality
{% endblock title %}


{% block custom_css %}
<!-- Custom CSS -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
{% endblock custom_css %}


{% block body %}
<!-- nav bar -->

<nav class="navbar navbar-expand-lg navbar-light shadow fixed-top" style="background-color: #e3f2fd;">
    <a class="navbar-brand" href="/">WineQualityPrediction</a>
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav"
        aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav">
            <li class="nav-item active">
                <a class="nav-link" href="/">Home <span class="sr-only">(current)</span></a>
            </li>
           
        </ul>
    </div>
</nav>
<div>
    <!-- nav bar ends -->

    <div class="container-fluid masthead">
        <br>
        <br>
        <br>
        <br>
        <div class="container">
            <div class="row">
                <div class="col">
                    <form method="POST">
                        <!-- Input block-->
                        <div class="form-group">
                            <label for="translation">
                                <blockquote class="blockquote">
                                    <p class="mb-2">Enter the details as indicated:</p>
                                </blockquote>
                            </label>
                            <textarea class="form-control" name="fixed_acidity" rows="1"
                                placeholder="fixed acidity expected range 4.6 to 15.9"></textarea>
                            <textarea class="form-control" name="volatile_acidity" rows="1"
                                placeholder="volatile acidity expected range 0.12 to 1.58"></textarea>
                            <textarea class="form-control" name="citric_acid" rows="1"
                                placeholder="citric acid expected range 0.0 to 1.0"></textarea>
                            <textarea class="form-control" name="residual_sugar" rows="1"
                                placeholder="residual sugar expected range 0.9 to 15.5"></textarea>
                            <textarea class="form-control" name="chlorides" rows="1" 
                                placeholder="chlorides expected range 0.012 to 0.611"></textarea>
                            <textarea class="form-control" name="free_sulfur_dioxide" rows="1"
                                placeholder="free sulfur dioxide expected range 1.0 to 72.0"></textarea>
                            <textarea class="form-control" name="total_sulfur_dioxide" rows="1"
                                placeholder="total sulfur dioxide expected range 6.0 to 289.0"></textarea>
                            <textarea class="form-control" name="density" rows="1" 
                                placeholder="density expected range 0.99007 to 1.00369"></textarea>
                            <textarea class="form-control" name="pH" rows="1" 
                                placeholder="pH expected range 2.74 to 4.01"></textarea>
                            <textarea class="form-control" name="sulphates" rows="1" 
                                placeholder="sulphates expected range 0.33 to 2.0"></textarea>
                            <textarea class="form-control" name="alcohol" rows="1" 
                                placeholder="alcohol expected range 8.4 to 14.9"></textarea>

                        </div>

                        <!-- Select output language here. -->
                        <div class="form-group">
                        </div>
                        <button type="submit" class="btn btn-primary mb-2">Predict</button></br>
                        </br>
                    </form>
                    <!-- Translated text returned by the Translate API is rendered here. -->
                </div>
                <div class="col">
                    <form>
                        <div class="form-group">
                            <label for="translation-result">

                                <blockquote class="blockquote">
                                    <p class="mb-2">Prediction:</p>
                                </blockquote>

                            </label>
                            <textarea readonly class="form-control" id="exTextarea" rows="5">{{ response }}</textarea>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        {% endblock body %}

        {% block custom_js %}
        <!-- Custom JS -->
        <script src="{{ url_for('static', filename='script/index.js') }}"></script>
        {% endblock custom_js %}
        

```


UPDATE BASE.HTML

```BASH
 
        <!doctype html>
        <html lang="en">
        
        <head>
          <!-- Required meta tags -->
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        
          <!-- Bootstrap CSS -->
          <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css"
            integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
          {% block custom_css %}
          {% endblock custom_css %}
        
          <title>
            {% block title %}
            {% endblock title %}
          </title>
        </head>
        
        <body>
          {% block body %}
          {% endblock body %}
        
          <!-- Optional JavaScript -->
          <!-- jQuery first, then Popper.js, then Bootstrap JS -->
          <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js"
            integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n"
            crossorigin="anonymous"></script>
          <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"
            integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo"
            crossorigin="anonymous"></script>
          <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"
            integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6"
            crossorigin="anonymous"></script>
        
          <!-- jQuery google CDN-->
          <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
        
          {% block custom_js %}
          {% endblock custom_js %}
        </body>
        
        </html>
        
    

```



UPDATE 404.HTML

```BASH


{% extends 'base.html' %}

{% block title %}
Error 404
{% endblock title %}


{% block custom_css %}
<!-- Custom CSS -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
{% endblock custom_css %}

{% block body %}

    <div class="container col-lg-4 col-lg-offset-4 main">
        <div class="row">
            <div class="col-md-12">
                <div class="error-template">
                    <h1> Oops! </h1>
                    <h2> 404 Not Found </h2>
                    <div class="error-details">
                        ERROR: {{ error.error }}
                    </div>
                    <div class="error-actions go-home">
                        <a href="/" class="btn btn-primary btn-lg"><span class="glyphicon glyphicon-home"></span>
                            Take Me Home </a>
                    </div>
                </div>
            </div>
        </div>
    </div>


    <style>
        body {
            font-size: large;

            text-align: center;
            background-color: #eff8ff;


        }

        .main {
            background-color: black;
            color: white;
            opacity: 70%;
            border-radius: 20px;
            margin-top: 10%;
            padding: 30px 20px 40px 20px;

        }

        .go-home {
            color: white;
            border-radius: 20px;
            padding: 20px 20px 40px 20px;

        }
    </style>

    {% endblock body %}

```



UPDATE APP.PY PART 1 

```BASH 

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
# from prediction_service import prediction

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" :
        pass
    else :
        return render_template("index.html")



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


```


RUN 

```BASH 

python app.py

```


UPDATE APP.PY - UPDATE 2 

```BASH 


from flask import Flask, render_template, request, jsonify
import yaml
import os
import json
import joblib
import numpy as np
# from prediction_service import prediction

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)


def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    print(prediction)
    return prediction


def api_response(request) :
    pass



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" :
        try :
            if request.form :
                 data = dict(request.form).values()
                 data = [list(map(float, data))]
                 response = predict(data)
                 return render_template("index.html", response = response)
            elif request.json :
                response = api_response(request)
                return jsonify(response)
                
        
        except Exception as e :
            print(e)
            error = {"error" : "Something Went Wrong !! try again "}
            return render_template("404.html", error=error)
    else :
        return render_template("index.html")



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


```


RUN 

```BASH 

 38  cp saved_models/model.joblib prediction_service/model/
 python app.py

```

IN BROWSER

```BASH 
TYPE  http://localhost:5000

```

ENTER SOME VALUES IN TO THE FIELDS AND CLICK PREDICT


RUN


```BASH 

git add . && git commit -m "wITH OUT VALLIDATION COMMITED APP.PY "
git push origin main

```

```BASH 
COPY schema_in.json from notebooks to prediction_service folder 
```


CREATE PREDICTION.PY UNDER PREDICTION SERVICE 

```bash 
import yaml
import os
import json
import joblib
import numpy as np


params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)



def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    try:
        if 3 <= prediction <= 8:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    
    return True


def form_response(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        response = predict(data)
        return response

def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = predict(data)
            response = {"response": response}
            return response
            
    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except NotInCols as e:
        response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
        return response


    except Exception as e:
        response = {"response": str(e) }
        return response

```


UPDATE APP.PY TO USE PREDICTION.PY AND DO API AND FORM BASED CALLS 


```BASH 

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from prediction_service import prediction


webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = prediction.form_response(dict_req)
                return render_template("index.html", response=response)
            elif request.json:
                response = prediction.api_response(request.json)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


```


RUN 
```BASH 

RERUN :
 python app.py
 

TEST http://localhost:5000 with proper range in the form 
```

```bash 

download postman 
install 

request http://localhost:5000 

CALL POST

BODY :

{"fixed_acidity": 5, 
    "volatile_acidity": 34, 
    "citric_acid": 0.5, 
    "residual_sugar": 10, 
    "chlorides": 0.5, 
    "free_sulfur_dioxide": 3, 
    "total_sulfur_dioxide": 75, 
    "density": 1, 
    "pH": 3, 
    "sulphates": 1, 
    "alcohol": 9
    }


TYPE : RAW
DROPDOWN : JSON


```


```BASH 
git add . && git commit -m "FINAL APP.PY UPDATE  "
git push origin main


```