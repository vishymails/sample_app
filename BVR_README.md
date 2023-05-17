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


