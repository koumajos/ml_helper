# ML helper Package

```
network_time_series/
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.py
├── src/
│   └── ml_helper/
│       ├── __init__.py
│       ├── error_handler.py
│       └── ml_helper.p
└── tests/
```

## Instalation

```shell
cd ml_helper/

python3 setup.py bdist_wheel

cd ..

sudo pip3 install -e ml_helper
```

## Usage

Then you can use in python modules as:

```python
import ml_helper

ml = ml_helper.MLHealper("database.csv")
ml.set_label(labelname="LABEL")
features = [...]
ml.train_test_split(features)
ml.RandomForestClassifier()
print(ml.get_classification_report())
```
