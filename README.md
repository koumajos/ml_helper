# LS Periodogram Method Package

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
from network_time_series.time_series import TimeSeries
from network_time_series.detection import perform_space_detection

id = "<ip>(<port>)-<ip>"
packets = [...]
bytes = [...]
start_time = [...]
end_time = [...]

time_serie = TimeSeries(id, packets, bytes, start_time, end_time)

spaces = perform_space_detection(time_serie)
```
