# HACTAP-Framework
Algorithms for Human+AI Crowd Task Assignment Problem

## Setup
```
pip install -r requirements.txt
pip install -e .
```

## Run tests
```
python -m unittest
# or
python -m unittest tests.test_utils
```

## Check coverage
```
coverage run -m unittest
coverage report -m
```

## Type checking
```
mypy hactap
```

## lint
```
flake8 .
```
