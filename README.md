# HACTAP-Framework
A research framework for Human+AI Crowd Task Assignment Problem

## Experiments
- experiments/demo_mnist (Experiment Using a Benchmark Dataset)
- experiments/mind (Experiment using a Real-World Dataset)

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
