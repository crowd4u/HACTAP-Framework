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

## Sync this repo to a remote server

```
rsync -avzu ../HACTAP-Framework makky@mlab-gpu:~/Projects/
```

```
rsync -avz ../HACTAP-Framework [user@host]:[path/to/deploy]
```
