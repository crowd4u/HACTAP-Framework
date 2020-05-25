import optuna

import hactap.solver.

def objective(trial):
    x = trial.suggest_uniform('x', -100, 100)
    y = trial.suggest_int('y', -100, 100)
    return x ** 2 + y ** 2

search_space = {
    'solver': [''],
    'y': [-99, 0, 99]
}

study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=3*3)
