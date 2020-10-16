# Performance Testing

## CTA
test
```
python cta.py --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.9 --solver cta_retire
```

batch
```
parallel --dry-run --joblog ./jobs.log --result ./parallel_out_cta -j 1 'python cta.py --group_id test_cta --trial_id {1} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements {2} --solver {3}' ::: {1..10} ::: 0.8 0.85 0.9 0.95 ::: cta cta_retire
```

## GTA
test
```
python gta.py --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.9 --solver gta_retire
```

batch
```
parallel --dry-run --joblog ./jobs.log --result ./parallel_out_gta -j 1 'python gta.py --group_id test_gta --trial_id {1} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements {2} --solver {3} --n_monte_carlo_trial {4}' ::: {1..10} ::: 0.8 0.85 0.9 0.95 ::: gta gta_retire gta_onetime ::: 100000
```

## ALA
test
```
python ala.py --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.9 --solver ala_qbc --test_with_random True
```

batch
```
parallel --dry-run --joblog ./jobs.log --result ./parallel_out_ala -j 1 'python ala.py --group_id test_ala --trial_id {1} --task_size 10000 --human_crowd_batch_size 500 --quality_requirements {2} --solver {3}' ::: {1..10} ::: 0.8 0.85 0.9 0.95 ::: ala_us ala_qbc
```
