# Experiment with MIND dataset

## download dataset
```
# download original images
aws s3 cp s3://projects.crowd4u.org/mind/flood_image/ ./original_images --recursive --profile=mlab
python prepare_dataset.py
```

## solo
```
python experiment.py --human_crowd_batch_size 12000 --quality_requirements 0.9 --solver gta
```

## batch

```
parallel --joblog ./jobs.log --result ./parallel_out -j 1 'python experiment.py --group_id demo --trial_id {1} --human_crowd_batch_size 30000 --quality_requirements {2} --solver {3}' ::: 1 ::: 0.85 0.9 0.95 ::: gta al
```


## memo
```
confusion_matrix 
[[31382    42   159]
 [  489   147     6]
 [  132     0   939]]
classification_report               precision    recall  f1-score   support

           0       0.98      0.99      0.99     31583
           1       0.78      0.23      0.35       642
           2       0.85      0.88      0.86      1071

    accuracy                           0.98     33296
   macro avg       0.87      0.70      0.73     33296
weighted avg       0.97      0.98      0.97     33296
```
