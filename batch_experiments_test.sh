#!/bin/bash

set -eu

# alias python="/mnt/c/Users/makky/AppData/Local/Programs/Python/Python37/python.exe"
alias python="/usr/bin/python3"

# catch(){
#     if [ "$?" = 1 ]; then
#         curl -X POST --data-urlencode "payload={\"channel\": \"#experiments\", \"username\": \"HACTAP BOT\", \"text\": \":scream: Some error occurred.\", \"icon_emoji\": \":satellite:\"}" https://hooks.slack.com/services/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
#     fi
    
# }

# trap catch ERR

# start_at=`date '+%Y-%m-%d_%H-%M-%S'`
# group_id="deim2020-${start_at}"
group_id="batch_test"

curl -X POST --data-urlencode "payload={\"channel\": \"#experiments\", \"username\": \"HACTAP BOT\", \"text\": \":rocket: Start batch script (group_id: ${group_id})\", \"icon_emoji\": \":satellite:\"}" https://hooks.slack.com/services/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

echo 

# python experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.7 --human_query_strategy random --enable_task_cluster 0
# python experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.8 --human_query_strategy random --enable_task_cluster 0

# python experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.7 --human_query_strategy uncertainty_sampling --enable_task_cluster 0
# python ./hactap/experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.8 --solver al

# python ./hactap/experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.7 --human_query_strategy random --enable_task_cluster 1
python3 ./experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.8 --solver doba
python3 ./experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.8 --solver al --human_query_strategy uncertainty_sampling
python3 ./experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.8 --solver oba

# python experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.7 --human_query_strategy random --enable_task_cluster 1 --enable_quality_guarantee 1
# python experiment.py --group_id ${group_id} --task_size 10000 --human_crowd_batch_size 2000 --quality_requirements 0.8 --human_query_strategy random --enable_task_cluster 1 --enable_quality_guarantee 1

curl -X POST --data-urlencode "payload={\"channel\": \"#experiments\", \"username\": \"HACTAP BOT\", \"text\": \":checkered_flag: Finished batch script (group_id: ${group_id})\", \"icon_emoji\": \":satellite:\"}" https://hooks.slack.com/services/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
