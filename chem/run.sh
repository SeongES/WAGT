tags=tmp

### GIN fine-tuning
device=0
split=scaffold
lr=0.01
pred_lr=0.01
gtot_weight=0.001 
tags=gtot_org
for dataset in bbbp tox21 toxcast sider clintox muv hiv bace 
do
for unsup in infomax masking contextpred simgrace
do
for runseed in 1 2 3 4 5 
do
model_file=${unsup}
python finetune_wagt.py --input_model_file model_gin/${model_file}.pth --filename ${dataset}/gin_${model_file} --runseed $runseed --gnn_type gin --dataset $dataset --tags $tags --eval_train 0 --lr $lr --pred_lr $pred_lr --gtot_weight $gtot_weight --tags $tags #--store
done
done
done
