device=0
split=species

for runseed in 1 2 3 4 5
do
for unsup in contextpred infomax edgepred masking simgrace
do
model_file=${unsup}

python finetune_wagt.py --input_model_file model_gin/${model_file}.pth --split $split --filename gin_${model_file} --device $device --runseed $runseed --gnn_type gin --lr 0.01 --pred_lr 0.01 --tags tags #--store
done

done