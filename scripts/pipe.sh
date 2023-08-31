dataset=$1
method=$2
tasks=$3
ValDatasets=$4
ADcmd=$5
FTcmd=$6
alias=$7

ValDatasets=${ValDatasets:-$dataset}
ADcmd=${ADcmd:-}
FTcmd=${FTcmd:-}
alias=${alias:-vit-b-16}

echo "method: $method"
echo "dataset: $dataset"
echo "tasks: $tasks"
echo "datasets for evaluation: $ValDatasets"


if [[ "${tasks[@]}" =~ "finetune " || 
      ("${tasks[@]}" =~ " finetune" && ! "${tasks[@]}" =~ "finetune_") ||
      ("${tasks[@]}" =~ "finetune" && ! "${tasks[@]}" =~ "finetune_") 
      ]]; then
    python3 finetune.py --pickle --dataset ${dataset} --method ${method} $FTcmd
    src=output/finetune/vit-b-16_${dataset}_${method}
    trg=${src}/${dataset}_subsets/100%
    mkdir -p ${trg}
    cp ${src}/log.txt ${trg}/log.txt
fi


if [[ "${tasks[@]}" =~ "finetune_fewshot" ]]; then
    bash scripts/ft_fewshot.sh ${dataset} ${method} $FTcmd
fi


if [[ "${tasks[@]}" =~ "adapt " || 
      ("${tasks[@]}" =~ " adapt" && ! "${tasks[@]}" =~ "adapt_") ||
      ("${tasks[@]}" =~ "adapt" && ! "${tasks[@]}" =~ "adapt_") 
      ]]; then
    python3 adapt.py --pickle --dataset ${dataset} --method ${method} $ADcmd
fi


if [[ "${tasks[@]}" =~ "adapt_zeroshot" ]]; then
    for ValDataset in $ValDatasets
    do
        python3 loop_evaluate.py \
                --pickle \
                --from_pretrained output/adapt/${alias}_${dataset}_${method} \
                --dataset ${ValDataset} \
                --folder ${ValDataset}_subsets/0% \
                $FTcmd
    done
fi


if [[ "${tasks[@]}" =~ "adapt_fewshot" ]]; then
    for ValDataset in $ValDatasets
    do
        bash scripts/ablate_fewshot.sh output/adapt/${alias}_${dataset}_${method} ${ValDataset} $FTcmd
    done
fi


if [[ "${tasks[@]}" =~ "adapt_full" ]]; then
    for ValDataset in $ValDatasets
    do
        python3 finetune.py \
                --clip_arch ViT-B/16 \
                --from_pretrained output/adapt/${alias}_${dataset}_${method} \
                --folder ${ValDataset}_subsets/100% \
                --ckpt_name ${ValDataset}_best.pth \
                --dataset ${ValDataset} \
                --pickle \
                $FTcmd
    done
fi
