n_threads=12
script_to_run=./register_single_step.sh
OUTPUT_DIR="/home/ciganche/Documents/AI_challange/data/preprocessed/generated/"

i=1
while read line
do
  echo $i". pair"
  i=$((i+1))

  disease=$(echo "$line" | cut -f 6)
  condition=$(echo "$line" | cut -f 5)
  healthy=$(echo "$line" | cut -f 12)
  healthy_segmented=$(echo "$line" | cut -f 13)

  if [[ $condition = "HD" ]]
  then
    run_mode=2
  else
    run_mode=1
  fi

  ${script_to_run} ${disease} ${healthy} ${healthy_segmented} ${OUTPUT_DIR} ${run_mode} ${n_threads}

done <<< $(tail -n +2 generation_pairs.tsv) # skip header
