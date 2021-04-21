export GLUE_DIR=/sailhome/mhahn/scr/PRETRAINED/CACHED/
export TASK_NAME=$1
rm /sailhome/mhahn/scr/PRETRAINED/CACHED/*/cached_*_Roberta* 
rm -r /tmp/$TASK_NAME
~/python-py37-mhahn run_textclas_other_finetuned.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
rm /sailhome/mhahn/scr/PRETRAINED/CACHED/*/cached_*_Roberta* 
rm -r /tmp/$1

