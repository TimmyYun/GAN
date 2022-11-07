train_path="/workspace/madina_abdrakhmanova/datasets/"
test_path="/workspace/madina_abdrakhmanova/datasets/"

train_list="/workspace/timur_unaspekov/projects/trimodal_person_verification/metadata/train_list_100_percent.txt"
valid_list="/workspace/timur_unaspekov/projects/trimodal_person_verification/metadata/valid_list_v2.txt"
test_list="/workspace/timur_unaspekov/projects/trimodal_person_verification/metadata/test_list_v2.txt"

# test_list="/workspace/timur_unaspekov/projects/trimodal_person_verification/metadata/test_list_vc_v2.txt"

run_file="/workspace/timur_unaspekov/projects/trimodal_person_verification/trainSpeakerNet.py"

train_lists_save_path="/workspace/timur_unaspekov/projects/trimodal_person_verification/metadata/meta_100_percent/train"
eval_lists_save_path="/workspace/timur_unaspekov/projects/trimodal_person_verification/metadata/meta_sf"
# eval_lists_save_path="/workspace/timur_unaspekov/projects/trimodal_person_verification/metadata/meta_vc"

var_eval_frames=400
var_max_frames=200
var_nspeaker=2
var_bsize=1000
var_encoder=SAP
var_fun=angleproto
var_num_images=1

var_nepoch=500
var_gpu=7

cpu_list=224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255

var_modality="wavrgbthr"

var_decay=0

var_seed=3
# initial_model=20
exp=171
save_path="/workspace/timur_unaspekov/meta/results/$var_modality/exp$exp"

# save_path="/workspace/timur_unaspekov/clean_clean/$var_modality/exp$exp"

#  --filters 32 64 128 256 --distributed
taskset --cpu-list $cpu_list python $run_file --model ResNetSE34Multi --modality $var_modality --log_input True \
	--gpu_id $var_gpu --trainfunc angleproto --batch_size $var_bsize \
	--max_epoch $var_nepoch --max_frames $var_max_frames --eval_frames $var_eval_frames  \
	--weight_decay $var_decay --nPerSpeaker $var_nspeaker \
	--save_path $save_path --train_path $train_path --test_path $test_path \
	--train_list $train_list --test_list $valid_list --seed $var_seed \
	--train_lists_save_path $train_lists_save_path --eval_lists_save_path $eval_lists_save_path \
	--test_interval 10 --nOut 512

# taskset --cpu-list $cpu_list python $run_file --model ResNetSE34Multi --modality $var_modality --log_input True \
# 	--gpu_id $var_gpu --trainfunc angleproto --batch_size $var_bsize \
# 	--max_epoch $var_nepoch --max_frames $var_max_frames --eval_frames $var_eval_frames  \
# 	--weight_decay $var_decay --nPerSpeaker $var_nspeaker \
# 	--save_path $save_path --train_path $train_path --test_path $test_path \
# 	--train_list $train_list --test_list $valid_list --seed $var_seed \
# 	--train_lists_save_path $train_lists_save_path --eval_lists_save_path $eval_lists_save_path \
# 	--test_interval 1  --mixedprec --distributed --port 8887

taskset --cpu-list $cpu_list python $run_file --model ResNetSE34Multi --modality $var_modality --log_input True \
	--gpu_id $var_gpu --trainfunc angleproto --batch_size $var_bsize \
	--max_epoch $var_nepoch --max_frames $var_max_frames --eval_frames $var_eval_frames  \
	--weight_decay $var_decay --nPerSpeaker $var_nspeaker \
	--save_path $save_path --train_path $train_path --test_path $test_path \
	--train_list $train_list  --test_list $test_list \
	--train_lists_save_path $train_lists_save_path --eval_lists_save_path $eval_lists_save_path \
	--eval True --valid_model True

taskset --cpu-list $cpu_list python $run_file --model ResNetSE34Multi --modality $var_modality --log_input True \
	--gpu_id $var_gpu --trainfunc angleproto --batch_size $var_bsize  \
	--max_epoch $var_nepoch --max_frames $var_max_frames --eval_frames $var_eval_frames  \
	--weight_decay $var_decay --nPerSpeaker $var_nspeaker \
	--save_path $save_path --train_path $train_path --test_path $test_path \
	--train_list $train_list  --test_list $test_list \
	--train_lists_save_path $train_lists_save_path --eval_lists_save_path $eval_lists_save_path \
	--eval True