#!/usr/bin/env bash
# Copyright 2021 songtongmail@163.com (Tongtong Song)

set -e
set -o pipefail

. ./path.sh # kaldi

AnacondaPath=/Work18/2020/songtongtong/anaconda3 # modify yourself
ENVIRONMENT=torch1.9_cuda11.1 # modify yourself
conda activate $ENVIRONMENT

export LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PATH=$PWD/tools:$PWD/openeat:$PWD:$AnacondaPath/envs/$ENVIRONMENT/bin/:$PATH
export LC_ALL=C
export CUDA_VISIBLE_DEVICES="-1" # modify yourself

corpus=/Work21/2020/songtongtong/data/corpus/AISHELL-1 # modify yourself
nj=16

stage=0
stop_stage=0

data_stage=-4

dict_stage=-3
discarded_threshold=5 # Discard characters that appear less than 5 times
dict=data/dict/lang_char.txt

feat_stage=-2
speed_perturb=true

formatdata_stage=-1

# stage 0: Training
training_stage=0
num_workers=1
train_set=train
dev_set=dev

train_config=conf/train.yaml
checkpoint=

# using wenet pre-trained model
pre_trained=../../pre-trained/aishell2_20210618_u2pp_conformer_exp
# bpe_model=../../pre-trained/librispeech_20210610_u2pp_conformer_exp/train_960_unigram5000.model
dict=$pre_trained/words.txt
checkpoint=$pre_trained/final.pt

exp_name=test # modify yourself


# stage 1: Average Model
avgm_stage=1
start=45
end=49

# stage 2: Decoding
decoding_stage=2
decoding_num_workers=2
recg_set="test" # modify yourself
decode_mode="ctc_greedy_search ctc_prefix_beam_search attention_rescoring attention"
beam_size=10
batch_size=1
ctc_weight=0.5
reverse_weight=0.3
# lm, modify yourself
lm_weight=
lm=
lm_config=

# stage 3: WER
wer_stage=3

. tools/parse_options.sh || exit 1;

exp_dir=exp/$exp_name
decode_checkpoint=$exp_dir/avg_${start}to${end}.pt

if [ ${stage} -le ${data_stage} ] && [ ${stop_stage} -ge ${data_stage} ]; then
    echo "===== stage ${data_stage}: Prepare data ====="
    local/aishell_data_prep.sh $corpus
    echo "===== stage ${data_stage}: Prepare data Successfully !====="
fi

if [ ${stage} -le ${dict_stage} ] && [ ${stop_stage} -ge ${dict_stage} ]; then
    echo "===== stage ${dict_stage}: Prepare dict ====="
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
    echo "<unk> 1"  >> ${dict}  # <unk> must be 1
    python3 openeat/bin/text2token.py data/train/text | cut -f 2- -d" " | tr " " "\n" | \
        sort | uniq -c | awk -v dt=$discarded_threshold '{if($1>=dt)print($2)}' | \
        grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict
    echo "===== stage ${dict_stage}: Prepare dict Successfully !====="
fi

if [ ${stage} -le ${feat_stage} ] && [ ${stop_stage} -ge ${feat_stage} ]; then
    echo "===== stage ${feat_stage}: Prepare Feat ====="
    utils/perturb_data_dir_speed.sh 0.9 data/train data/train_sp0.9
    utils/perturb_data_dir_speed.sh 1.1 data/train data/train_sp1.1
    utils/combine_data.sh data/train_sp data/train data/train_sp0.9 data/train_sp1.1
    for x in ${train_set} dev test; do
        steps/make_fbank.sh --nj $nj \
            --write_utt2num_frames true --fbank_config $fbank_conf --compress true data/$x
    done
    echo "===== stage ${feat_stage}: Prepare Feat Successfully !====="
fi

if [ ${stage} -le ${formatdata_stage} ] && [ ${stop_stage} -ge ${formatdata_stage} ]; then
    echo "===== stage ${formatdata_stage}: Format data ====="
    for data in test dev ${train_set};do
        tools/fix_data_dir.sh data/$data
        tools/format_data.sh --nj ${nj} \
            --feat-type "wav" --feat data/$data/wav.scp \
            --raw true data/$data > data/$data/format.data
        # kaldi feature
        # tools/format_data.sh --nj ${nj} \
        #     --feat data/$data/feats.scp data/$data > data/$data/format.data
    done
    echo "===== stage ${formatdata_stage}: Format data Successfully !====="
fi

if [ ${stage} -le ${training_stage} ] && [ ${stop_stage} -ge ${training_stage} ]; then
    echo "===== stage ${training_stage}: Training ====="
    mkdir -p $exp_dir
    [ -f $exp_dir/train.log ] && rm $exp_dir/train.log
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    python3 openeat/bin/train.py \
          --ngpus $num_gpus \
          --num_workers $num_workers \
          --config $train_config \
          --dict $dict \
          --train_data data/$train_set/format.data \
          --cv_data data/$dev_set/format.data \
          --exp_dir $exp_dir \
          ${checkpoint:+--checkpoint $checkpoint}
    echo "===== stage ${training_stage}: Training Successfully !====="
fi

if [ ${stage} -le ${avgm_stage} ] && [ ${stop_stage} -ge ${avgm_stage} ]; then
    echo "===== stage ${avgm_stage}: Average Model ====="
    echo "average models and final checkpoint is $decode_checkpoint"
    python3 openeat/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $exp_dir  \
        --start $start \
        --end $end
    echo "=====  stage ${avgm_stage}: Average Model Successfully !====="
fi


if [ ${stage} -le ${decoding_stage} ] && [ ${stop_stage} -ge ${decoding_stage} ]; then
    echo "===== stage ${decoding_stage}: Decoding ====="
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    for data in $recg_set; do
    {
        for mode in $decode_mode;do
        {
            idx=0
            echo $data $mode
            decode_dir=$exp_dir/$data/${mode}/${start}to${end}/beam${beam_size}_batch${batch_size}_ctc${ctc_weight}_reverse${reverse_weight}_lm${lm_weight}
            mkdir -p $decode_dir
            [ -f $decode_dir/recognize.log ] && rm $decode_dir/recognize.log
            tmpdir=$(mktemp -d tmp-XXXXX)
            split --additional-suffix .slice -d -n l/$decoding_num_workers data/$data/format.data $tmpdir/tmp_
            for slice in `ls $tmpdir/tmp_*.slice`; do
            {
                name=$(basename $slice)
                gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
                # gpu_id -1 for cpu
                python3 openeat/bin/recognize.py \
                    --gpu $gpu_id \
                    --mode $mode \
                    --config $exp_dir/train.yaml \
                    --test_data  $slice \
                    --checkpoint $decode_checkpoint \
                    --beam_size $beam_size \
                    --batch_size $batch_size \
                    --dict $dict \
                    --result_file $tmpdir/$name.text \
                    ${lm:+--lm $lm} \
                    ${lm_weight:+--lm_weight $lm_weight} \
                    ${lm_config:+--lm_config $lm_config}
            }&
            ((idx+=1))
            if [ $idx -eq $num_gpus ]; then
                idx=0
            fi
            done
            wait
            cat $tmpdir/*.text > $decode_dir/text
            cat $tmpdir/recognize.log > $decode_dir/recognize.log
            rm -r $tmpdir
        }
        done
    }
    done
    echo "===== stage ${decoding_stage}: Decoding Successfully !====="
fi

if [ ${stage} -le ${wer_stage} ] && [ ${stop_stage} -ge ${wer_stage} ]; then
    echo "===== stage ${wer_stage} : Compute WER ====="
    for data in $recg_set;do
    {
        for mode in $decode_mode;do
        {
            echo $data $mode
            decode_dir=$exp_dir/$data/${mode}/${start}to${end}/beam${beam_size}_batch${batch_size}_ctc${ctc_weight}_reverse${reverse_weight}_lm${lm_weight}
            python3 tools/compute-wer.py --char=1 --v=1 \
                   data/$data/text $decode_dir/text > $decode_dir/wer
            echo "wer path is $decode_dir/wer"
            cat $decode_dir/wer| grep -E "Overall|Mandarin|Other"
        }
        done
    }
    done
    echo "===== stage ${wer_stage} : Compute WER Successfully !====="
fi
