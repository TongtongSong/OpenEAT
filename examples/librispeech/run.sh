#!/usr/bin/env bash
# Copyright 2021 songtongmail@163.com (Tongtong Song)

set -e
set -o pipefail

AnacondaPath=/Work18/2020/songtongtong/anaconda3 # modify yourself
ENVIRONMENT=torch1.9_cuda11.1 # modify yourself
conda activate $ENVIRONMENT

export PYTHONIOENCODING=UTF-8
export PATH=$PWD/tools:$PWD/openeat:$PWD:$AnacondaPath/envs/$ENVIRONMENT/bin/:$PATH
export LC_ALL=C
export CUDA_VISIBLE_DEVICES="3" # modify yourself

corpus=/Work21/2020/songtongtong/data/corpus/Librispeech # modify yourself
stage=0
stop_stage=3

data_stage=-3

dict_stage=-2
vocab_size=1000
bpe_model=data/dict/libri.bpe.${vocab_size}
dict=data/dict/lang_char.txt

formatdata_stage=-1
nj=16

# stage 0: Training
training_stage=0
num_workers=4
train_set=train
dev_set=dev
exp_name=conformer_sort_10000_1_continue # modify yourself
train_config=conf/train.yaml
checkpoint=exp/conformer_sort_10000_1/2.pt

# stage 1: Average Model
avgm_stage=1
start=90
end=99

# stage 2: Decoding
decoding_stage=2
decoding_num_workers=3
recg_set="test_clean test_other" # modify yourself
decode_mode="ctc_greedy_search ctc_prefix_beam_search attention_rescoring attention"
beam_size=10
batch_size=1
ctc_weight=0.5
reverse_weight=0.3
# lm, modify yourself
lm_weight=0
lm=
lm_config=

# stage 3: WER
wer_stage=3

. tools/parse_options.sh || exit 1;

exp_dir=exp/$exp_name
decode_checkpoint=$exp_dir/avg_${start}to${end}.pt

if [ ${stage} -le ${data_stage} ] && [ ${stop_stage} -ge ${data_stage} ]; then
    echo "===== stage ${data_stage}: Prepare data ====="
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/librispeech_data_prep.sh $corpus/$part/ data/$(echo $part | sed s/-/_/g)
        echo "process $part succeeded"
    done
    tools/combine_data.sh data/train data/train_clean_100 data/train_clean_360 data/train_other_500
    tools/combine_data.sh data/dev data/dev_clean data/dev_other
    tools/combine_data.sh data/test data/test_clean data/test_other
    echo "===== stage ${data_stage}: Prepare data Successfully !====="
fi

if [ ${stage} -le ${dict_stage} ] && [ ${stop_stage} -ge ${dict_stage} ]; then
    echo "===== stage ${dict_stage}: Prepare dict ====="
    dict_dir=$(dirname $dict)
    mkdir -p $dict_dir
    echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
    echo "<unk> 1"  >> ${dict}  # <unk> must be 1
    cut -f 2- -d" " data/train/text |tr a-z A-Z > data/dict/input.txt
    tools/spm_train --input=data/dict/input.txt --vocab_size=$vocab_size --model_type='bpe' --model_prefix=${bpe_model} --input_sentence_size=100000000
    tools/spm_encode --model=${bpe_model}.model --output_format=piece < data/dict/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict
    echo "===== stage ${dict_stage}: Prepare dict Successfully !====="
fi

if [ ${stage} -le ${formatdata_stage} ] && [ ${stop_stage} -ge ${formatdata_stage} ]; then
    echo "===== stage ${formatdata_stage}: Format data ====="
    for data in test_clean test_other;do
        tools/fix_data_dir.sh data/$data
        tools/format_data.sh --nj ${nj} \
            --feat-type "wav" --feat data/$data/wav.scp \
            --raw true data/$data > data/$data/format.data
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
          --bpe_model ${bpe_model}.model \
          --train_data data/$train_set/format.data.gpu05 \
          --cv_data data/$dev_set/format.data.gpu05 \
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
                    --bpe_model ${bpe_model}.model \
                    --dict $dict \
                    --result_file $tmpdir/$name.text \
                    --lm_weight $lm_weight \
                    ${lm:+--lm $lm} \
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
            cat  $decode_dir/text |cut -d ' ' -f 1 > $decode_dir/uttid
            cat  $decode_dir/text |cut -d ' ' -f 2- |sed 's| ||g'|sed 's|▁| |g' > $decode_dir/content
            paste -d ' ' $decode_dir/uttid $decode_dir/content > $decode_dir/text.format
            python3 tools/compute-wer.py --char=1 --v=1 \
                   data/$data/text $decode_dir/text.format > $decode_dir/wer
            echo "wer path is $decode_dir/wer"
            cat $decode_dir/wer| grep -E "Overall|Englsih|Other"
        }
        done
    }
    done
    echo "===== stage ${wer_stage} : Compute WER Successfully !====="
fi
