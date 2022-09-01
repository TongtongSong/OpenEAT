#!/usr/bin/env bash
# Copyright 2022 songtongmail@163.com (Tongtong Song)
set -e
set -o pipefail

AnacondaPath=/Work18/2020/songtongtong/anaconda3 # modify yourself
ENVIRONMENT=torch1.9_cuda11.1 # modify yourself
conda activate $ENVIRONMENT

export LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PATH=$PWD/tools:$PWD/openeat:$PWD:$AnacondaPath/envs/$ENVIRONMENT/bin/:$PATH
export LC_ALL=C
export CUDA_VISIBLE_DEVICES="0,1" # modify yourself

corpus=/CDShare3/biaobei_t2/

stage=0
stop_stage=2

data_stage=0
formatdata_stage=1
nj=16

decoding_stage=2
decoding_num_workers=4
# English
# pre_trained=../../pre-trained/gigaspeech_20210728_u2pp_conformer_exp # modify yourself

# Chinese
pre_trained=../../pre-trained/wenetspeech_20220506_u2pp_conformer_exp
train_config=$pre_trained/train_aed.yaml
dict=$pre_trained/words.txt
decode_checkpoint=$pre_trained/final.pt
ctc_weight=0.5
reverse_weight=0.3
decode_dir=exp/decoding_new

mode="attention_rescoring"
if [ ${stage} -le ${data_stage} ] && [ ${stop_stage} -ge ${data_stage} ]; then
    echo "===== stage ${data_stage}: Prepare data ====="
    mkdir -p data
    find $corpus -iname "*.wav" |awk -F '/' '{print($NF" "$0)}' > data/wav.scp
    cat data/wav.scp > data/text
    cat data/wav.scp |awk '{print($1" "$1)}' > data/utt2spk
    cat data/utt2spk |awk '{print($1" "$1)}' > data/spk2utt
    echo "===== stage ${data_stage}: Prepare data Successfully !====="
fi

if [ ${stage} -le ${formatdata_stage} ] && [ ${stop_stage} -ge ${formatdata_stage} ]; then
    echo "===== stage ${formatdata_stage}: Format data ====="
    tools/fix_data_dir.sh data/
    tools/format_data.sh --nj ${nj} \
        --feat-type "wav" --feat data/wav.scp \
        --raw true data > data/format.data
    echo "===== stage ${formatdata_stage}: Format data Successfully !====="
fi


if [ ${stage} -le ${decoding_stage} ] && [ ${stop_stage} -ge ${decoding_stage} ]; then
    echo "===== stage ${decoding_stage}: Decoding ====="
    
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    idx=0
    tmpdir=$(mktemp -d tmp-XXXXX)
    mkdir -p $tmpdir/encoder
    mkdir -p $tmpdir/decoder
    split --additional-suffix .slice -d -n l/$decoding_num_workers data/format.data $tmpdir/tmp_
    for slice in `ls $tmpdir/tmp_*.slice`; do
    {
        name=$(basename $slice)
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
        python3 openeat/bin/recognize.py --gpu $gpu_id \
            --mode $mode \
            --model "wenet" \
            --config $train_config \
            --test_data $slice \
            --checkpoint $decode_checkpoint \
            --beam_size 20 \
            --batch_size 1 \
            --dict $dict \
            --result_file $tmpdir/$name.text \
            --ctc_weight $ctc_weight
    }&
    ((idx+=1))
    if [ $idx -eq $num_gpus ]; then
        idx=0
    fi
    done
    wait
    mkdir -p $decode_dir
    mv $tmpdir/encoder > $decode_dir/
    mv $tmpdir/decoder > $decode_dir/
    cat $tmpdir/*.text |sort -k1 > $decode_dir/text
    cat $tmpdir/recognize.log > $decode_dir/recognize.log
    rm -r $tmpdir
    echo "===== stage ${decoding_stage}: Decoding Successfully !====="
fi