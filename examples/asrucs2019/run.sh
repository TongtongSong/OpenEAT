#!/usr/bin/env bash
set -e
set -o pipefail
AnacondaPath=/Work18/2020/songtongtong/anaconda3
ENVIRONMENT=torch1.9_cuda11.1
conda activate $ENVIRONMENT

export PYTHONIOENCODING=UTF-8
export PATH=$PWD/tools:$PWD/openeat:$PWD:$AnacondaPath/envs/$ENVIRONMENT/bin/:$PATH
export LC_ALL=C
export CUDA_VISIBLE_DEVICES="-1"

corpus=/Work21/2020/songtongtong/data/corpus/ASRUCS2019
librispeech=/Work21/2020/songtongtong/data/corpus/Librispeech
data_suffix=""

stage=0
stop_stage=0

data_stage=-3

formatdata_stage=-2
nj=16

subset_stage=-1

# stage 0: Training
training_stage=0
num_workers=1

# using wenet pre-trained model
model="wenet"
pre_trained=../../pre-trained/librispeech_20210610_u2pp_conformer_exp
bpe_model=../../pre-trained/gigaspeech_20210728_u2pp_conformer_exp/train_xl_unigram5000.model
train_config=$pre_trained/train_aed.yaml
dict=$pre_trained/words.txt
checkpoint=$pre_trained/final.pt

# training your model
# cn_dict=local/lang_char.cn.5.txt
# en_dict=local/lang_char.en.1000.txt
# cs_dict=local/lang_char.cn.5-en.1000.txt
# data_type="cs"
# dict=$(eval echo \$${data_type}_dict)
# bpe_model=local/libri.bpe.1000.model
# train_config=conf/train.yaml
# checkpoint=

train_set=train_cs
dev_set=dev_cs
exp_name=test # modify yourself

# stage 1: Average Model
avgm_stage=1
start=45
end=49

# stage 2: Decoding
decoding_stage=2
decoding_num_workers=9
recg_set="dev_cs train_cs" # modify yourself
decode_mode="ctc_greedy_search"
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
    local/asrucs2019_data_prep.sh $corpus
    tools/subset_data_dir_tr_cv.sh --cv-spk-percent 4 data/all_cn data/train_cn data/tmp_cn
    tools/subset_data_dir_tr_cv.sh --cv-spk-percent 50 data/tmp_cn data/dev_cn data/test_cn
    rm -r data/tmp_cn data/all_cn

    # use librispeech as English corpus
    for part in dev-clean test-clean train-clean-100 train-clean-360; do
        # use underscore-separated names in data directories.
        local/librispeech_data_prep.sh $librispeech/$part/ data/$(echo $part | sed s/-/_/g)
        echo "process $part succeeded"
    done
    tools/combine_data.sh data/train_en data/train_clean_100 data/train_clean_360 && \
    rm -r data/train_clean_100 data/train_clean_360 || exit 1;
    mv data/test_clean data/test_en
    mv data/dev_clean data/dev_en
    echo "===== stage ${data_stage}: Prepare data Successfully !====="
fi

if [ ${stage} -le ${formatdata_stage} ] && [ ${stop_stage} -ge ${formatdata_stage} ]; then
    echo "===== stage ${formatdata_stage}: Format data ====="
    for data_type in cs cn en; do
        for data_set in test dev train;do
            data=${data_set}_${data_type}
            echo $data
            tools/fix_data_dir.sh data/$data
            tools/format_data.sh --nj ${nj} --feat-type "wav" \
                --feat data/$data/wav.scp \
                --raw true data/$data > data/$data/format.data
        done
    done
    mkdir -p data/mixture/cn_en_cs
    cat data/train_cn/format.data data/train_en/format.data data/train_cs/format.data > data/mixture/cn_en_cs/format.data
    echo "===== stage ${formatdata_stage}: Format data Successfully !====="
fi

if [ ${stage} -le ${training_stage} ] && [ ${stop_stage} -ge ${training_stage} ]; then
    echo "===== stage ${training_stage}: Training ====="
    # CN, EN, CS
    mkdir -p $exp_dir
    [ -f $exp_dir/train.log ] && rm $exp_dir/train.log
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    python3 openeat/bin/train.py \
          --ngpus $num_gpus \
          --model $model \
          --num_workers $num_workers \
          --config $train_config \
          --dict $dict \
          --train_data data/$train_set/format.data${data_suffix} \
          --cv_data data/$dev_set/format.data${data_suffix} \
          --exp_dir $exp_dir \
          ${checkpoint:+--checkpoint $checkpoint} \
          ${bpe_model:+--bpe_model $bpe_model}
    echo "===== stage ${training_stage}: Training Successfully !====="
fi

if [ ${stage} -le ${avgm_stage} ] && [ ${stop_stage} -ge ${avgm_stage} ]; then
    echo "===== stage ${avgm_stage}: Average Model ====="
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
            decode_dir=$exp_dir/$data/test_${mode}_beam${beam_size}_batch${batch_size}_ctc${ctc_weight}_reverse${reverse_weight}_lm${lm_weight}
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
                    --model $model \
                    --gpu $gpu_id \
                    --mode $mode \
                    --config $exp_dir/train.yaml \
                    --test_data  $slice \
                    --checkpoint $decode_checkpoint \
                    --beam_size $beam_size \
                    --batch_size $batch_size \
                    --dict $dict \
                    --result_file $tmpdir/$name.text \
                    --ctc_weight $ctc_weight \
                    --reverse_weight $reverse_weight \
                    ${bpe_model:+--bpe_model $bpe_model} \
                    ${lm_weight:+--lm_weight $lm_weight} \
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
            decode_dir=$exp_dir/$data/test_${mode}_beam${beam_size}_batch${batch_size}_ctc${ctc_weight}_reverse${reverse_weight}_lm${lm_weight}
            cat  $decode_dir/text |cut -d ' ' -f 1 > $decode_dir/uttid
            cat  $decode_dir/text |cut -d ' ' -f 2- |sed 's| ||g'|sed 's|▁| |g' > $decode_dir/content
            paste -d ' ' $decode_dir/uttid $decode_dir/content > $decode_dir/text_format
            python3 tools/compute-wer.py --char=1 --v=1 \
                   data/$data/text $decode_dir/text_format > $decode_dir/wer
            echo "wer path is $decode_dir/wer"
            cat $decode_dir/wer| grep -E "Overall|Mandarin|English|Other"
        }
        done
    }
    done
    echo "===== stage ${wer_stage} : Compute WER Successfully !====="
fi
