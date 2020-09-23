# DiffKS: Difference-aware Knowledge Selection

Codes for the paper: **Difference-aware Knowledge Selection for Knowledge-grounded Conversation Generation**

Please cite this repository using the following reference:

```bib
@inproceedings{diffks-zheng-2020,
  title="{D}ifference-aware Knowledge Selection for Knowledge-grounded Conversation Generation",
  author="Zheng, Chujie  and
      Cao, Yunbo  and
      Jiang, Daxin and
      Huang, Minlie",
  booktitle="Findings of EMNLP",
  year="2020"
}
```

## Requirements

See `requirements.txt`.

## Prepare Data

Download the [Wizard of Wikipedia](https://drive.google.com/drive/folders/1eowwYSfJKaDtYgKHZVqh8alNmqP3jv9A?usp=sharing) dataset (downloaded using [Parlai](https://github.com/facebookresearch/ParlAI), please refer to the [Sequential Latent Knowledge Selection](https://github.com/bckim92/sequential-knowledge-transformer) for the download details) and put the files in the folder `./Wizard-of-Wikipedia`, or download the [Holl-E](https://drive.google.com/drive/folders/1xQBRDs5q_2xLOdOpbq7UeAmUM0Ht370A?usp=sharing) dataset and put the files in the folder `./Holl-E`.

For Wizard of Wikipedia (WoW):

```bash
python prepare_wow_data.py
```

For Holl-E:

```bash
python prepare_holl_data.py
```

Besides, download the pretrained [wordvector](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip), unzip the files in `./` and rename the 300-d embedding file as `glove.txt`.

## Training

Our codes now only support single-GPU training, which requires at least 12GB memory.

For Wizard of Wikipedia:

```bash
python run.py \
    --mode train \
    --dataset WizardOfWiki \
    --datapath ./Wizard-of-Wikipedia/prepared_data \
    --wvpath ./ \
    --cuda 0 \
    --droprate 0.5 \
    --disentangle \ # the disentangled model, delete this line if train the fused model
    --hist_len 2 \
    --hist_weights 0.7 0.3 \
    --out_dir ./output \
    --model_dir ./model \
    --cache
```

For Holl-E:

```bash
python run.py \
    --mode train \
    --dataset HollE \
    --datapath ./Holl-E/prepared_data \
    --wvpath ./ \
    --cuda 0 \
    --droprate 0.5 \
    --disentangle \ # the disentangled model, delete this line if train the fused model
    --hist_len 2 \
    --hist_weights 0.7 0.3 \
    --out_dir ./output \
    --model_dir ./model \
    --cache
```

You can modify `run.py` and `myCoTK/dataloader.py` to change more hyperparameters.

## Evaluation

For Wizard of Wikipedia:

```bash
python run.py \
    --mode test \
    --dataset WizardOfWiki \
    --cuda 0 \
    --restore best \
    --disentangle \ # the disentangled model, delete this line if train the fused model
    --hist_len 2 \
    --hist_weights 0.7 0.3 \
    --out_dir ./output \
    --model_dir ./model \
    --cache
```

For Holl-E:

```bash
python run.py \
    --mode test \
    --dataset Holl-E \
    --cuda 0 \
    --restore best \
    --disentangle \ # the disentangled model, delete this line if train the fused model
    --hist_len 2 \
    --hist_weights 0.7 0.3 \
    --out_dir ./output \
    --model_dir ./model \
    --cache
```

