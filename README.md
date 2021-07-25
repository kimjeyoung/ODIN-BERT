# :fire: ODIN-BERT :fire:
This is a reimplementation for ENHANCING THE RELIABILITY OF OUT-OF-DISTRIBUTION IMAGE DETECTION IN NEURAL NETWORKS using BERT.

The codes are based on [official repo (Pytorch)](https://github.com/facebookresearch/odin) and [huggingface](https://huggingface.co/).

Original Paper : [Link](https://arxiv.org/pdf/1706.02690.pdf)

## Installation :coffee:

Training environment : Ubuntu 18.04, python 3.6
```bash
pip3 install torch torchvision torchaudio
pip install scikit-learn
```

Download `bert-base-uncased` checkpoint from [hugginface-ckpt](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin)  
Download `bert-base-uncased` vocab file from [hugginface-vocab](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt)  
Download CLINC OOS intent detection benchmark dataset from [tensorflow-dataset](https://github.com/jereliu/datasets/raw/master/clinc_oos.zip)

The downloaded files' directory should be:

```bash
ODIN-BERT
ㄴckpt
  ㄴbert-base-uncased-pytorch_model.bin
ㄴdataset
  ㄴclinc_oos
    ㄴtrain.csv
    ㄴval.csv
    ㄴtest.csv
    ㄴtest_ood.csv
  ㄴvocab
    ㄴbert-base-uncased-vocab.txt
ㄴmodels
...
```


## Dataset Info :book:

In their paper, the authors conducted OOD experiment for NLP using CLINC OOS intent detection benchmark dataset, the OOS dataset contains data for 150 in-domain services with 150 training
sentences in each domain, and also 1500 natural out-of-domain utterances.
You can download the dataset at [Link](https://github.com/jereliu/datasets/raw/master/clinc_oos.zip).

Original dataset paper, and Github : [Paper Link](https://aclanthology.org/D19-1131/), [Git Link](https://github.com/clinc/oos-eval)

## Run :star2:

#### Train
```bash
python main.py --train_or_test train --task classification --device gpu --gpu 0
```

#### Test

```bash
python main.py --train_or_test test --task classification --device gpu --gpu 0
```

## Results :sparkles:

Results for `ODIN-BERT` on CLINC OOS.  
**NOTE** : Depending on the random seed, the result may be slightly different.

| Version | ACC | AUROC | AUPRC |
| --- | --- | --- | --- |
| Pytorch (batch size = 64) | 96.5 | 0.975 | 0.903 |


## References

[1] https://github.com/facebookresearch/odin  
[2] https://huggingface.co/  
