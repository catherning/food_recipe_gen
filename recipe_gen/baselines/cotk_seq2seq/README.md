[![Main Repo](https://img.shields.io/badge/Main_project-cotk-blue.svg?logo=github)](https://github.com/thu-coai/cotk)
[![This Repo](https://img.shields.io/badge/Model_repo-pytorch--seq2seq-blue.svg?logo=github)](https://github.com/thu-coai/seq2seq-pytorch)
[![Coverage Status](https://coveralls.io/repos/github/thu-coai/seq2seq-pytorch/badge.svg?branch=master)](https://coveralls.io/github/thu-coai/seq2seq-pytorch?branch=master)
[![Build Status](https://travis-ci.com/thu-coai/seq2seq-pytorch.svg?branch=master)](https://travis-ci.com/thu-coai/seq2seq-pytorch)

# Seq2Seq (PyTorch)

Seq2seq with attention mechanism is a basic model for single turn dialog. In addition, batch normalization and dropout has been applied. You can also choose beamsearch, greedy, random sample, random sample from top-k when decoding.

You can refer to the following paper for details:

Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In *Advances in neural information processing systems* (pp. 3104-3112).

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In *International Conference on Learning Representation*.

## Require Packages

* **python3**
* cotk
* pytorch == 1.0.0
* tensorboardX >= 1.4

## Quick Start

* Using ``cotk download thu-coai/seq2seq-pytorch/master`` to download codes.
* Execute ``python run.py`` to train the model.
  * The default dataset is ``OpenSubtitles``. You can use ``--dataset`` to specify other ``dataloader`` class and ``--dataid`` to specify other data path (can be a local path, a url or a resources id). For example: ``--dataset OpenSubtitles --dataid resources://OpenSubtitles``
  * It doesn't use pretrained word vector by default setting. You can use ``--wvclass`` to specify ``wordvector`` class and ``--wvpath`` to specify pretrained word embeddings. For example: ``--wvclass gloves``. For example: ``--dataset Glove --dataid resources://Glove300``
  * If you don't have GPUs, you can add `--cpu` for switching to CPU, but it may cost very long time for either training or test.
* You can view training process by tensorboard, the log is at `./tensorboard`.
  * For example, ``tensorboard --logdir=./tensorboard``. (You have to install tensorboard first.)
* After training, execute  ``python run.py --mode test --restore best`` for test.
  * You can use ``--restore filename`` to specify checkpoints files, which are in ``./model``. For example: ``--restore pretrained-opensubtitles`` for loading ``./model/pretrained-opensubtitles.model``
  * ``--restore last`` means last checkpoint, ``--restore best`` means best checkpoints on dev.
  * ``--restore NAME_last`` means last checkpoint with model named NAME. The same as``--restore NAME_best``.
* Find results at ``./output``.

## Arguments

```none
    usage: run.py [-h] [--name NAME] [--restore RESTORE] [--mode MODE] [--lr LR]
                  [--eh_size EH_SIZE] [--dh_size DH_SIZE] [--droprate DROPRATE]
                  [--batchnorm] [--decode_mode {max,sample,gumbel,samplek,beam}]
                  [--top_k TOP_K] [--length_penalty LENGTH_PENALTY]
                  [--dataset DATASET] [--dataid DATAID] [--epoch EPOCH]
                  [--batch_per_epoch BATCH_PER_EPOCH] [--wvclass WVCLASS]
                  [--wvid WVID] [--out_dir OUT_DIR] [--log_dir LOG_DIR]
                  [--model_dir MODEL_DIR] [--cache_dir CACHE_DIR] [--cpu]
                  [--debug] [--cache] [--seed SEED]

    A seq2seq model with GRU encoder and decoder. Attention, beamsearch, dropout
    and batchnorm is supported.

    optional arguments:
      -h, --help            show this help message and exit
      --name NAME           The name of your model, used for tensorboard, etc.
                            Default: runXXXXXX_XXXXXX (initialized by current
                            time)
      --restore RESTORE     Checkpoints name to load. "NAME_last" for the last
                            checkpoint of model named NAME. "NAME_best" means the
                            best checkpoint. You can also use "last" and "best",
                            by default use last model you run. Attention:
                            "NAME_last" and "NAME_best" are not guaranteed to work
                            when 2 models with same name run in the same time.
                            "last" and "best" are not guaranteed to work when 2
                            models run in the same time. Default: None (don't load
                            anything)
      --mode MODE           "train" or "test". Default: train
      --lr LR               Learning rate. Default: 0.001
      --eh_size EH_SIZE     Size of encoder GRU
      --dh_size DH_SIZE     Size of decoder GRU
      --droprate DROPRATE   The probability to be zerod in dropout. 0 indicates
                            for don't use dropout
      --batchnorm           Use bathnorm
      --decode_mode {max,sample,gumbel,samplek,beam}
                            The decode strategy when freerun. Choices: max,
                            sample, gumbel(=sample), samplek(sample from topk),
                            beam(beamsearch). Default: beam
      --top_k TOP_K         The top_k when decode_mode == "beam" or "samplek"
      --length_penalty LENGTH_PENALTY
                            The beamsearch penalty for short sentences. The
                            penalty will get larger when this becomes smaller.
      --dataset DATASET     Dataloader class. Default: OpenSubtitles
      --dataid DATAID       Resource id for data set. It can be a resource name or
                            a local path. Default: resources://OpenSubtitles
      --epoch EPOCH         Epoch for training. Default: 100
      --batch_per_epoch BATCH_PER_EPOCH
                            Batches per epoch. Default: 1500
      --wvclass WVCLASS     Wordvector class, none for not using pretrained
                            wordvec. Default: Glove
      --wvid WVID           Resource id for pretrained wordvector. Default:
                            resources://Glove300d
      --out_dir OUT_DIR     Output directory for test output. Default: ./output
      --log_dir LOG_DIR     Log directory for tensorboard. Default: ./tensorboard
      --model_dir MODEL_DIR
                            Checkpoints directory for model. Default: ./model
      --cache_dir CACHE_DIR
                            Checkpoints directory for cache. Default: ./cache
      --cpu                 Use cpu.
      --debug               Enter debug mode (using ptvsd).
      --cache               Use cache for speeding up load data and wordvec. (It
                            may cause problems when you switch dataset.)
      --seed SEED           Specify random seed. Default: 0
```

## TensorBoard Example

Execute ``tensorboard --logdir=./tensorboard``, you will see the plot in tensorboard pages:

![tensorboard_plot_example](./images/tensorboard_plot_example.png)

Following plot are shown in this model:

* gen/loss (``gen`` means training process)

* gen/perplexity (``=exp(gen/word_loss)``)

* gen/word_loss (``=gen/loss`` in this model)

* dev/loss
* dev/perplexity_avg_on_batch
* test/loss
* test/perplexity_avg_on_batch

And text output:

![tensorboard_plot_example](./images/tensorboard_text_example.png)

Following text are shown in this model:

* args
* dev/show_str%d (``%d`` is according to ``args.show_sample`` in ``run.py``)

## Case Study of Model Results

Execute ``python run.py --mode test --restore best``

The following output will be in `./output/[name]_[dev|test].txt`:

```none
perplexity:     48.194050
bleu:    0.320098
post:   my name is josie .
resp:   <unk> <unk> , pennsylvania , the <unk> state .
gen:    i' m a teacher .
post:   i put premium gasoline in her .
resp:   josie , i told you .
gen:    i don' t know .
post:   josie , dont hang up
resp:   they do it to aii the new kids .
gen:    aii right , you guys , you know what ?
post:   about playing a part .
resp:   and thats the theme of as you like it .
gen:    i don' t know .
......
```

## Experiment

### Subset Experiment

Based on `OpenSubtitles_small` (a smaller version of `OpenSubtitles`), we did the following experiments.

| encoder | decoder | batchnorm | learning rate | droprate | dev perplexity | test perplexity |
| :-----: | :-----: | :-------: | :-----------: | :------: | :------------: | :-------------: |
|   175   |   175   |    no     |    0.0001     |   0.2    |     88.971     |     94.698      |
| **128** | **128** |    no     |    0.0001     |  **0**   |     95.207     |     100.676     |
| **128** | **128** |    no     |    0.0001     |   0.2    |     93.559     |     99.287      |
| **128** | **128** |  **yes**  |    0.0001     |  **0**   |    105.649     |     112.818     |
| **128** | **128** |  **yes**  |    0.0001     |   0.2    |     95.894     |     102.243     |
| **150** | **150** |    no     |    0.0001     |   0.2    |     90.153     |     95.072      |
| **256** | **256** |    no     |    0.0001     |   0.2    |     89.374     |     94.272      |
|   175   |   175   |  **yes**  |    0.0001     |   0.2    |     97.704     |     102.851     |
|   175   |   175   |    no     |    0.0001     | **0.1**  |     90.149     |     95.310      |
|   175   |   175   |    no     |    0.0001     | **0.3**  |     89.750     |     95.042      |
|   175   |   175   |    no     |  **0.0005**   |   0.2    |   **88.688**   |   **93.421**    |

The following experiments are based on the parameters of the first experiment.

To reproduce the experiment, run the following command to train the model:

```bash
python run.py --dataset=OpenSubtitles --dataid resources://OpenSubtitles_small --eh_size 175 --dh_size 175 --droprate 0.2 --epoch 35 --lr 0.0001
```

Based on the best parameters,  we did another five experiments in order to analyze the model's performance totally.

|        seed        | dev perplexity | test perplexity | dev bleu(x1e-3) | test bleu(x1e-3) |
| :----------------: | :------------: | :-------------: | :------------: | :-------------: |
|        7913        |  89.123   |   93.898   |  3.24  |  2.82  |
|        1640        |  88.224   |   94.019   |  4.51  |  4.13  |
|        3739        |  87.969   |   92.730   |  2.37  |  2.15  |
|        972         |  88.083   |   93.515   |  2.99  |  2.34  |
|        3594        |  87.933   |   94.258   |  4.22  |  3.44  |
|      Average       | 88.266 ± 0.440 | 93.684 ± 0.534 |3.47 ± 7.92|2.98 ± 7.29|

Run the following command to test the trained model:

```bash
python run.py --dataset=OpenSubtitles --dataid resources://OpenSubtitles_small --eh_size 175 --dh_size 175 --decode_mode samplek --droprate 0.2 --epoch 35 --mode test --restore [model name] --seed [seed value]
```

### Fullset experiment

Based on the result of subset experiment, we did the following experiment on `OpenSubtitles`.

|  Parameters   | Value  |
| :-----------: | :----: |
|    encoder    |  256   |
|    decoder    |  256   |
|   batchnorm   |   no   |
| learning rate | 0.0003 |
|   droprate    |  0.2   |
|     epoch     |  200   |
|     seed      |   0    |

| decode mode |dev perplexity | test perplexity | dev bleu(x1e-3) | test bleu(x1e-3) |
|:--: | :------------: | :-------------: | :------: | :-------: |
| **samplek** |   41.457   |   43.868   | 3.24  |  2.82  |
| **beam** |   41.457   |   43.868   | 11.5 |  10.4  |

To reproduce the experiment, run the following command to train the model:

```bash
python run.py --dataset=OpenSubtitles --dataid resources://OpenSubtitles --eh_size 256 --dh_size 256 --droprate 0.2 --lr 0.0003 --epoch 200
```

Run the following command to test the trained model:

```bash
python run.py --dataset=OpenSubtitles --dataid resources://OpenSubtitles --eh_size 256 --dh_size 256 --decode_mode [decode mode] --droprate 0.2 --lr 0.0003 --epoch 200 --mode test --restore best
```

## Performance

|               | Perplexity | BLEU  |
| ------------- | ---------- | ----- |
| OpenSubtitles | 43.868     | 0.0115 |

## Author

[HUANG Fei](https://github.com/hzhwcmhf)
