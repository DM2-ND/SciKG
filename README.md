# MIMO
semi-supervised MIMO, a construction of Scientific Knowledge  Graph 

## ENV

torch/0.4.0, python/2.7.14

The versions of python in training and predicting shoud be the same!!! (the pre-trained model was in python2, so please run extractor with python2, or you will get wrong prediction.) 

## Quick Start: RUN Extractor

`mkdir predictions`

`python MIMO_Extractor.py --cuda --udata ./self_train/udata/stmts-demo-unlabeled-small.tsv --out_file ./predictions/stmts-demo-small-prediction --language_model ./models/LM/model.pt --wordembed ./models/WE/pubmed-vectors=50.bin`

## TRAIN MODEL

`mkdir models`
`mkdie results`

### Supervised MIMO (single featrue with multi-input gates):
 
an example:

`python train.py --cuda --config 000111000 --language_model ./models/LM/model.pt --wordembed ./models/WE/pubmed-vectors=50.bin`

### Supervised MIMO (multi-input gates, multi-input ensembles):

an example:

`python train_ensemble.py --cuda --config 111 --language_model ./models/LM/model.pt --wordembed ./models/WE/pubmed-vectors=50.bin`

### Semi-supervised MIMO (single featrue with multi-input gates)

an example:

`python self_train.py --cuda --language_model ./models/LM/model.pt --wordembed ./models/WE/pubmed-vectors=50.bin --check_point ../models/supervised_model_011000000.torch --AR --TC --SH --DEL`

### Semi-supervised MIMO (multi-input gates, multi-input ensembles):

an example:

`python self_train_ensemble.py --cuda --language_model ./models/LM/model.pt --wordembed ./models/WE/pubmed-vectors=50.bin --check_point ../models/ensemble_supervised_model_111.torch --AR --TC --SH --DEL`

## DOWNLOAD

* The word embedding we use can be found [here](https://www.dropbox.com/sh/6yx1l8euehgw12k/AAB9mWc3m8H7niuEF7NBYUdRa?dl=0}{\underline{here}}\footnote{\url{https://www.dropbox.com/sh/6yx1l8euehgw12k/AAB9mWc3m8H7niuEF7NBYUdRa?dl=0}).

* The pre-trained language model we use can be found [here](https://www.dropbox.com/sh/q1kehix8q58sxmh/AADU35QFu1ZMuNQFTiEYWSxUa?dl=0}{\underline{here}}\footnote{\url{https://www.dropbox.com/sh/q1kehix8q58sxmh/AADU35QFu1ZMuNQFTiEYWSxUa?dl=0}}).

* Pre-trained model: For a quick use of the proposed semi-supervised MIMO model, we put the pre-trained model files [here](https://www.dropbox.com/sh/rfm95k9kopmfdpb/AACCUzHvpR2M3GOIs9nyNo1Ua?dl=0}{\underline{online}}\footnote{\url{https://www.dropbox.com/sh/rfm95k9kopmfdpb/AACCUzHvpR2M3GOIs9nyNo1Ua?dl=0}}).
