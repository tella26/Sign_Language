# This repository contains the sign language recognition task. 
# The task is to recognize the sign in each sign’s video. 
# The skeleton of the signer was already extracted using MediaPipe from all the videos’ frames. 
============================================================================================

#  The code is based on the  WLASL: A large-scale dataset for Word-Level American Sign Language [WASL](https://dxli94.github.io/WLASL/)

The code has implementation of Transformer and TGCN (Temporal Graph Convolutional Networks) for Sign Language. 

# To run the Transformer code

```
cd code/Transformer
```
```
python train_transformer.py  
```

# To run the TGCN code

```
cd code/TGCN
```
```
python train_tgcn.py  
```

# Transformer Results

```
INFO:root:Top-1 acc: 15.0000
INFO:root:Top-3 acc: 43.0000
INFO:root:Top-5 acc: 69.0000
INFO:root:Top-10 acc: 100.0000
INFO:root:Top-30 acc: 100.0000

```
