#!/bin/bash
wget https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz?raw=true
mv t10k-images-idx3-ubyte.gz?raw=true t10k-images-idx3-ubyte.gz
wget https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-labels-idx1-ubyte.gz?raw=true
mv t10k-labels-idx1-ubyte.gz?raw=true t10k-labels-idx1-ubyte.gz
wget https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-images-idx3-ubyte.gz?raw=true
mv train-images-idx3-ubyte.gz?raw=true train-images-idx3-ubyte.gz
wget https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-labels-idx1-ubyte.gz?raw=true
mv train-labels-idx1-ubyte.gz?raw=true train-labels-idx1-ubyte.gz
