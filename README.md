# mxnet-transducer
A fast parallel implementation of RNN Transducer, on both CPU and GPU for mxnet.

## Compilation
First get mxnet and the code:
``` bash
git clone --recursive https://github.com/apache/incubator-mxnet
git clone https://github.com/HawkAaron/mxnet-transducer
```

Copy all files into mxnet dir:
``` bash
cp -r mxnet-transducer/rnnt* incubator-mxnet/src/operator/contrib/
```

Then follow the installation instructions of mxnet:
```
https://mxnet.incubator.apache.org/install/index.html
```

## Reference
* [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* [Baidu warp-ctc](https://github.com/baidu-research/warp-ctc)
* [warp-transducer](https://github.com/HawkAaron/warp-transducer)
