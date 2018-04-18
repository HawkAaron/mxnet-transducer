# warp-transducer
A fast parallel implementation of RNN Transducer, on CPU for mxnet.

## Compilation
First get mxnet and the code:
``` bash
git clone --recursive https://github.com/apache/incubator-mxnet
git clone https://github.com/HawkAaron/warp-transducer
```

Move all files into mxnet dir:
``` bash
move warp-transducer/rnnt* incubator-mxnet/src/operator/contrib/
```

Then follow the installation instructions of mxnet:
```
https://mxnet.incubator.apache.org/install/index.html
```

## TODO
* Performance test
* GPU implementation
