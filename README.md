# mxnet-transducer
A fast parallel implementation of RNN Transducer (Graves 2013 joint network), on both CPU and GPU for mxnet.

[GPU version is now available for Graves2012 add network.](https://github.com/HawkAaron/mxnet-transducer/tree/add_network)

## Install and Test
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

Finally, add Python API into `/path/to/mxnet_root/mxnet/gluon/loss.py`:

```python
class RNNTLoss(Loss):
    def __init__(self, batch_first=True, blank_label=0, weight=None, **kwargs):
        batch_axis = 0 if batch_first else 2
        super(RNNTLoss, self).__init__(weight, batch_axis, **kwargs)
        self.batch_first = batch_first
        self.blank_label = blank_label

    def hybrid_forward(self, F, pred, label, pred_lengths, label_lengths):
        if not self.batch_first:
            pred = F.transpose(pred, (2, 0, 1, 3))

        loss = F.contrib.RNNTLoss(pred, label.astype('int32', False), 
                                    pred_lengths.astype('int32', False), 
                                    label_lengths.astype('int32', False), 
                                    blank_label=self.blank_label)
        return loss

```

From the repo test with:

```bash
python test/test.py --mx
```

## Reference
* [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* [SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1303.5778.pdf)
* [Baidu warp-ctc](https://github.com/baidu-research/warp-ctc)
* [warp-transducer](https://github.com/HawkAaron/warp-transducer)
