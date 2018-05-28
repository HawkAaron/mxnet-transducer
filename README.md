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
    def __init__(self, layout='NTC', label_layout='NT', blank_label=0, weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
            "Only 'NTC' and 'TNC' layouts for pred are supported. Got: %s"%layout
        assert label_layout in ['NT', 'TN'],\
            "Only 'NT' and 'TN' layouts for label are supported. Got: %s"%label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(RNNTLoss, self).__init__(weight, batch_axis, **kwargs)
        self.blank_label = blank_label

    def hybrid_forward(self, F, pred, label, pred_lengths, label_lengths):
        if self._layout == 'NTC':
            pred = F.moveaxis(pred, 0, 2)
        if self._batch_axis == 1:
            label = F.swapaxes(label, 0, 1)

        loss = F.contrib.RNNTLoss(pred, label, pred_lengths, 
                        label_lengths, blank_label=self.blank_label)
        return loss

```

From the repo test with:

```bash
python test/test.py 10 300 100 50 --mx
```

## Reference
* [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* [SPEECH RECOGNITION WITH DEEP RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1303.5778.pdf)
* [Baidu warp-ctc](https://github.com/baidu-research/warp-ctc)
* [warp-transducer](https://github.com/HawkAaron/warp-transducer)
