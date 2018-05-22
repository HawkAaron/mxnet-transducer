# mxnet-transducer
A fast parallel implementation of RNN Transducer, on both CPU and GPU for mxnet.

## Install
First get mxnet and the code:
``` bash
git clone --recursive https://github.com/apache/incubator-mxnet
git clone https://github.com/HawkAaron/mxnet-transducer
```

Copy all files into mxnet dir:
``` bash
cd mxnet-transducer; git checkout add_network; cd ..
cp -r mxnet-transducer/rnnt* incubator-mxnet/src/operator/contrib/
```

Then follow the installation instructions of mxnet:
```
https://mxnet.incubator.apache.org/install/index.html
```

Finally, add python API into `/mxnet_package_root/mxnet/gluon/loss.py`:
```
class RNNTLoss(Loss):
    def __init__(self, layout='NTC', label_layout='NT', blank_label=0, weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
            "Only 'NTC' and 'TNC' layouts for acts are supported. Got: %s"%layout
        assert label_layout in ['NT', 'TN'],\
            "Only 'NT' and 'TN' layouts for label are supported. Got: %s"%label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(RNNTLoss, self).__init__(weight, batch_axis, **kwargs)
        self.blank_label = blank_label
        
    def hybrid_forward(self, F, trans_acts, pred_acts, label, pred_lengths, label_lengths):
        if self._layout == 'TNC':
            trans_acts = F.swapaxes(trans_acts, 0, 1)
            pred_acts = F.swapaxes(pred_acts, 0, 1)
        if self._batch_axis == 1:
            label = F.swapaxes(label, 0, 1)

        loss = F.contrib.RNNTLoss(trans_acts, pred_acts, label, 
            pred_lengths, label_lengths, blank_label=self.blank_label)
        return loss
```

## Reference
* [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* [Baidu warp-ctc](https://github.com/baidu-research/warp-ctc)
* [warp-transducer](https://github.com/HawkAaron/warp-transducer)
