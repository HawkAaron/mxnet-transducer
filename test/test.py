"""
Tests for the C implementation of the sequence transducer.

From outside the package directory, run
`python -m transducer.test.`
"""
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import mxnet as mx

from rnnt_mx import RNNTLoss
from rnnt_np import RNNTLoss as rnntloss

parser = argparse.ArgumentParser(description='MXNet RNN Transducer Test.')
parser.add_argument('B', type=int, default=1, help='batch size')
parser.add_argument('T', type=int, default=300, help='time step')
parser.add_argument('U', type=int, default=100, help='prediction step')
parser.add_argument('V', type=int, default=60, help='vocab size')
parser.add_argument('--mx', default=False, action='store_true')
args = parser.parse_args()

def wrap_and_call(trans_acts, pred_acts, labels, blank=0):
    trans_acts = mx.nd.array(trans_acts, ctx=ctx)
    pred_acts = mx.nd.array(pred_acts, ctx=ctx)
    lengths = mx.nd.array([trans_acts.shape[1]] * trans_acts.shape[0], dtype='int32', ctx=ctx)
    label_lengths = mx.nd.array([len(l) for l in labels], dtype='int32', ctx=ctx)
    labels = mx.nd.array(labels, dtype='int32', ctx=ctx)
    fn = RNNTLoss(blank_label=blank) if args.mx else rnntloss(blank_label=blank)
    with mx.autograd.record():
        trans_acts.attach_grad()
        pred_acts.attach_grad()
        costs = fn(trans_acts, pred_acts, labels, lengths, label_lengths)
        costs.backward()
    return costs.asnumpy(), trans_acts.grad.asnumpy(), pred_acts.grad.asnumpy()

def small_test():
    B = 1; T = 4; U = 3; V = 5
    trans_acts = np.array([[[0.06860042, 0.02114975, 0.9319364,  0.32975072, 0.13327402],
                            [0.12564015, 0.43477476, 0.5019661,  0.6318551,  0.56487304],
                            [0.7360039,  0.39700246, 0.50331944, 0.6365183,  0.19341111],
                            [0.7335776,  0.9189112,  0.65545017, 0.89854324, 0.0246467 ]]])
    pred_acts = np.array([[[0.1928187,  0.63991195, 0.7839058,  0.8364832,  0.55822134],
                            [0.11235255, 0.44893646, 0.46855223, 0.20775604, 0.9215501 ],
                            [0.1369242,  0.6790915,  0.3493181,  0.72844136, 0.46717978]]])

    labels = [[1, 2]]

    cost, trans_grads, pred_grads = wrap_and_call(trans_acts, pred_acts, labels)

    expected_cost = 9.3370

    expected_trans_grads = np.array([[[-0.85036933, -0.17704993,  0.4379695,   0.32460052,  0.26484928],
                                    [-0.86005664, -0.01435287,  0.13164294,  0.36458912,  0.37817746],
                                    [-0.7415191,   0.10448333,  0.01432155,  0.36094004,  0.2617742 ],
                                    [-0.7804755,   0.3140569,  -0.12812297,  0.4177104,   0.17683122]]])
    expected_pred_grads = np.array([[[-0.80384177, -0.66368896,  0.6344613,   0.51796013,  0.31510928],
                                    [-0.65244085,  0.36658645, -0.5477955,   0.34172153,  0.49192834],
                                    [-1.7761381,   0.5242399,   0.36914518,  0.60815847,  0.27459458]]])

    assert np.allclose(cost, expected_cost, rtol=1e-4), \
        "small_test costs mismatch."
    assert np.allclose(trans_grads, expected_trans_grads, rtol=1e-4), \
        "trans gradient mismatch."
    assert np.allclose(pred_grads, expected_pred_grads, rtol=1e-4), \
        "pred gradient mismathc."

def big_test():
    B = 2; T = 4; U = 3; V = 6; blank = 5
    # minibatch x T x U x alphabet_size
    trans_acts = np.array([0.20836472511291504 ,0.6848891377449036 ,0.8508703112602234 ,0.5761988759040833 ,0.19992691278457642 ,0.8066366910934448 ,
                        0.7215913534164429 ,0.2725244164466858 ,0.20181220769882202 ,0.7149978280067444 ,0.7996553182601929 ,0.5940970182418823 ,
                        0.9547191262245178 ,0.3950379490852356 ,0.9179182648658752 ,0.6635433435440063 ,0.5223862528800964 ,0.3065106272697449 ,
                        0.9895828366279602 ,0.7198997735977173 ,0.3201969265937805 ,0.4918731451034546 ,0.827298641204834 ,0.7208738327026367 ,
                        0.5181616544723511 ,0.8324336409568787 ,0.29219377040863037 ,0.3595501780509949 ,0.6904011964797974 ,0.8513350486755371 ,
                        0.5865226984024048 ,0.8465507626533508 ,0.7300622463226318 ,0.24205732345581055 ,0.24660539627075195 ,0.931033730506897 ,
                        0.8725375533103943 ,0.06845909357070923 ,0.7426746487617493 ,0.7473852038383484 ,0.6735857129096985 ,0.8149459958076477 ,
                        0.6253803968429565 ,0.5640403628349304 ,0.5929765701293945 ,0.6260771751403809 ,0.23223882913589478 ,0.04109394550323486]
                        ).reshape(T, B, V).transpose(1, 0, 2)

    pred_acts = np.array([0.06116640567779541 ,0.14563453197479248 ,0.5638840198516846 ,0.6632290482521057 ,0.19838422536849976 ,0.1820780634880066 ,
                        0.6904842257499695 ,0.30375921726226807 ,0.6189450621604919 ,0.0328218936920166 ,0.7522785663604736 ,0.826593279838562 ,
                        0.9041121006011963 ,0.31825321912765503 ,0.10209769010543823 ,0.4442335367202759 ,0.7338142991065979 ,0.22434186935424805 ,
                        0.7973766326904297 ,0.8608612418174744 ,0.4400267004966736 ,0.8985074758529663 ,0.37170130014419556 ,0.9338418245315552 ,
                        0.7007454037666321 ,0.6552602648735046 ,0.5205059051513672 ,0.30149775743484497 ,0.605181872844696 ,0.1901898980140686 ,
                        0.9128827452659607 ,0.6805384159088135 ,0.019013822078704834 ,0.8405444622039795 ,0.5298664569854736 ,0.27262967824935913]
                        ).reshape(U, B, V).transpose(1, 0, 2)

    expected_costs = [9.8610, 8.7003]

    expected_trans_grads = np.array([0.18298208713531494 ,-0.1863221824169159 ,0.24029867351055145 ,0.31508156657218933 ,0.17985235154628754 ,-0.7318925261497498 ,
                                    0.26263949275016785 ,-0.10713391751050949 ,0.13813626766204834 ,0.16261409223079681 ,0.2808478772640228 ,-0.7371038794517517 ,
                                    0.3619690537452698 ,-0.06026348099112511 ,0.08730498701334 ,0.25879740715026855 ,0.22078825533390045 ,-0.8685961961746216 ,
                                    0.3360159993171692 ,-0.1639198213815689 ,0.1387038677930832 ,0.1545054018497467 ,0.2540736496448517 ,-0.7193790674209595 ,
                                    0.27959010004997253 ,0.03639250993728638 ,-0.061875589191913605 ,0.19594523310661316 ,0.30305129289627075 ,-0.753103494644165 ,
                                    0.29537156224250793 ,-0.274471253156662 ,0.2312944084405899 ,0.18443721532821655 ,0.16433767974376678 ,-0.6009695529937744 ,
                                    0.385039746761322 ,0.06576273590326309 ,-0.18127907812595367 ,0.2354261875152588 ,0.28227531909942627 ,-0.7872248888015747 ,
                                    0.4264937937259674 ,-0.4397970736026764 ,0.22451627254486084 ,0.4020277261734009 ,0.20442542433738708 ,-0.8176661133766174]).reshape(T, B, V).transpose(1, 0, 2)
    expected_pred_grads = np.array([0.2336210012435913 ,-0.7242842316627502 ,0.49153390526771545 ,0.4382175803184509 ,0.2350836545228958 ,-0.6741719245910645 ,
                                    0.5328136086463928 ,-0.7057985663414001 ,0.3393358588218689 ,0.2234761267900467 ,0.5142948627471924 ,-0.9041219353675842 ,
                                    0.5180250406265259 ,0.26190635561943054 ,-0.7572638988494873 ,0.29955601692199707 ,0.3859791159629822 ,-0.7082027196884155 ,
                                    0.4035489857196808 ,-0.5864231586456299 ,0.24028921127319336 ,0.3576064705848694 ,0.20584842562675476 ,-0.6208699345588684 ,
                                    0.4579349160194397 ,0.31794747710227966 ,0.35017895698547363 ,0.26747676730155945 ,0.3649044632911682 ,-1.7584426403045654 ,
                                    0.38415825366973877 ,0.30689966678619385 ,0.15302574634552002 ,0.3225018382072449 ,0.18354131281375885 ,-1.35012686252594]).reshape(U, B, V).transpose(1, 0, 2)

    labels = [[1, 2],
              [1, 1]]

    costs, trans_grads, pred_grads = wrap_and_call(trans_acts, pred_acts, labels, blank)

    assert np.allclose(costs, expected_costs), \
        "big_test average costs mismatch."
    assert np.allclose(trans_grads, expected_trans_grads), \
        "big_test trans_grads mismatch."
    assert np.allclose(pred_grads, expected_pred_grads), \
        "big_test pred_grads mismatch."

def time_test(blank=0):
    batch_size = args.B
    vocab_size = args.V
    input_len = args.T
    output_len = args.U

    trans_acts = np.random.rand(batch_size, input_len, vocab_size)
    pred_acts = np.random.rand(batch_size, output_len + 1, vocab_size)

    labels = np.random.randint(1, vocab_size, (batch_size, output_len))

    start = time.time()
    iters = 10
    for _ in range(iters):
        cost, trans_grad, pred_grad = wrap_and_call(trans_acts, pred_acts, labels)
    end = time.time()

    print("Time per iteration: {:.3f}(s)".format((end-start)/iters))


if __name__ == "__main__":
    ctx = mx.cpu()
    small_test()
    big_test()
    print("CPU Tests passed!")
    ctx = mx.gpu()
    big_test()
    print('GPU Test passed!')
    time_test()