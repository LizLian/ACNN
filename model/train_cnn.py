# codeing: utf-8

import argparse
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon.data import DataLoader
import gluonnlp as nlp
from sklearn.metrics import precision_recall_curve, average_precision_score

from load_data import load_dataset
from model_cnn import RelationClassifier
from utils import logging_config
from mxnet.gluon.loss import Loss

parser = argparse.ArgumentParser(description='Train a simple binary relation classifier')
parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
parser.add_argument('--epochs', type=int, default=20, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.1)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=64)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--embedding_source', type=str, default='freebase-vectors-skipgram1000-en', help='Pre-trained embedding source name')
parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')

# freebase-vectors-skipgram1000-en

args = parser.parse_args()
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()


def classify_test_data(model, data_test, ctx=mx.cpu()):
    """
    Generate predictions on the test data and write to file in same format as training data
    """
    preds = []
    for i, x in enumerate(data_test):
        data, inds, label = x
        data = data.as_in_context(ctx)
        inds = inds.as_in_context(ctx)
        score = model(data, inds)
        predictions = predict(score)
        preds.extend(predictions)
    return preds


def train_classifier(vocabulary, transformer, data, data_test=None, ctx=mx.cpu()):
    """
    Main loop for training a classifier
    """
    if data_test:
        data_test  = gluon.data.SimpleDataset(data_test).transform(transformer)    
    if data_test:
        test_dataloader  = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape if vocabulary.embedding else (len(vocabulary), 128)

    num_classes = 19 # XXX - parameterize and/or derive from dataset
    model = RelationClassifier(emb_input_dim, emb_output_dim, num_classes=num_classes)
    differentiable_params = []

    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)  ## initialize model parameters on the context ctx
    if not args.random_embedding:
        model.embedding.weight.set_data(vocabulary.embedding.idx_to_vec) ## set the embedding layer parameters to pre-trained embedding
    elif args.fixed_embedding:
        model.embedding.collect_params().setattr('grad_req', 'null')

    # model.hybridize() ## OPTIONAL for efficiency - perhaps easier to comment this out during debugging

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in model.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    # wd - weight decay (regularization)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': args.lr, 'wd': 0.0001})
    distance_loss = DistanceLoss(ctx)

    k = 1000
    for epoch in range(args.epochs):
        train_start = epoch%8 * k
        train_end = epoch%8 * k + int(8000 * 0.85)
        if train_end>8000:
            data_train = data[0:train_end-8000] + data[train_start: train_end]
            data_val = data[train_end-8000: train_start]
        else:
            data_train = data[train_start: train_end]
            data_val = data[train_end:] + data[0: train_start]
        data_train = gluon.data.SimpleDataset(data_train).transform(transformer)
        data_val = gluon.data.SimpleDataset(data_val).transform(transformer)
        train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
        val_dataloader = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False)

        epoch_loss = 0
        for i, x in enumerate(train_dataloader):
            data1, inds, label = x
            data1 = data1.as_in_context(ctx)
            label = label.as_in_context(ctx)
            inds = inds.as_in_context(ctx)
            with autograd.record():
                score = model(data1, inds)
                # l = loss_fn(score, label).mean()
                l = distance_loss(score, label)
            l.backward()
            grads = [p.grad(ctx) for p in differentiable_params]
            gluon.utils.clip_global_norm(grads, 1)
            trainer.step(1) ## step = 1 since we took the mean of the loss over the batch
            epoch_loss += l.asscalar()
        logging.info(f"Epoch{epoch} loss = {epoch_loss}")
        val_acc = _eval(model, val_dataloader, ctx)
        train_acc = _eval(model, train_dataloader, ctx)
        logging.info(f"Train Acc = {train_acc}, Validation Acc = {val_acc}")
    model.save_parameters('base.params')
    preds = classify_test_data(model, test_dataloader, ctx)
    return model, preds


class DistanceLoss(nn.Block):
    """
    this is the ranking loss function implemented from paper https://www.aclweb.org/anthology/P15-1061.pdf
    Like some other ranking approaches that only update two classes/examples at every training round ,
    this ranking approach can efficiently train the network for tasks which have a very large number of classes.

    """

    def __init__(self, ctx):
        super(DistanceLoss, self).__init__()
        self.ctx = ctx

    def forward(self, score, label, mplus=2.5, mNeg=0.5, gamma=2):
        # rel_weight (dc=500, nr=19)
        # score (batch_size, dr=19)
        rows = mx.nd.array(list(range(len(score))))
        # ground truth score
        gt_score = score[rows, label.transpose()[0,:]].as_in_context(self.ctx)
        gt_score = mx.nd.log(1 + mx.nd.exp(gamma * (mplus - gt_score))) + mx.nd.log(
            1 + mx.nd.exp(gamma * (-100 + gt_score)))  # positive loss

        # top two scores for each batch, return the score that's different from ground truth
        val, inds = mx.nd.topk(score, axis=1, k=2, ret_typ='both')
        predT = inds[:, 0].astype('int').as_in_context(self.ctx) == label.transpose((1,0)).astype('int').as_in_context(self.ctx)
        predF = inds[:, 0].astype('int').as_in_context(self.ctx) != label.transpose((1,0)).astype('int').as_in_context(self.ctx)
        predT = predT[0,:]
        predF = predF[0,:]

        part2 = mx.nd.log(1 + mx.nd.exp(gamma * (mNeg + val))) + mx.nd.log(
            1 + mx.nd.exp(gamma * (-100 - val)))  # negative loss
        part2 = mx.nd.dot(predT.astype('float'), part2[:, 1].astype('float')) + mx.nd.dot(predF.astype('float'), part2[:, 0].astype('float'))

        # exclude other loss
        # noneOtherInd = label != 18  # not Other index
        # noneOtherInd = noneOtherInd.transpose((1,0))[0, :]
        # loss = mx.nd.dot(noneOtherInd.astype('float'), gt_score.astype('float')) + part2.astype('float')  # exclude other loss
        # include other loss
        loss = mx.nd.sum(gt_score.astype('float')) + part2.astype('float')
        return loss/label.shape[0]


def predict(score):
    # implementation of
    # val, ind = mx.nd.topk(score, axis=1, k=2, ret_typ='both')
    # res = [ind[i][0].asscalar() if ind[i][0].asscalar() != 18 or (ind[i][0].asscalar() == 18 and val[i][1].asscalar() < 0)
           # else ind[i][1].asscalar() for i in range(len(score))]
    # predictions = mx.nd.array(res)
    predictions = mx.nd.argmax(score, axis=1)
    return predictions


def _eval(model, dataloader, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    acc = 0
    total_correct = 0
    total = 0
    for i, (data, inds, label) in enumerate(dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        inds = inds.as_in_context(ctx)
        score = model(data, inds)
        predictions = predict(score)
        correctPreds = predictions.astype('int').as_in_context(ctx) == \
                       label.transpose()[0,:].astype('int').as_in_context(ctx)
        total_correct += mx.nd.sum(correctPreds).asscalar()
        total += len(data)
    acc = total_correct / float(total)
    return acc


if __name__ == '__main__':
    logging_config(args.log_dir, 'train', level=logging.INFO)
    vocab, dataset, test_dataset, transform = load_dataset(args.train_file, args.test_file)

    if args.embedding_source:
        ## set embeddings as in PA1 or as appropriate for your approach
        glove_twitter = nlp.embedding.create('word2vec', source=args.embedding_source)
        vocab.set_embedding(glove_twitter)
        # .....
    emb_dim = vocab.embedding.idx_to_vec.shape[1]

    ctx = mx.cpu() ## or mx.gpu(N) if GPU device N is available
    model, preds = train_classifier(vocab, transform, dataset, test_dataset, ctx)

    label_map = transform._label_map
    label_map = {label_map[label]: label for label in label_map}
    pred_labels = []
    try:
        for pred in preds:
            pred_labels.append(pred.asscalar())
    except:
        print(pred)
    # pred_labels = [label_map[pred.asscalar()] for pred in preds]
    with open("predictions.tsv", "w") as outf:
        for label in pred_labels:
            outf.write(label_map[label]+"\n")
