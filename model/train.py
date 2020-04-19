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
from model import RelationClassifier, GRUModel, N
from utils import logging_config
from mxnet.gluon.loss import Loss

parser = argparse.ArgumentParser(description='Train a simple binary relation classifier')
parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
parser.add_argument('--epochs', type=int, default=20, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=64)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.1)
parser.add_argument('--embedding_source', type=str, default='GoogleNews-vectors-negative300', help='Pre-trained embedding source name')
parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')

# freebase-vectors-skipgram1000

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
        wo, rel_weight = model(data, inds)
        predictions = predict(wo, rel_weight)
        preds.extend(predictions)
    return preds


def train_classifier(vocabulary, transformer, data_train, data_val, data_test=None, ctx=mx.cpu()):
    """
    Main loop for training a classifier
    """
    data_train = gluon.data.SimpleDataset(data_train).transform(transformer)
    data_val   = gluon.data.SimpleDataset(data_val).transform(transformer)
    if data_test:
        data_test  = gluon.data.SimpleDataset(data_test).transform(transformer)    
    
    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader   = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
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
    y = mx.nd.one_hot(mx.nd.array([i for i in range(19)]), 19).as_in_context(ctx)
    distance_loss = DistanceLoss()

    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, x in enumerate(train_dataloader):
            data, inds, label = x
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            inds = inds.as_in_context(ctx)
            with autograd.record():
                wo, rel_weight = model(data, inds)
                # l = loss_fn(output, label).mean()
                l = distance_loss(wo, rel_weight, y, label)
                # l = dist_loss2(wo, rel_weight, label, margin=1)
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


def evaluate(model, dataloader, ctx=mx.cpu()):
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
        out = model(data, inds)
        predictions = mx.nd.argmax(out, axis=1).astype('int32')
        for j in range(out.shape[0]):
            pred = int(predictions[j].asscalar())
            lab = int(label[j].asscalar())
            if lab == pred:
                total_correct += 1
            total += 1
    acc = total_correct / float(total)
    return acc


class DistanceLoss(nn.Block):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, wo, rel_weight, all_y, label, margin=1):
        # rel_weight (nr=19, n=100)
        label_onehot = mx.nd.one_hot(label[:, 0], 19)
        # wo_norm = mx.nd.L2Normalization(wo, 1) # (batch_size, dc=500)
        wo_norm = wo/mx.nd.expand_dims(mx.nd.max(wo, axis=1),1)  # (batch_size, n=100)
        wo_norm_tile = mx.nd.repeat(mx.nd.expand_dims(wo_norm, axis=1), repeats=label_onehot.shape[-1],
                                    axis=1)  # (batch_size, nr=19, dc=500)
        rel_emb = mx.nd.dot(label_onehot, rel_weight)  # (batch_size, n=100)
        ay_emb = mx.nd.dot(all_y, rel_weight)
        gt_dist = mx.nd.norm(wo_norm-rel_emb, 2, 1)  # (batch_size, 1)
        all_dist = mx.nd.norm(wo_norm_tile - ay_emb, 2, 2) # (batch_size, nr=19)
        masking_y = mx.nd.multiply(label_onehot, 10000)
        # get longest distance y
        neg_dist = mx.nd.min(all_dist+masking_y, 1)
        loss = mx.nd.sum(margin + gt_dist - neg_dist)
        return loss


def predict(wo, rel_weight):
    wo_norm = wo/mx.nd.expand_dims(mx.nd.max(wo, axis=1),1) # (batch_size, dc=500)
    wo_norm_tile = mx.nd.repeat(mx.nd.expand_dims(wo_norm, axis=1), repeats=19, axis=1)  # (batch_size, nr=19, dc=500)
    # all_dist = mx.nd.norm(wo_norm_tile - mx.nd.expand_dims(rel_weight, axis=0), 2, 2)  # (batch_size, nr=19)
    dist = mx.nd.norm(wo_norm_tile - rel_weight,2,2) # (batch_size, nr=19)
    predictions = mx.nd.argmin(dist, axis=1).astype('int32')
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
        wo, rel_weight = model(data, inds)
        predictions = predict(wo, rel_weight)
        for j in range(label.shape[0]):
            lab = int(label[j].asscalar())
            pred = int(predictions[j].asscalar())
            if lab == pred:
                total_correct += 1
            total += 1
    acc = total_correct / float(total)
    return acc


if __name__ == '__main__':
    logging_config(args.log_dir, 'train', level=logging.INFO)
    vocab, train_dataset, val_dataset, test_dataset, transform = load_dataset(args.train_file, args.val_file, args.test_file)

    if args.embedding_source:
        ## set embeddings as in PA1 or as appropriate for your approach
        glove_twitter = nlp.embedding.create('word2vec', source=args.embedding_source)
        vocab.set_embedding(glove_twitter)
        # .....
    emb_dim = vocab.embedding.idx_to_vec.shape[1]

    # set embeddings for e1_start, e1_end, e2_start, e2_end
    # vocab.embedding.idx_to_vec[5] = mx.nd.random.normal(0, 1, shape=(emb_dim,))
    # vocab.embedding.idx_to_vec[6] = mx.nd.random.normal(0, 1, shape=(emb_dim,))
    # vocab.embedding.idx_to_vec[7] = mx.nd.random.normal(0, 1, shape=(emb_dim,))
    # vocab.embedding.idx_to_vec[8] = mx.nd.random.normal(0, 1, shape=(emb_dim,))
    ctx = mx.cpu() ## or mx.gpu(N) if GPU device N is available
    model, preds = train_classifier(vocab, transform, train_dataset, val_dataset, test_dataset, ctx)

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
