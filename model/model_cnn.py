# codeing: utf-8

import math
import os
import numpy as np
import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn
from mxnet import autograd as ag
from mxnet.gluon.loss import L2Loss
from mxnet.gluon.block import HybridBlock, Block
from gluonnlp.model import AttentionCell, MLPAttentionCell, DotProductAttentionCell, MultiHeadAttentionCell

N = 50

def _get_attention_cell(attention_cell, units=None,
                        scaled=True, num_heads=None,
                        use_bias=False, dropout=0.0):
    """
    Parameters
    ----------
    attention_cell : AttentionCell or str
    units : int or None

    Returns
    -------
    attention_cell : AttentionCell
    """
    if isinstance(attention_cell, str):
        if attention_cell == 'scaled_luong':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=True)
        elif attention_cell == 'scaled_dot':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'dot':
            return DotProductAttentionCell(units=units, scaled=False, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'cosine':
            return DotProductAttentionCell(units=units, scaled=False, use_bias=use_bias,
                                           dropout=dropout, normalized=True)
        elif attention_cell == 'mlp':
            return MLPAttentionCell(units=units, normalized=False)
        elif attention_cell == 'normed_mlp':
            return MLPAttentionCell(units=units, normalized=True)
        elif attention_cell == 'multi_head':
            base_cell = DotProductAttentionCell(scaled=scaled, dropout=dropout)
            return MultiHeadAttentionCell(base_cell=base_cell, query_units=units, use_bias=use_bias,
                                          key_units=units, value_units=units, num_heads=num_heads)
        else:
            raise NotImplementedError
    else:
        assert isinstance(attention_cell, AttentionCell),\
            'attention_cell must be either string or AttentionCell. Received attention_cell={}'\
                .format(attention_cell)
        return attention_cell


def _position_encoding_init(max_length, dim):
    """ Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc


def _pos(x):
  '''
  map the relative distance between [0, 123)
  '''
  if x < -60:
      return 0
  if x >= -60 and x <= 60:
      return x + 61
  if x > 60:
      return 122


class RelationClassifier(HybridBlock):
    """
    Your primary model block for attention-based, convolution-based or other classification model
    emb_input_dim: Size of the vocabulary
    emb_output_dim: embedding length
    """
    def __init__(self, emb_input_dim, emb_output_dim, max_seq_len=100, filters=[2,3,4,5], num_classes=19, dropout=0.2, is_training=True):
        super(RelationClassifier, self).__init__()
        self.max_len = max_seq_len
        # dw - embeding size
        self.dp = 50
        dc = 1000
        n = 100
        np = 123
        nr = 19
        self.is_training = is_training

        self.d = emb_output_dim + 2*self.dp
        ## model layers defined here
        with self.name_scope():
            self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
            self.dist_embedding = nn.Embedding(np, self.dp)

            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            # sliding window + convolution layer
            self.conv1 = nn.Conv2D(dc, (filters[0], self.d), (1, self.d), in_channels=1, activation='relu')
            self.conv2 = nn.Conv2D(dc, (filters[1], self.d), (1, self.d), in_channels=1, activation='relu')
            self.conv3 = nn.Conv2D(dc, (filters[2], self.d), (1, self.d), in_channels=1, activation='relu')
            self.conv4 = nn.Conv2D(dc, (filters[3], self.d), (1, self.d), in_channels=1, activation='relu')

            # self.maxpool = nn.MaxPool1D(max_seq_len, strides=1)
            self.wl = nn.Dense(nr, use_bias=False)

    def input_attention(self, data, inds):
        # d1 - relative distance from each word to entity1
        # d2 - relative distance from each word to entity2
        dist1 = []
        dist2 = []
        for sent, pos in zip(data, inds):
            d1 = [_pos(int(pos[0].asscalar()) - idx) for idx, _ in enumerate(sent)]
            d2 = [_pos(int(pos[1].asscalar()) - idx) for idx, _ in enumerate(sent)]
            dist1.append(d1)
            dist2.append(d2)
        dist1 = mx.nd.array(dist1)
        dist2 = mx.nd.array(dist2)
        dist1_emb = self.dist_embedding(dist1) # (batch_size, n=100)
        dist2_emb = self.dist_embedding(dist2) # (batch_size, n=100)
        x_emb = self.embedding(data) # (batch_size, n=100, hidden_units=300)
        x_concat = mx.nd.concat(x_emb, dist1_emb, dist2_emb, dim=2) # (batch_size, n=100, d=350)

        # self-attention layer
        # inds = mx.nd.one_hot(inds, self.max_len)
        # ind_embeddings = mx.nd.batch_dot(inds, x_emb) #(batch_size, 2, hidden_units=300)
        # attention_scores = mx.nd.batch_dot(x_emb, ind_embeddings.transpose((0, 2, 1))) # (batch_size, n=100, 2)
        # attention_scores = mx.nd.mean(mx.nd.softmax(attention_scores, axis=1), axis=2) # (batch_size, n=100)
        # R = x_concat * mx.nd.expand_dims(attention_scores, axis=2) # R shape (batch_size, n=100, dw=350)
        # return R
        return x_concat

    def scoring(self, R_star):
        # R_star (batch_size, dc=500, n=100)
        # R_star.transpose x WL
        # (1, dc) x (dc, nr)
        score = self.wl(R_star.transpose((0,2,1))) # (batch_size, n=100, nr=19)
        return score

    def hybrid_forward(self, F, data, inds):
        """
        Inputs:
         - data The sentence representation (token indices to feed to embedding layer)
         - inds A vector - shape (2,) of two indices referring to positions of the two arguments
        NOTE: Your implementation may involve a different approach
        """
        R = self.input_attention(data, inds) # R shape (batch_size, n=100, dw=350)
        R = self.dropout1(R)
        R = mx.nd.expand_dims(R, 1) # (batch_size, in_channel=1, n=100, dw=350)

        conv1 = self.conv1(R)[:, :, :, 0]  # (batch_size, dc=500, n=100, 1)
        maxpool1 = mx.nd.max(conv1, 2)
        conv2 = self.conv2(R)[:, :, :, 0]
        maxpool2 = mx.nd.max(conv2, 2)
        conv3 = self.conv3(R)[:, :, :, 0]
        maxpool3 = mx.nd.max(conv3, 2)
        conv4 = self.conv4(R)[:, :, :, 0]
        maxpool4 = mx.nd.max(conv4, 2)
        maxpool_out = mx.nd.concat(maxpool1, maxpool2, maxpool3, maxpool4, dim=1)  # (batch_size, kernal_size*dc)
        maxpool_out = self.dropout2(maxpool_out)
        score = self.wl(maxpool_out) # (batch_size, nr=19)
        return score


class PositionwiseFFN(HybridBlock):
    """
    Taken from the gluon-nlp library.
    """
    def __init__(self, units=512, hidden_size=1024, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros', activation='relu',
                 prefix=None, params=None):
        super(PositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  activation=activation,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        outputs = self.ffn_1(inputs)
        outputs = self.ffn_2(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        return outputs

class BaseEncoderCell(HybridBlock):
    """Structure of the Transformer Encoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BaseEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell = _get_attention_cell(attention_cell,
                                                      units=units,
                                                      num_heads=num_heads,
                                                      scaled=scaled,
                                                      dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        arg_inputs: Symbol or NDArray
            Input arguments. Shape (batch_size, 2)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the encoder cell. Shape (batch_size, 2, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        outputs, attention_weights = self.attention_cell(inputs, inputs, None, None)
        outputs = self.proj(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        return outputs


class BaseEncoder(HybridBlock):

    def __init__(self, attention_cell='multi_head', 
                 units=512, hidden_size=2048, max_length=64,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BaseEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            ## !!! Original code creates a number of attention layers
            ## !!! Hard-coded here for a single base encoder cell for simplicity
            #self.transformer_cells = nn.HybridSequential()
            #for i in range(num_layers):
            #    self.transformer_cells.add(
            self.base_cell = BaseEncoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_cell=attention_cell,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout,
                        use_residual=use_residual,
                        scaled=scaled,
                        output_attention=output_attention,
                        prefix='transformer')

    def __call__(self, inputs): #pylint: disable=arguments-differ
        return super(BaseEncoder, self).__call__(inputs)

    def hybrid_forward(self, F, inputs, position_weight): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        arg_pos: int array pair

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        """
        batch_size = inputs.shape[0]
        steps = F.arange(self._max_length)
        inputs = F.broadcast_add(inputs, F.expand_dims(F.Embedding(steps, position_weight,
                                                                       self._max_length,
                                                                       self._units), axis=0))
        inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        outputs = self.base_cell(inputs)
        return outputs


class BinRelEncoderCell(HybridBlock):
    """This is a TransformerEncoder block/layer that generates outputs corresponding to
    exactly two positions in the input.  These should be integer offsets (0-based) provided
    as an ndarray (with exactly two elements).
    """
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=128, num_heads=4, scaled=True,
                 dropout=0.2, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BinRelEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell = _get_attention_cell(attention_cell,
                                                      units=units,
                                                      num_heads=num_heads,
                                                      scaled=scaled,
                                                      dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, arg_inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        arg_inputs: Symbol or NDArray
            Input arguments. Shape (batch_size, 2)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the encoder cell. Shape (batch_size, 2, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        ## the query should just be the inputs for the two RELATION ARGUMENTS (e1, e2)
        arg_outputs, attention_weights = self.attention_cell(arg_inputs, inputs, None, None)
        arg_outputs = self.proj(arg_outputs)
        arg_outputs = self.dropout_layer(arg_outputs)
        if self._use_residual:
            arg_outputs = arg_outputs + arg_inputs
        arg_outputs = self.layer_norm(arg_outputs)
        arg_outputs = self.ffn(arg_outputs)
        return arg_outputs


class BinRelEncoder(HybridBlock):
    """Same as a TransformerEncoder but generating only two hidden outputs
    at the positions of the two relation arguments.
    """
    def __init__(self, attention_cell='multi_head',
                 units=128, hidden_size=256, max_length=64,
                 num_heads=4, scaled=True, dropout=0.2,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BinRelEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.binrel_cell = BinRelEncoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_cell=attention_cell,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout,
                        use_residual=use_residual,
                        scaled=scaled,
                        output_attention=output_attention,
                        prefix='binrel_transformer')

    def __call__(self, inputs, arg_pos): #pylint: disable=arguments-differ
        return super(BinRelEncoder, self).__call__(inputs, arg_pos)

    def hybrid_forward(self, F, inputs, arg_pos): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        arg_pos: int array pair

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        """
        batch_size = inputs.shape[0]
        inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        ## take/slice the inputs at the argument positions across the batch
        offsets = F.transpose(F.expand_dims(F.arange(batch_size) * self._max_length, axis=0))
        input_stacked = F.reshape(inputs, (-1, self._units))
        to_take = arg_pos + offsets
        args = F.take(input_stacked, to_take, axis=0)

        outputs = self.binrel_cell(inputs, args)
        return outputs

    
