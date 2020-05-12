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


def abs_position(max_length, dim):
    pos_enc = _position_encoding_init(max_length, dim)
    pos_enc = mx.nd.expand_dims(mx.nd.array(pos_enc), axis=0)
    return pos_enc


def relative_position(data, max_len, inds):
    dist1 = []
    dist2 = []
    for sent, pos in zip(data, inds):
        d1 = [_pos(int(pos[0].asscalar())-idx) for idx, _ in enumerate(sent)]
        d1 += [123] * (max_len - len(d1))
        d2 = [_pos(int(pos[1].asscalar())-idx) for idx, _ in enumerate(sent)]
        d2 += [123] * (max_len - len(d2))
        dist1.append(d1)
        dist2.append(d2)
    return mx.nd.one_hot(mx.nd.array(dist1), 124), mx.nd.one_hot(mx.nd.array(dist2), 124)


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


def input_rep(embeddings, dist1, dist2):
    pos_emb = _position_encoding_init(N, 124)
    pos_emb = mx.nd.array(pos_emb.transpose((1, 0)))
    d1 = mx.nd.dot(dist1, pos_emb)
    d2 = mx.nd.dot(dist2, pos_emb)
    return mx.nd.concat(embeddings, d1, d2, dim=2)


class RelationClassifier(HybridBlock):
    """
    Your primary model block for attention-based, convolution-based or other classification model
    emb_input_dim: Size of the vocabulary
    emb_output_dim: embedding length
    """
    def __init__(self, emb_input_dim, emb_output_dim, max_seq_len=100, num_hidden = 512, num_classes=19, dropout=0.2, is_training=True):
        super(RelationClassifier, self).__init__()
        self.max_len = max_seq_len
        # dw - embeding size
        self.dp = 25
        dc = 1000
        n = 100
        np = 124
        nr = 19
        self.is_training = is_training

        self.d = emb_output_dim + 2*self.dp
        ## model layers defined here
        with self.name_scope():
            self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
            # self.rel_weight = gluon.Parameter('rel_weight', shape=(nr, max_seq_len))
            self.relation_weight_layer = nn.Dense(dc, use_bias=False, flatten=False)
            self.dropout = nn.Dropout(dropout)
            # sliding window + convolution layer
            self.conv1 = nn.Conv2D(dc, (3, self.d), (1, self.d), (1, 0), in_channels=1)
            self.conv = nn.Conv1D(dc, kernel_size=3, padding=1, use_bias=True, activation='tanh')
            # self.U = gluon.Parameter('u', shape=(n, nr))
            self.U_layer = nn.Dense(nr, use_bias=False, flatten=False)
            self.max_pool = nn.MaxPool1D(n, strides=1)

            self.dist_embedding = nn.Embedding(np, self.dp)

    def input_attention(self, data, inds):
        dist1 = []
        dist2 = []
        for sent, pos in zip(data, inds):
            d1 = [_pos(int(pos[0].asscalar()) - idx) for idx, _ in enumerate(sent)]
            d1 += [123] * (self.max_len - len(d1))
            d2 = [_pos(int(pos[1].asscalar()) - idx) for idx, _ in enumerate(sent)]
            d2 += [123] * (self.max_len - len(d2))
            dist1.append(d1)
            dist2.append(d2)
        dist1 = mx.nd.array(dist1)
        dist2 = mx.nd.array(dist2)
        dist1_emb = self.dist_embedding(dist1) # (batch_size, n=100)
        dist2_emb = self.dist_embedding(dist2) # (batch_size, n=100)
        x_emb = self.embedding(data) # (batch_size, n=100, hidden_units=300)
        x_concat = mx.nd.concat(x_emb, dist1_emb, dist2_emb, dim=2) # (batch_size, n=100, d=350)
        if self.is_training:
            x_concat = self.dropout(x_concat)

        # z = self.sliding_window(x_concat)
        z = self.conv1(mx.nd.expand_dims(x_concat, 1))[:,:,:,0] # (batch_size, dc=500, n=100)
        z = z.transpose((0,2,1))

        inds = mx.nd.one_hot(inds, self.max_len)
        ind_embeddings = mx.nd.batch_dot(inds, x_emb) #(batch_size, 2, hidden_units=300)
        attention_scores = mx.nd.batch_dot(x_emb, ind_embeddings.transpose((0, 2, 1))) # (batch_size, n=100, 2)
        attention_scores = mx.nd.mean(mx.nd.softmax(attention_scores, axis=1), axis=2) # (batch_size, n=100)
        R = z * mx.nd.expand_dims(attention_scores, axis=2) # R shape (batch_size, n=100, dw=500)
        return R

    def sliding_window(self, input_data, window_size=3):
        #     input shape (batch_size, seq_len, dim)
        batch_size, seq_len, dim = input_data.shape
        for i in range(input_data.shape[1] - (window_size - 1)):
            input_concat = mx.nd.reshape(input_data[:, i:i + window_size, :], (batch_size, 1, window_size * dim))
            if i == 0:
                output = input_concat
            else:
                output = mx.nd.concat(output, input_concat, dim=1)
        # padding using edges
        output = mx.nd.pad(mx.nd.expand_dims(output, axis=0), mode="edge", pad_width=(0, 0, 0, 0, 1, 1, 0, 0))[0]
        return output

    def attentive_pooling(self, R_star):
        # R_star (batch_size, dc=500, n=100)
        # R.transpose x U x WL
        # (n, dc) x (dc, nr) x (nr, dc)
        RU = self.U_layer(R_star.transpose((0,2,1))) # (batch_size, n=100, nr=19)
        G = self.relation_weight_layer(RU) # (batch_size, n=100, dc=500)
        AP = mx.nd.softmax(G, axis=1) # (batch_size, n=100, dc=500)
        RA = R_star*AP.transpose((0,2,1)) # (batch_size, dc=500, n=100)
        # input for pooling should be (batch, channel, time), pooling is applied on time dim
        wo = self.max_pool(RA) # (batch_size, dc=500, 1)
        if self.is_training:
            wo = self.dropout(wo)
        return wo[:,:,0]

    def hybrid_forward(self, F, data, inds):
        """
        Inputs:
         - data The sentence representation (token indices to feed to embedding layer)
         - inds A vector - shape (2,) of two indices referring to positions of the two arguments
        NOTE: Your implementation may involve a different approach
        """
        R = self.input_attention(data, inds) # R shape (batch_size, n=100, dw=500)
        # R = mx.nd.expand_dims(R, axis=1)  # (batch_size, 1, n=100, dc=500)
        R_star = self.conv(R.transpose((0,2,1)))  # (batch_size, dc=500, n=100)
        # R_star = mx.nd.tanh(R_star)
        wo = self.attentive_pooling(R_star)
        rel_weight = self.relation_weight_layer.weight.data()
        return wo, rel_weight.transpose((1,0))


class GRUModel(HybridBlock):
    """
    Your primary model block for attention-based, convolution-based or other classification model
    """
    def __init__(self, emb_input_dim, emb_output_dim, max_seq_len=100, num_hidden=512, num_classes=19, dropout=0.3):
        super(GRUModel, self).__init__()
        self.max_len = max_seq_len
        ## model layers defined here
        with self.name_scope():
            self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
            self.position_encoder = PositionEncoding(emb_output_dim, dropout)
            self.attn_layer = BaseEncoder(units=num_hidden, max_length=max_seq_len)
            self.binrel_encoder = BinRelEncoder(units=num_hidden, max_length=max_seq_len)

            # CNN model
            self.cnn = nn.HybridSequential()
            with self.cnn.name_scope():
                self.cnn.add(nn.Conv1D(channels=512, kernel_size=3, activation='tanh', padding=1))
                self.cnn.add(nn.MaxPool1D(pool_size=3, strides=1, padding=1))

            # GRU model
            # self.gru = rnn.GRU(hidden_size=256, num_layers=2, bidirectional=True, dropout=dropout)

            # BiLSTM
            # self.lstm = rnn.LSTM(hidden_size=256, num_layers=2, bidirectional=True, dropout=dropout)
            # self.max_pool = nn.MaxPool1D(pool_size=3, strides=1, padding=1)
            # self implemented attention
            self.attn_out = Attention(max_seq_len, num_hidden)
            # transformer attention
            # self.attn_out = BaseEncoder(units=512, max_length=max_seq_len)
            self.dropout = nn.Dropout(dropout)

            self.decoder = nn.HybridSequential()
            with self.decoder.name_scope():
                self.decoder.add(nn.Dense(units=256, use_bias=True, activation='relu'))
                self.decoder.add(nn.Dropout(dropout))
                self.decoder.add(nn.Dense(units=num_classes, use_bias=True, activation='relu'))

    def hybrid_forward(self, F, data, indices):
        """
        Inputs:
         - data The sentence representation (token indices to feed to embedding layer)
         - inds A vector - shape (2,) of two indices referring to positions of the two arguments
         - pos_emb a randonly generated vector for positional embeddings
        NOTE: Your implementation may involve a different approach
        """
        embedded = self.embedding(data) ## shape (batch_size, length, emb_dim)
        batch_size, max_len, emb_output_dim = embedded.shape

        # relative position encoding
        dist1, dist2 = relative_position(data, self.max_len, indices)
        rel_position = input_rep(embedded, dist1, dist2).as_in_context(data.context)  # shape (batch_size, seq_len, embedding_dim+pos_dim)
        pos_encoded = self.position_encoder(rel_position)

        # sliding_input = sliding_window(pos_encoded, 3) # output shape (batch_size, seq_len-window_size, dim)
        # attended_input = _input_attention(embedded, indices, sliding_input) # self-implemented attention
        cnn_encoded = self.cnn(pos_encoded.transpose((0, 2, 1))) # cnn input (batch_size, channels, seq_len)
        cnn_encoded = cnn_encoded.transpose((0,2,1))

        # attention layer
        attended = self.attn_layer(cnn_encoded)  # shape (batch_size, seq_len, emb_dim)
        encoded = self.binrel_encoder(attended, indices)  ## shape (batch_size, 2, attention_out)
        attended_input = self.attn_out(cnn_encoded, encoded)

        # input for rnn should be (seq_len, batch_size, num_hidden)
        # cnn_encoded = cnn_encoded.transpose((2, 0, 1))
        # experiment - biGRU model
        # state = self.gru.begin_state(batch_size, ctx=data.context) # bidirectional (2*num_layers, batch_size, num_hidden)
        # gru_output, state = self.gru(cnn_encoded, state) # output shape (seq_len, batch_size, 2*num_hidden)
        # gru_output = gru_output.transpose((1, 0, 2)) # (batch_size, seq_len, 2*num_hidden)
        # experiment - cnn only model
        # cnn_encoded = cnn_encoded.transpose((0,2,1))
        # attended_out = self.attn_out(cnn_encoded)
        # attended_out = cnn_encoded * attended_out

        # experiment - cnn + rnn model
        # attended_out = self.attn_out(gru_output.transpose((1,0,2)))
        # attended_out = attended_out * gru_output.transpose((1,0,2))
        # lstm_output = lstm_output.transpose((1, 0, 2))

        out = self.decoder(attended_input)
        return out


def _get_entity_attn(embeddings, entity_encodings, encodings):
    attn_scores = mx.nd.batch_dot(embeddings, entity_encodings.transpose((0,2,1)))
    attn_scores = mx.nd.mean(mx.nd.softmax(attn_scores, axis=1), axis=2)
    attn_input = encodings * mx.nd.expand_dims(attn_scores, axis=2)
    return attn_input


def sliding_window(input, window_size = 3):
#     input shape (batch_size, seq_len, dim)
    batch_size, seq_len, dim = input.shape
    for i in range(input.shape[1]-(window_size-1)):
        input_concat = mx.nd.reshape(input[:, i:i+window_size, :], (batch_size, 1, window_size*dim))
        if i==0:
            output = input_concat
        else:
            output = mx.nd.concat(output, input_concat, dim=1)
    # padding using edges
    output = mx.nd.pad(mx.nd.expand_dims(output, axis=0), mode="edge", pad_width=(0,0,0,0,1,1,0,0))[0]
    return output


class PositionEncoding(HybridBlock):
    def __init__(self, units, dropout=0.1):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.p = nn.Dense(units=units, use_bias=True, flatten=False, activation='relu')

    def forward(self, x, *args):
        x = self.p(x)
        # x = x + self.p.as_in_context(x.context)
        return self.dropout(x)


class Attention(HybridBlock):
    def __init__(self, units=100, num_hidden=300):
        super(Attention, self).__init__()
        self.w = nn.Dense(units=units, use_bias=True, flatten=False, activation='tanh')
        # self.ln = nn.LayerNorm()

    def forward(self, x, entity_encodings, *args):
        # normalized = self.ln(self.w(mx.nd.tanh(x))
        g = self.w(mx.nd.batch_dot(entity_encodings, x.transpose((0,2,1))))
        alpha = mx.nd.mean(mx.nd.softmax(g, axis=2), axis=1)
        alpha = mx.nd.expand_dims(alpha, axis=2)
        out = alpha * x
        return out


def _input_attention(embeddings, inds, input):
    batch_size, seq_len, emb_len = embeddings.shape
    inds = mx.nd.one_hot(inds, seq_len)
    ind_embeddings = mx.nd.batch_dot(inds, embeddings)
    attention_scores = mx.nd.batch_dot(embeddings, ind_embeddings.transpose((0,2,1)))
    attention_scores = mx.nd.mean(mx.nd.softmax(attention_scores, axis=1), axis=2)
    attn_input = input * mx.nd.expand_dims(attention_scores, axis=2)
    return attn_input


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

    
