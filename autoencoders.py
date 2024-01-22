import tensorflow as tf


# Helper layers
# ========================================================================================================
class _TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(_TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2


# Available encoders
# =======================================================================================================
def rectified_tanh(x):
    return tf.keras.activations.relu(tf.keras.activations.tanh(x))


def abs_norm(x):
    return tf.abs(x) / tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)


def rectified_tanh_norm(x):
    ht = rectified_tanh(x)

    return ht / tf.reduce_sum(ht, axis=-1, keepdims=True)


def soft_rectified_tanh(x, gamma=10.0):
    return (1 / gamma) * tf.math.log(1 + tf.math.exp(gamma * tf.keras.activations.tanh(x)))


class _Encoder(tf.keras.Model):
    def __init__(self, bottleneck_dim, asc, latent_sparsity):
        super(_Encoder, self).__init__()

        activation = 'softmax' if asc else soft_rectified_tanh
        self.bottleneck_layer = tf.keras.layers.Dense(
            bottleneck_dim,
            activation=activation,
            activity_regularizer=tf.keras.regularizers.l1(latent_sparsity) if latent_sparsity > 0.0 else None
        )

    def call(self, x):
        x_hat = self.bottleneck_layer(x)

        return x_hat


class _DenseEncoder(_Encoder):
    def __init__(self, bottleneck_dim, hidden_dims, asc, latent_sparsity):
        super(_DenseEncoder, self).__init__(bottleneck_dim, asc, latent_sparsity)

        self.encoder = tf.keras.models.Sequential()

        if hidden_dims is None:
            hidden_dims = []

        for dim in hidden_dims:
            self.encoder.add(tf.keras.layers.Dense(dim, activation=tf.keras.layers.LeakyReLU(0.02)))

    def call(self, x):
        encoded = self.encoder(x)
        z = super().call(encoded)

        return z


class _ConvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_sizes, num_filters):
        super().__init__()
        
        assert len(kernel_sizes) == len(num_filters), "kernel_sizes and num_filters must have the same length"

        self.conv_layers = []
        for kernel_size, num_filter in zip(kernel_sizes, num_filters):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(num_filter, kernel_size=kernel_size, padding='same', activation='relu'))

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.linear_dim_reduce = tf.keras.layers.Dense(1)

    def call(self, x):
        intensities = tf.expand_dims(x, axis=-1)

        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_outputs.append(conv_layer(intensities))

        x = self.concat(conv_outputs)
        x = self.linear_dim_reduce(x)
        x = tf.squeeze(x, axis=-1)

        return x


class _ConvolutionalDenseEncoder(_DenseEncoder):
    def __init__(self, bottleneck_dim, kernel_sizes, num_filters, hidden_dims, asc, latent_sparsity):
        assert len(kernel_sizes) == len(num_filters), "kernel_sizes and num_filters must have the same length"

        super(_ConvolutionalDenseEncoder, self).__init__(bottleneck_dim, hidden_dims, asc, latent_sparsity)

        self.conv_block = _ConvolutionalBlock(kernel_sizes, num_filters)

    def call(self, x):
        x = self.conv_block(x)
        z = super().call(x)

        return z


class _TransformerEncoder(_Encoder):
    def __init__(self, bottleneck_dim, d_model, nhead, num_layers, asc, latent_sparsity):
        super(_TransformerEncoder, self).__init__(bottleneck_dim, asc, latent_sparsity)

        self.linear_intensity = tf.keras.layers.Dense(d_model)

        self.transformer_encoder_layers = [_TransformerEncoderLayer(d_model, nhead, d_model * 2) for _ in range(num_layers)]

    def call(self, x, training=True):
        x_transformed = self.linear_intensity(x)

        x_transformed = tf.expand_dims(x_transformed, axis=1)

        for layer in self.transformer_encoder_layers:
            x_transformed = layer(x_transformed, training=training)

        x_encoded = tf.squeeze(x_transformed, axis=1)

        z = super().call(x_encoded)

        return z


class _ConvolutionalTransformerEncoder(_TransformerEncoder):
    def __init__(self, bottleneck_dim, d_model, nhead, num_layers, kernel_sizes, num_filters, asc, latent_sparsity):
        super(_ConvolutionalTransformerEncoder, self).__init__(bottleneck_dim, d_model, nhead, num_layers, asc, latent_sparsity)

        self.conv_block = _ConvolutionalBlock(kernel_sizes, num_filters)

    def call(self, x, training=False):
        x = self.conv_block(x)
        z = super().call(x, training)

        return z


# Available decoders
# =======================================================================================================
class _Decoder(tf.keras.Model):
    def _get_endmembers(self):
        # get input dimension of first layer
        bottleneck_dim = self.trainable_weights[0].shape[0]
        one_hot_latent_vectors = tf.eye(bottleneck_dim)

        endmembers = self.call(one_hot_latent_vectors)

        return endmembers.numpy()


class _LinearUnmixingDecoder(_Decoder):
    def __init__(self, output_dim, use_bias):
        _Decoder.__init__(self)

        self.linear_layer = tf.keras.layers.Dense(
            output_dim, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg(), use_bias=use_bias)

    def call(self, x):
        x = self.linear_layer(x)

        return x

    def _get_endmembers(self):
        return self.linear_layer.trainable_variables[0].numpy()


class _FanBiLinearUnmixingDecoder(_LinearUnmixingDecoder):
    def __init__(self, output_dim, use_bias):
        _LinearUnmixingDecoder.__init__(self, output_dim, use_bias)

    def call(self, x):
        linear_part = self.linear_layer(x)

        # Bilinear part
        bottleneck_dim = self.linear_layer.weights[0]

        bottleneck_dim_multiplied = x[..., :, None] * bottleneck_dim[:, ...]

        bilinear_terms = tf.math.multiply(tf.expand_dims(bottleneck_dim_multiplied, 1), tf.expand_dims(bottleneck_dim_multiplied, 2))

        shape = tf.shape(bilinear_terms)
        i, j = tf.meshgrid(tf.range(shape[1]), tf.range(shape[2]), indexing='ij')
        mask = (i - j) > 1

        mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)

        upper_triangular = bilinear_terms * tf.dtypes.cast(mask, bilinear_terms.dtype)

        bilinear_part = tf.reduce_sum(upper_triangular, [1, 2])

        return linear_part + bilinear_part

    def _get_endmembers(self):
        return _Decoder._get_endmembers(self)


class _PPNMUnmixingDecoder(_LinearUnmixingDecoder):
    def __init__(self, output_dim, use_bias):
        _LinearUnmixingDecoder.__init__(self, output_dim, use_bias)
        self.b = tf.Variable(0.0, trainable=True)

    def call(self, x):
        linear_part = self.linear_layer(x)

        # Bilinear part
        bilinear_part = self.b + linear_part ** 2

        return linear_part + bilinear_part


class _GeneralUnmixingDecoder(_Decoder):
    def __init__(self, output_dim, hidden_dims):
        _Decoder.__init__(self)

        self.decoder = tf.keras.models.Sequential()

        if hidden_dims is None:
            hidden_dims = []
        for dim in hidden_dims:
            self.decoder.add(tf.keras.layers.Dense(dim, activation=tf.keras.layers.LeakyReLU(0.02)))

        self.decoder.add(tf.keras.layers.Dense(output_dim, activation=tf.keras.layers.LeakyReLU(0.02), kernel_constraint=tf.keras.constraints.NonNeg()))

    def call(self, x):
        x = self.decoder(x)
        return x


class _PostnonlinearUnmixingDecoder(_LinearUnmixingDecoder, _GeneralUnmixingDecoder):
    def __init__(self, output_dim, use_bias, hidden_dims):
        _LinearUnmixingDecoder.__init__(self, output_dim, use_bias)
        _GeneralUnmixingDecoder.__init__(self, output_dim, hidden_dims)

    def call(self, x):
        x_linear = super().call(x)
        x_postnonlinear = self.decoder(x_linear)
        return x_postnonlinear

    def _get_endmembers(self):
        return _Decoder._get_endmembers(self)


class _AdditivePostnonlinearUnmixingDecoder(_PostnonlinearUnmixingDecoder):
    def call(self, x):
        x_linear = self.linear_layer(x)
        x_postnonlinear = self.decoder(x_linear)
        x_hat = tf.keras.layers.Add()([x_linear, x_postnonlinear])

        return x_hat

    def _get_endmembers(self):
        return _Decoder._get_endmembers(self)


# Factory method for creating decoders
# ------------------------------------
def get_decoder(decoder_type, output_dim, *, use_bias, hidden_dims):
    if decoder_type == 'linear':
        return _LinearUnmixingDecoder(output_dim, use_bias)
    elif decoder_type == 'postnonlinear':
        return _PostnonlinearUnmixingDecoder(output_dim, use_bias, hidden_dims)
    elif decoder_type == 'additivepostnonlinear':
        return _AdditivePostnonlinearUnmixingDecoder(output_dim, use_bias, hidden_dims)
    elif decoder_type == 'fanbilinear':
        return _FanBiLinearUnmixingDecoder(output_dim, use_bias)
    elif decoder_type == 'ppnm':
        return _PPNMUnmixingDecoder(output_dim, use_bias)
    elif decoder_type == 'general':
        return _GeneralUnmixingDecoder(output_dim, hidden_dims)
    else:
        raise ValueError(f'Unknown decoder type: {decoder_type}')


# Available autoencoders
# =======================================================================================================
class _UnmixingAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(_UnmixingAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat

    def get_abundances(self, data):
        return self.encoder(data)

    def get_endmembers(self):
        return self.decoder._get_endmembers()


# Autoencoders
# ------------
class DenseAE(_UnmixingAE):
    def __init__(self, input_dim, bottleneck_dim, decoder_type, *, encoder_hidden_dims=None, decoder_hidden_dims=None, use_bias=False, asc=True, latent_sparsity=0.0):
        encoder = _DenseEncoder(bottleneck_dim, encoder_hidden_dims, asc, latent_sparsity)
        decoder = get_decoder(decoder_type, input_dim, use_bias=use_bias, hidden_dims=decoder_hidden_dims)

        super(DenseAE, self).__init__(encoder, decoder)


class ConvolutionalAE(_UnmixingAE):
    def __init__(self, input_dim, bottleneck_dim, decoder_type, *, kernel_sizes, num_filters, encoder_hidden_dims=None, decoder_hidden_dims=None, use_bias=False, asc=True, latent_sparsity=0.0):
        encoder = _ConvolutionalDenseEncoder(bottleneck_dim, kernel_sizes, num_filters, encoder_hidden_dims, asc, latent_sparsity)
        decoder = get_decoder(decoder_type, input_dim, use_bias=use_bias, hidden_dims=decoder_hidden_dims)

        super(ConvolutionalAE, self).__init__(encoder, decoder)


class TransformerAE(_UnmixingAE):
    def __init__(self, input_dim, bottleneck_dim, decoder_type, *, d_model, num_heads, num_layers, decoder_hidden_dims=None, use_bias=False, asc=True, latent_sparsity=0.0):
        encoder = _TransformerEncoder(bottleneck_dim, d_model, num_heads, num_layers, asc, latent_sparsity)
        decoder = get_decoder(decoder_type, input_dim, use_bias=use_bias, hidden_dims=decoder_hidden_dims)

        super(TransformerAE, self).__init__(encoder, decoder)


class ConvolutionalTransformerAE(_UnmixingAE):
    def __init__(self, input_dim, bottleneck_dim, decoder_type, *, d_model, num_heads, num_layers, kernel_sizes, num_filters, decoder_hidden_dims=None, use_bias=False, asc=True, latent_sparsity=0.0):
        encoder = _ConvolutionalTransformerEncoder(bottleneck_dim, d_model, num_heads, num_layers, kernel_sizes, num_filters, asc, latent_sparsity)
        decoder = get_decoder(decoder_type, input_dim, use_bias=use_bias, hidden_dims=decoder_hidden_dims)

        super(ConvolutionalTransformerAE, self).__init__(encoder, decoder)



# Available training losses
# =======================================================================================================
cos = tf.losses.CosineSimilarity()
mse = tf.losses.MeanSquaredError()


def SAD(x, y):
    return tf.math.acos(-1*cos(x, y))


def PCC(y_true, y_pred):
    # Compute means
    mean_y_true = tf.reduce_mean(y_true, axis=-1, keepdims=True)
    mean_y_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)

    # Compute Pearson’s Correlation Coefficient (r)
    numerator = tf.reduce_sum((y_true - mean_y_true) * (y_pred - mean_y_pred), axis=-1)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true - mean_y_true), axis=-1) *
                          tf.reduce_sum(tf.square(y_pred - mean_y_pred), axis=-1))

    # Ensure denominator isn't zero
    denominator = tf.where(denominator == 0, 1e-10, denominator)

    r = numerator / denominator

    # Convert to loss.
    return 1 - r


def MSE_SAD(beta=1.0):
    def loss(y_true, y_pred):
        return beta * mse(y_true, y_pred) + SAD(y_true, y_pred)
    return loss
