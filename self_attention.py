import tensorflow as tf
from keras.layers import Input, Layer
from keras.models import Model

from keras import regularizers, initializers

class SelfAttention(Layer):
    def __init__(self, hidden_dim, k_size, Nh, strides=1, padding='SAME', m_for_stem=None, **kwargs):

        self.hidden_dim = hidden_dim
        self.k_size = k_size
        self.Nh = Nh
        self.strides=strides
        self.padding=padding
        self.m_for_stem = m_for_stem
        super(SelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.hidden_dim)

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.W_K = self.add_weight(name='W_K',
                                   shape=(self.Nh, self.hidden_dim // self.Nh, input_dim),
                                   initializer='he_normal',
                                   trainable=True,
                                   regularizer=regularizers.l2(1e-4))

        if self.m_for_stem is None:
            self.W_V = self.add_weight(name='W_V',
                                       shape=(self.Nh, self.hidden_dim // self.Nh, input_dim),
                                       initializer='he_normal',
                                       trainable=True,
                                       regularizer=regularizers.l2(1e-4))
        else:
            self.W_V = self.add_weight(name='W_V',
                                       shape=(self.Nh, self.hidden_dim // self.Nh, input_dim, self.m_for_stem),
                                       initializer='he_normal',
                                       trainable=True,
                                       regularizer=regularizers.l2(1e-4))

        self.W_Q = self.add_weight(name="W_Q",
                                   shape=(self.Nh, self.hidden_dim//self.Nh, input_dim),
                                   initializer='he_normal',
                                   trainable=True,
                                   regularizer=regularizers.l2(1e-4))



        self.Rel_W = self.add_weight(name = "Rel_W",
                                     shape = (self.Nh, 1, self.k_size, (self.hidden_dim//2)//self.Nh),
                                     initializer=initializers.truncated_normal(),
                                     trainable=True,
                                     regularizer=regularizers.l2(1e-4)
                                     )

        self.Rel_H = self.add_weight(name = "Rel_H",
                                     shape = (self.Nh, self.k_size, 1, (self.hidden_dim//2)//self.Nh),
                                     initializer=initializers.truncated_normal(),
                                     trainable=True,
                                     regularizer=regularizers.l2(1e-4)
                                     )


        if self.m_for_stem is not None:
            self.emb_a = self.add_weight(name = "emb_a",
                                         shape = (self.k_size, 1, self.hidden_dim // self.Nh),
                                         initializer='he_normal',
                                         trainable=True,
                                         regularizer=regularizers.l2(1e-4)
                                         )
            self.emb_b = self.add_weight(name = "emb_b",
                                         shape = (1, self.k_size, self.hidden_dim // self.Nh),
                                         initializer='he_normal',
                                         trainable=True,
                                         regularizer=regularizers.l2(1e-4)
                                         )

            self.emb_mix= self.add_weight(name = "emb_mix",
                                          shape = (self.m_for_stem, self.hidden_dim // self.Nh),
                                          initializer='he_normal',
                                          trainable=True,
                                          regularizer=regularizers.l2(1e-4)
                                          )


        super(SelfAttention, self).build(input_shape)


    def call(self, x, **kwarges):

        input_patches = tf.extract_image_patches(x,
                                                 ksizes=[1, self.k_size, self.k_size, 1],
                                                 strides=[1, self.strides, self.strides, 1],
                                                 rates=[1, 1, 1, 1],
                                                 padding=self.padding)

        batch, out_row, out_col, sizes = input_patches.get_shape().as_list()


        input_patches = tf.reshape(input_patches,
                                   [-1, out_row, out_col, self.k_size ** 2, 1,  x.get_shape()[-1]])
        input_patches = tf.tile(input_patches, [1, 1, 1, 1, self.Nh, 1])

        x = tf.reshape(x, [-1, out_row, out_col, 1, x.get_shape()[-1]])
        x = tf.tile(x, [1,1,1, self.Nh, 1])
        Q = tf.einsum('BxyHi,Hoi->BxyHo', x, self.W_Q)
        K = tf.einsum('BxyKHi,Hoi->BxyKHo', input_patches, self.W_K)

        K = tf.transpose(K, [0,1,2,4,3,5])

        if self.m_for_stem is not None:
            emb = tf.add(self.emb_a, self.emb_b)
            emb = tf.reshape(emb, [self.k_size ** 2, self.hidden_dim // self.Nh])
            emb = tf.einsum("Ko,mo->Km", emb, self.emb_mix)
            softmax_emb = tf.nn.softmax(emb, axis=-1)

            V = tf.einsum("Hoim,Km->HoiK", self.W_V, softmax_emb)
            V = tf.transpose(V, [3,0,1,2])
            V = tf.einsum('BxyKHi,KHoi->BxyKHo', input_patches, V)
        else:
            V = tf.einsum('BxyKHi,Hoi->BxyKHo', input_patches, self.W_V)


        V = tf.transpose(V, [0, 1, 2, 4, 3, 5])
        dot_QK = tf.einsum('BxyHKo,BxyHo->BxyHK', K, Q)



        height = tf.tile(self.Rel_H, [1, 1, self.k_size, 1])
        width = tf.tile(self.Rel_W, [1, self.k_size, 1, 1])

        rel_pos = tf.concat([height, width], axis=-1)
        rel_pos = tf.reshape(rel_pos, [self.Nh, self.k_size**2, self.hidden_dim//self.Nh])
        rel_pos = tf.einsum('BxyHo,HKo->BxyHK', Q, rel_pos)
        #rel_pos = tf.divide(rel_pos, tf.sqrt(tf.cast(self.hidden_dim // self.Nh, dtype='float32')))
        dot_QK = tf.add(dot_QK, rel_pos)

        dot_QK = tf.divide(dot_QK, tf.sqrt(tf.cast(self.hidden_dim // self.Nh, dtype='float32')))

        dot_QK = tf.nn.softmax(dot_QK, axis=-1)
        out = tf.einsum('BxyHK,BxyHKo->BxyHKo', dot_QK, V)

        out = tf.reduce_sum(out, axis=-2)

        out = tf.reshape(out, [-1, out_row, out_col, self.hidden_dim])
        return out



if __name__ == "__main__":
    import os

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    input_tensor = Input((320, 320, 16))
    sa = SelfAttention(128, 5, 8, m_for_stem=4)(input_tensor)
    model = Model(input_tensor, sa)
    model.summary()
