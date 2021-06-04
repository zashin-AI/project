import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
from tensorflow.python.training import optimizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nmb_utils import iter_data, count_parameters

nb_classes = 2

x = np.load('C:\\nmb\\nmb_data\\5s_last_0510\\total_data.npy')
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

y = np.load('C:\\nmb\\nmb_data\\5s_last_0510\\total_label.npy')
print(x.shape, y.shape) # (4536, 128, 862) (4536,) -> (4536, 110336) (4536,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.9, shuffle=True, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_valid = to_categorical(y_valid)

print(x_train.shape, y_train.shape) # (3265, 128, 862) (3265, 2) -> (3265, 110336) (3265,)
print(x_test.shape, y_test.shape)   # (908, 128, 862) (908, 2) -> (908, 110336) (908,)
print(x_valid.shape, y_valid.shape)   # (363, 110336) (363, 2) -> (363, 110336) (363,)

X = tf.compat.v1.placeholder(tf.int32, [128, 862])
Y = tf.compat.v1.placeholder(tf.float32, [None, 2])

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    shape_result = [dynamic[i] if s is None else s for i, s in enumerate(static)]
    print("shape_result \t", shape_result)
    return shape_result

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    sfm = ex / tf.reduce_sum(ex, axis=axis, keepdims=True)
    print("softmax result \t", sfm)
    return sfm

def gelu(x):
    g_elu = 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
    print("g_elu \t", g_elu)
    return g_elu

def gelu2(x):
    g_elu_2 = x * tf.sigmoid(1.702 * x)
    print("g_elu_2 \t", g_elu_2)
    return g_elu_2

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[axis].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        s = tf.reduce_mean(tf.square(x), axis=axis, keepdims=True)
        x = x * tf.rsqrt(s + epsilon)
        x = x*g
        print("norm result \t", x)
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    split_state_result = tf.reshape(x, start + [n, m//n])
    print("split_state_result \t", split_state_result)
    return split_state_result

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    merge_result = tf.reshape(x, start + [a*b])
    print("merge_result \t", merge_result)
    return merge_result

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])), start+[nf])
        print("conv1d layer \t", c)
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def attn(x, scope, n_state, *, past, hparams):
    print("=====att_n_state", n_state, "\t||\tatt_n_head", hparams.n_head, "=====") # =====att_n_state 862||att_n_head 128 =====
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    # assert n_state % hparams.n_head == 0 # 이거 생략해보자.
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        print(" mask_attn_weights \t", w)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        if not hparams.bert:
            w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        print("multihead_attn \t", a)
        return a

    with tf.variable_scope(scope):
        *start, nx = shape_list(x)

        wk = tf.get_variable("k_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        wq = tf.get_variable("q_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        wv = tf.get_variable("v_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        k = tf.einsum("bsf,hef->bhse", x, wk)
        q = tf.einsum("bsf,hef->bhse", x, wq)
        v = tf.einsum("bsf,hef->bhse", x, wv)

        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        wc = tf.get_variable("c_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state*hparams.n_layer)))
        a = tf.einsum("bhse,hef->bsf", a, wc)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu2(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

def set_hparams():
    return HParams(
        n_ctx=128,
        n_embd=862,
        n_head=128,
        n_layer=24,
        n_vocab=512,
        bert=False,
        bert_mask_prob=0.15,
        clf=False,
    )

hparams = set_hparams()

def model(hparams, X, Y=None, past=None, scope='model', reuse=False):
    print("===========================================")
    print("hparams \t",hparams) # n_ctx=128,n_embd=862,n_head=8,n_layer=24,n_vocab=512,bert=False,bert_mask_prob=0.15,clf=False
    print("X \t", X)            # Tensor("Placeholder:0", shape=(128, 862), dtype=int32)
    print("Y \t", Y)            # Tensor("Placeholder_1:0", shape=(?, 2), dtype=float32)
    print("past \t", past)      # None
    print("scope \t", scope)    # model
    print("reuse \t", reuse)    # False
    print("===========================================")

    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        if hparams.bert:
            M = tf.greater(tf.random.uniform([batch, sequence]), hparams.bert_mask_prob)
            M = tf.cast(M, tf.float32)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        wtet = tf.get_variable('wtet', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.0))
        past_length = 0 if past is None else tf.shape(past)[-2]

        h = tf.gather(wte, X)

        if hparams.bert:
            h = h * tf.expand_dims(M, 2)
        else:
            sos = tf.get_variable('sos', [hparams.n_embd],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            sos_tok = tf.ones([batch, 1, hparams.n_embd], dtype=tf.float32) * sos
            h = tf.concat([sos_tok, h[:,:-1,:]], axis=1)

        h += tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)

        # Generative loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        gen_logits = tf.matmul(h_flat, wtet, transpose_b=True)
        gen_logits = tf.reshape(gen_logits, [batch, sequence, hparams.n_vocab])
        results['gen_logits'] = gen_logits
        print("gen_logits \t", gen_logits)

        gen_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gen_logits, labels=X)
        print("softmax : \t", gen_losses)
        if hparams.bert:
            IM = 1.0 - M
            gen_losses = tf.reduce_sum(gen_losses * IM, axis=1) / tf.reduce_sum(IM, axis=1)
            results['gen_loss'] = tf.reduce_mean(gen_losses)
        else:
            results['gen_loss'] = tf.reduce_mean(gen_losses)
        print("mean softmax \t" ,  gen_losses)

        # Classification loss.
        with tf.variable_scope('clf', reuse=reuse):
            classes = shape_list(Y)[1]
            if hparams.clf:
                wclf = tf.get_variable('wclf', [classes, hparams.n_embd],
                                      initializer=tf.random_normal_initializer(stddev=0.0))
            else:
                wclf = tf.zeros([classes, hparams.n_embd], dtype=tf.float32)

        h = tf.reduce_mean(h, axis=1)  # average pool over sequence
        clf_logits = tf.matmul(h, wclf, transpose_b=True)
        clf_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=clf_logits, labels=Y)    # loss
        results['clf_loss'] = tf.reduce_mean(clf_losses)
        print("===clf_losses===\t", results['clf_loss'])

        correct = tf.equal(tf.argmax(clf_logits, -1), tf.argmax(Y, -1))
        results['accuracy'] = tf.reduce_mean(tf.cast(correct, tf.float32)) * 100.0  # accuracy
        print("===correct===\t", results['accuracy'])

        return results

results = model(hparams, X, Y)

'''
shape_result     [128, 862]

norm result      Tensor("model/h0/ln_1/mul_1:0", shape=(128, 862, 862), dtype=float32)
=====att_n_state 862    ||      att_n_head 128 =====
shape_result     [128, 862, 862]
shape_result     [128, 128, 862, 862]
 mask_attn_weights       Tensor("model/h0/attn/sub_2:0", shape=(128, 128, 862, 862), dtype=float32)
.
.
.
softmax :        Tensor("model/SparseSoftmaxCrossEntropyWithLogits/Reshape_2:0", shape=(128, 862), dtype=float32)
mean softmax     Tensor("model/SparseSoftmaxCrossEntropyWithLogits/Reshape_2:0", shape=(128, 862), dtype=float32)
shape_result     [<tf.Tensor 'model/clf/strided_slice:0' shape=() dtype=int32>, 2]
===clf_losses===         Tensor("model/Mean_2:0", shape=(), dtype=float32)
===correct===    Tensor("model/mul_1:0", shape=(), dtype=float32)
'''

print("=================")
print(results)
# {'present': <tf.Tensor 'model/stack:0' shape=(128, 24, 2, 128, 862, 6) dtype=float32>, 
# 'gen_logits': <tf.Tensor 'model/Reshape_1:0' shape=(128, 862, 512) dtype=float32>, 
# 'gen_loss': <tf.Tensor 'model/Mean:0' shape=() dtype=float32>, 
# 'clf_loss': <tf.Tensor 'model/Mean_2:0' shape=() dtype=float32>, 
# 'accuracy': <tf.Tensor 'model/mul_1:0' shape=() dtype=float32>}
print("=================")
print(results['clf_loss'])
print("=================")

def evaluate(sess, evX, evY, X, Y, gen_loss, clf_loss, accuracy, n_batch, desc, permute=False):
    metrics = []
    for xmb, ymb in iter_data(evX, evY, n_batch=n_batch, truncate=True, verbose=True):
        metrics.append(sess.run([gen_loss[0], clf_loss[0], accuracy[0]], {X: xmb, Y: ymb}))
    eval_gen_loss, eval_clf_loss, eval_accuracy = [np.mean(m) for m in zip(*metrics)]
    print(f"{desc} gen: {eval_gen_loss:.4f} clf: {eval_clf_loss:.4f} acc: {eval_accuracy:.2f}")

opti_mizer = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(results['clf_loss'])
training_epochs = 28
batch_size = 10
total_batch = int(len(x_train)/batch_size)  # x_train / 100 = 362
n_batch = 128

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())

    trX = x_train
    trY = y_train
    vaX = x_valid
    vaY = y_valid
    teX = x_test
    teY = y_test
    evaluate(sess, trX[:len(vaX)], trY[:len(vaY)], X, Y, results['gen_loss'], results['clf_loss'], results['accuracy'], n_batch, "train")
    # ValueError: Index out of range using input dim 0; input has only 0 dims for 'strided_slice' (op: 'StridedSlice') with input shapes: [], [1], [1], [1] and with computed input tensors: input[3] = <1>.  
    for epoch in range(training_epochs):
        avg_loss= 0

        for i in range(total_batch):  # 600번 돈다
            start = i * batch_size
            end = start + batch_size

            batch_x, batch_y = x_train[start:end], y_train[start:end]
            feed_dict = {X:batch_x, Y:batch_y}
            l, _ = sess.run([results['clf_loss'], opti_mizer], feed_dict=feed_dict)
            avg_loss += l/total_batch
        print('epoch : ', '%04d' %(epoch+1),
            'loss = {:.9f}'.format(avg_loss))

print("== Done ==")
