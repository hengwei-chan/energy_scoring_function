import pandas as pd, numpy as np, pickle, os, sys, json, functions as f, plots
from keras.models import Sequential, Model
from keras.layers import Dense, Layer, Add, Input, GaussianNoise, Lambda, Activation, Subtract
from keras.constraints import NonNeg, MaxNorm
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

flat = lambda x: [i for j in x for i in j]
toint = lambda j: int(j) if j.isdigit() else int(j[:-1])
tolist = np.ndarray.tolist
unq = lambda x: np.array(np.unique(x))
np.set_printoptions(threshold=sys.maxsize, suppress=True)

def table(path):
    r = 7 #number of intervals
    ttm = list(filter(lambda x: (x.startswith('train')) and not (x.endswith('d.csv')), os.listdir(path)))
    tvm = list(filter(lambda x: (x.startswith('test')) and not (x.endswith('d.csv')), os.listdir(path)))
    xmt = group(path, ttm)
    xmv = group(path, tvm)

    yt, nat, dft = read_table2(path + 'traind.csv')
    yv, nav, dfv = read_table2(path + 'testd.csv')
    mins = [min([i[:, x].min() for i in xmt]) for x in range(xmt[0].shape[1])]
    maxs = [max([max([j for j in i[:, x] if j!=100]) for i in xmt if i[:, x].mean()!=100.0]) for x in range(xmt[0].shape[1])]
    rl = [(maxs[i]-mins[i])/(r+1) for i in range(xmt[0].shape[1])]

    weights = []
    for x in range(r):
        dist = [mins[i] + rl[i]*(x+1) for i in range(xmt[0].shape[1])]
        weights.append(np.array([dist]))
    weights += [np.zeros((xmt[0].shape[1], 1))] * len(weights)
    weights += [np.array([[-0.03]]), np.array([0.69])]*r + [np.array([[1.1]]), np.array([-11.7])]*r
    #optmize_weights_whole(xmt, nat, yt, weights, r)
    check_weights(xmv, nav, yv, dfv, r)

def check_weights(xmv, nav, yv, dfv, r):

    model, model2 = neuralnetwork(len(xmv), xmv[0].shape[1], r)
    name_weights = 'data_files/weights/weights' + str(r) + '.h5'
    model.load_weights(name_weights)
    y_pred = model.predict(xmv+[nav])

    df = pd.DataFrame(columns=['#code', 'score'])
    df['#code'] = dfv['complex_name']
    df['score'] = np.round(y_pred/-1.365 , 2)
    #df.to_csv('data_files/Score_test/MyScore_NN_SF' + str(r) + '.dat', sep='\t', mode='w')

    df2 = pd.DataFrame(columns=['y_pred', 'y_test', 'dev'])
    df2['y_pred'] = y_pred.reshape(-1)
    df2['y_test'] = yv
    df2['err'] = df2['y_test'] - df2['y_pred']

    k = df2['err'].values
    sigma, mu, aue, mue = np.std(k), np.median(k), abs(k).mean(), abs(k).max()
    print([round(i, 4) for i in [aue, mue, mu, sigma]])

    plots.distribution_sigma(k, mu, sigma, r)
    plots.density_scatter(df2['y_test'].values, df2['y_pred'].values, int((r-1)/2))

def optmize_weights_whole(xmt, nat, yt, w, r):
    from keras.callbacks import ModelCheckpoint

    model, model2 = neuralnetwork(len(xmt), xmt[0].shape[1], r)
    #model.load_weights('weights8.h5')
    model.set_weights(w)

    ranges = list(range(0, xmt[0].shape[0], 500))
    ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 1)] + [[ranges[-1], xmt[0].shape[0]]]

    for rdx, r in enumerate(ranges[::-1]):
        name_weights = "weights" + str(rdx) + ".h5"

        xmt_r_val = [i[r[0]:r[1]] for i in xmt]
        xmt_r_tr = [np.vstack((i[:r[0]], i[r[1]:])) for i in xmt]

        yt_r_val, nat_r_val = yt[r[0]:r[1]], nat[r[0]:r[1]]
        yt_r_tr, nat_r_tr = np.hstack((yt[:r[0]], yt[r[1]:])), np.hstack((nat[:r[0]], nat[r[1]:]))

        checkpoint = ModelCheckpoint(name_weights, save_best_only = True,  mode='auto', save_weights_only=True)
        model.fit(x=xmt_r_tr + [nat_r_tr], y=yt_r_tr, epochs=150, verbose=2, validation_data=(xmt_r_val + [nat_r_val], yt_r_val), batch_size=200,
                  callbacks=[checkpoint], shuffle=True)

def neuralnetwork(l, n, k):

    def _to_tensor(x, dtype):
        return ops.convert_to_tensor(x, dtype=dtype)

    def custom_activation(x):
        x = (10.0 * x) + 0.5
        zero = _to_tensor(0., x.dtype.base_dtype)
        one = _to_tensor(1., x.dtype.base_dtype)
        x = clip_ops.clip_by_value(x, zero, one)
        return x

    get_custom_objects().update({'custom_activation': Activation(custom_activation)})

    inputs = [Input(shape=(n,)) for i in range(l)] + [Input(shape=(1,))]
    layer_noise = GaussianNoise(stddev=0.04)
    layers_custom = [MyLayer(n) for i in range(k)]
    layer_col = Sequential([Lambda(lambda x: K.sum(x, axis=1)), Lambda(lambda x: K.reshape(x, (-1, 1)))])
    layers_f = [Dense(1, use_bias=False) for i in range(k)]
    layers_n, layers_s = [Dense(1) for i in range(k)], [Dense(1) for i in range(k)]

    xl = [[]]*k
    for rdx in range(l):
        x1n = layer_noise(inputs[rdx])
        xt = [l(x1n) for ldx, l in enumerate(layers_custom)]
        xa = [Activation('custom_activation')(x) for x in xt]
        xl = [x + [xa[xdx]] for xdx, x in enumerate(xl)]

    xls = [Add()(l) for l in xl]
    xs = [xls[0]] + [Subtract()([xls[i], xls[i-1]]) for i in range(1, len(xls))]
    xc = [layer_col(x) for x in xs]
    xd = [Lambda(lambda x: x[0] / x[1])([t, inputs[-1]]) for t in xc]

    xf = Add()([layers_f[tdx](t) for tdx, t in enumerate(xs)])
    xn = Add()([layers_n[tdx](t) for tdx, t in enumerate(xc)])
    xv = Add()([layers_s[tdx](t) for tdx, t in enumerate(xd)])

    x = Add()([xf, xn, xv])
    out = (x)
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model2 = Model(inputs=inputs, outputs=xc)
    return model, model2

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(1, input_shape[1]), initializer='zeros', name = 'kernel', constraint = NonNeg())
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        output = -x + self.kernel
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def group(path, tables):
    all_x = []
    for table in tables:
        with open(path+table, 'r') as fl:
            arr = json.load(fl)
        all_x.append(np.array(arr))
    print(all_x[0].shape)
    input_tables = []
    for r in range(130):
        input_tables.append(np.vstack([i[:, r, :] for i in all_x]))
    return input_tables

def read_table2(dir):
    df = pd.read_csv(dir, sep=' ')
    df = df.drop(['Unnamed: 0'], axis=1)
    be, la = df['exp_binding_energy'], df['num_ligand_atoms']
    return be, la, df

dir='data_files/NNdata/'
table(dir)
