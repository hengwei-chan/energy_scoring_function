import pandas as pd, numpy as np, pickle, os, sys, json, functions as f
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression, RANSACRegressor
from keras.models import Sequential, Model
from keras.layers import Dense, Layer
from keras.constraints import NonNeg, MaxNorm
from keras import backend as K
from keras.initializers import Constant
import tensorflow as tf

flat = lambda x: [i for j in x for i in j]
toint = lambda j: int(j) if j.isdigit() else int(j[:-1])
tolist = np.ndarray.tolist
unq = lambda x: np.array(np.unique(x))
np.set_printoptions(threshold=sys.maxsize, suppress=True)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def table(path):
    ttm = list(filter(lambda x: (x.startswith('train')) and not (x.endswith('d.csv')), os.listdir(path)))
    tvm = list(filter(lambda x: (x.startswith('test')) and not (x.endswith('d.csv')), os.listdir(path)))
    xmt, colsmt = group(path, ttm)
    xmv, colsmv = group(path, tvm)

    yt, nat, dft = read_table2(path + 'traind.csv')
    yv, nav, dfv = read_table2(path + 'testd.csv')
    colsm = np.unique(np.array([i for j in colsmt for i in j]))
    print(colsm.shape)
    xmt = [make_x_matrix(colsm, i, colsmt[idx]) for idx, i in enumerate(xmt)]
    xmv = [make_x_matrix(colsm, i, colsmv[idx]) for idx, i in enumerate(xmv)]

    mins = [min([i[:, x].min() for i in xmt]) for x in range(xmt[0].shape[1])]
    maxs = [max([max([j for j in i[:, x] if j!=100]) for i in xmt if i[:, x].mean()!=100.0]) for x in range(xmt[0].shape[1])]
    avgs = [mins[i]+(maxs[i]-mins[i])/2 for i in range(xmt[0].shape[1])]
    w = [np.array([avgs]), np.zeros((xmt[0].shape[1],1))]

    #w = optmize_weights_whole(xmt, nat, yt, w)
    check_weights(xmv, nav, yv, dfv)

def check_weights(xmv, nav, yv, df):

    model, model2 = neuralnetwork(len(xmv), xmv[0].shape[1], 2.5)
    name_weights = "txt/weights2/fold1_5_weights.h5"
    model.load_weights(name_weights)
    #weights = model.get_weights()
    #print([i for i in weights[:2]])
    #plots.distribution(weights[2], 'deviation')
    #plots.distribution(weights[3], 'deviation')
    y_pred = model.predict(xmv+[nav])
    make_summary(yv, y_pred.reshape(-1, ), df)

def optmize_weights_whole(xmt, nat, yt, w):
    from keras.callbacks import ModelCheckpoint

    model, model2 = neuralnetwork(len(xmt), xmt[0].shape[1], 3)
    #model.load_weights('fold8_weights.h5')
    model.set_weights(w)

    ranges = list(range(0, xmt[0].shape[0], 500))
    ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 1)] + [[ranges[-1], xmt[0].shape[0]]]
    print(ranges)

    for rdx, r in enumerate(ranges[::-1]):
        name_weights = "fold3" + str(rdx) + "_weights.h5"

        #if rdx > 0:
        #    model.load_weights("fold" + str(rdx-1) + "_weights.h5")
        xmt_r_val = [i[r[0]:r[1]] for i in xmt]
        xmt_r_tr = [np.vstack((i[:r[0]], i[r[1]:])) for i in xmt]

        yt_r_val, nat_r_val = yt[r[0]:r[1]], nat[r[0]:r[1]]
        yt_r_tr, nat_r_tr = np.hstack((yt[:r[0]], yt[r[1]:])), np.hstack((nat[:r[0]], nat[r[1]:]))

        checkpoint = ModelCheckpoint(name_weights, save_best_only=True, mode='auto', save_weights_only=True)
        model.fit(x=xmt_r_tr + [nat_r_tr], y=yt_r_tr, epochs=10, verbose=2, validation_data=(xmt_r_val + [nat_r_val], yt_r_val), batch_size=200,
                  callbacks=[checkpoint], shuffle=True)

    weights = model.get_weights()
    return weights

def neuralnetwork(l, n, k):
    from keras.layers import Add, Subtract, Input, GaussianNoise, Lambda, Activation
    from keras import backend as K
    from keras.utils.generic_utils import get_custom_objects
    from tensorflow.python.ops import clip_ops
    from tensorflow.python.framework import ops
    from keras.optimizers import Adam

    def _to_tensor(x, dtype):
        return ops.convert_to_tensor(x, dtype=dtype)

    def custom_activation(x):
        x = (5.0 * x) + 0.5
        zero = _to_tensor(0., x.dtype.base_dtype)
        one = _to_tensor(1., x.dtype.base_dtype)
        x = clip_ops.clip_by_value(x, zero, one)
        return x

    get_custom_objects().update({'custom_activation': Activation(custom_activation)})

    inputs = [Input(shape=(n,)) for i in range(l)] + [Input(shape=(1,))]
    layer_noise = GaussianNoise(stddev=0.04)
    layer_custom = MyLayer(n, k, trainable=False)
    layer_col = Sequential([Lambda(lambda x: K.sum(x, axis=1)), Lambda(lambda x: K.reshape(x, (-1, 1)))])
    layer_f = Dense(1, use_bias=False, trainable=False)
    layer_n, layer_s = Dense(1), Dense(1)

    xl1 = []
    for rdx in range(l):
        x1n = layer_noise(inputs[rdx])
        x1t = layer_custom(x1n)
        x1a = Activation('custom_activation')(x1t)
        xl1.append(x1a)

    x1s = Add()(xl1)
    xc1 = layer_col(x1s)
    xd1 = Lambda(lambda x: x[0] / x[1])([xc1, inputs[-1]])

    #x1f = layer_f(x1s)
    x1n = layer_n(xc1)
    x1v = layer_s(xd1)
    x = Add()([x1n, x1v])
    out = (x)
    model = Model(inputs=inputs, outputs=out)
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model2 = Model(inputs=inputs, outputs=x1s)
    return model, model2

class MyLayer(Layer):

    def __init__(self, output_dim, k, **kwargs):
        self.output_dim = output_dim
        self.k = k
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(1, input_shape[1]), initializer=Constant(value=self.k), name = 'kernel', constraint = NonNeg())
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        output = -x + self.kernel
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def make_summary(y_test, y_pred, df):
    import plots
    df_sum = pd.DataFrame(columns=['y_pred', 'y_test', 'dev'])
    df_sum2 = pd.DataFrame(columns=['#code', 'score'])
    df_sum2['#code'] = df['cpx']
    df_sum2['score'] = np.round(y_pred/-1.365, 2)
    #df_sum2.to_csv('txt/MyScore.dat', sep='\t', mode='w')

    df_sum['y_pred'] = y_pred
    df_sum['y_test'] = y_test
    df_sum['dev'] = df_sum['y_test'] - df_sum['y_pred']
    #print(df_sum)
    #osvm = EllipticEnvelope(contamination=0.005)

    #y_check = np.array(y_test).reshape(-1, 1)
    #df_sum['y_main'] = osvm.fit(y_check).predict(y_check)
    #y_check = np.array(y_pred).reshape(-1, 1)
    #df_sum['y_main2'] = osvm.fit(y_check).predict(y_check)
    #df_sum = df_sum[df_sum['y_main'] == 1]
    #df_sum = df_sum[df_sum['y_main2'] == 1]
    #df_sum = df_sum.drop(['y_main', 'y_main2'], axis = 1)
    #plots.distribution_sigma(df_sum['dev'])
    plots.density_scatter(df_sum['y_test'].values, df_sum['y_pred'].values)

    k = df_sum['dev'].values
    sigma, mu, aue, mue = np.std(k), np.median(k), abs(k).mean(), abs(k).max()
    r = f.corr_coef(df_sum['y_test'].values, df_sum['y_pred'].values)
    #plots.distribution_sigma(k, mu, sigma)
    print([round(i, 4) for i in [aue, mue, mu, sigma, r]])

def group(path, tables):
    all_cols = []
    all_x = []
    for table in tables:
        X, cols = read_table1(path, table)
        all_cols.append(cols)
        all_x.append(X)
    return all_x, all_cols

def read_table1(dir, table):
    df = pd.read_csv(dir + table, sep=' ')
    df = df.drop(['Unnamed: 0'], axis=1)
    cols = [i for i in df.columns]
    X = df.iloc[:, :]
    del df
    return X, cols

def read_table2(dir):
    df = pd.read_csv(dir, sep=' ')
    df = df.drop(['Unnamed: 0'], axis=1)
    ki, la = df['ki'], df['la']
    return ki, la, df

def make_x_matrix(cols_base, x, cols):
    x = x.values
    w = np.full((x.shape[0], len(cols_base)), 100.0)
    for adx, a in enumerate(cols_base):
        if a in cols:
            w[:, adx] = x[:,cols.index(a)]
    del x
    return w

dir='txt/vwr_130/'
table(dir)
