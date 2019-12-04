import pandas as pd, numpy as np, os, sys, custom_layers, pickle
from keras.models import Sequential, Model
from keras.layers import Dense, Add, Input, Lambda, Activation, Concatenate, GaussianNoise
from keras import backend as K
import tensorflow as tf
from keras import regularizers
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

flat = lambda x: [i for j in x for i in j]
toint = lambda j: int(j) if j.isdigit() else int(j[:-1])
tolist = np.ndarray.tolist
unq = lambda x: np.array(np.unique(x))
np.set_printoptions(threshold=sys.maxsize, suppress=True)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def find_min_max_ints(dim, x1, x2, empty_v, max_v):
    mins1 = [min([i[:, x].min() for i in x1]) for x in range(dim)]
    maxs1 = [max([max([j for j in i[:, x] if j!=empty_v]) if i[:, x].mean()!=empty_v else max_v for i in x1]) for x in range(dim)]
    mins2 = [min([i[:, x].min() for i in x2]) for x in range(dim)]
    maxs2 = [max([max([j for j in i[:, x] if j!=empty_v]) if i[:, x].mean()!=empty_v else max_v for i in x2]) for x in range(dim)]
    mins = [min(mins1[idx], mins2[idx]) for idx in range(dim)]
    maxs = [max(maxs1[idx], maxs2[idx]) if max(maxs1[idx], maxs2[idx]) != max_v else min(maxs1[idx], maxs2[idx]) for idx in range(dim)]
    avgs = [mins[i]+(maxs[i]-mins[i])/2 for i in range(dim)]
    return avgs

def make_ranges(k, n):
    ranges = list(range(0, k, n))
    ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 1)] + [[ranges[-1], k]]
    if ranges[-1][1] - ranges[-1][0] < n / 2:
        ranges = ranges[:-2] + [[ranges[-2][0], ranges[-1][1]]]
    return ranges

def table(path):
    def make_inputs(name):
        tables, apts = [], []
        for i in range(2):
            tables_i = list(filter(lambda x: x.startswith(name + '_' + str(i + 1)), os.listdir(path)))
            tables_i, apt = group(path, tables_i)
            tables.append(tables_i)
            apts.append(apt)
        return tables, apts

    ref_tables, ref_apts = make_inputs('ref_train')
    test_tables, test_apts = make_inputs('test')
    yr, nar, dfr = read_table2(path + 'ref_traind.csv')
    yt, nat, dft = read_table2(path + 'testd.csv')

    gen_tables_m, ref_tables_m, test_tables_m = [], [], []
    apts_list = []
    for k in range(2):
        apts = np.unique(np.array([i for j in ref_apts[k] + test_apts[k] for i in j]))
        ref_tables_m.append([make_x_matrix(apts, i, ref_apts[k][idx]) for idx, i in enumerate(ref_tables[k])])
        test_tables_m.append([make_x_matrix(apts, i, test_apts[k][idx]) for idx, i in enumerate(test_tables[k])])
        apts_list.append(apts)

    empty_v, max_v = 20.0, 1.0
    w1 = [np.array([find_min_max_ints(ref_tables_m[0][0].shape[1], ref_tables_m[0], test_tables_m[0], empty_v, max_v)])]
    w2 = [np.array([find_min_max_ints(ref_tables_m[1][0].shape[1], ref_tables_m[1], test_tables_m[1], empty_v, max_v)])]

    train = [ref_tables_m, nar, yr]
    test = [test_tables_m, nat, yt]
    optmize_weights([w1, w2], train, test)

def optmize_weights(weights, train, test):

    len1, len2, dim1, dim2 = len(train[0][0]), len(train[0][1]), train[0][0][0].shape[1], train[0][1][0].shape[1]
    weights_dir = "data_files/weights/"

    def step_train(w_t, w_t_prev, ep, step_new, step_old, lr):
        print(w_t)
        x1, x2, na, y, nb = train[0][0], train[0][1], train[1], train[2], 500
        ranges = make_ranges(x1[0].shape[0], nb)
         
        model, custom_layersl = neuralnetwork([len1, len2], [dim1, dim2], w_t, lr)
        if 'initial' not in step_old:
            name_weights0 = weights_dir + 'base_' + ''.join([str(j) for j in w_t_prev]) + '_' + step_old + '_weights.pkl'
            with open(name_weights0, 'rb') as fl:
                weights = pickle.load(fl)
            model.set_weights(weights)
        else:
            name_weights0 = weights_dir + 'base_00_initial_weights.h5'
            model.load_weights(name_weights0)
        name_weights1 = weights_dir + 'base_' + ''.join([str(j) for j in w_t]) + "_" + step_new + '_weights.pkl'
        for ndx, n in enumerate(ranges):
            x1_val, x2_val = [i[n[0]:n[1]] for i in x1], [i[n[0]:n[1]] for i in x2]
            y_val, na_val = y[n[0]:n[1]], na[n[0]:n[1]]
            x1_tr = [np.vstack((i[ranges[0][0]:n[0]], i[n[1]:ranges[-1][-1]])) for i in x1]
            x2_tr = [np.vstack((i[ranges[0][0]:n[0]], i[n[1]:ranges[-1][-1]])) for i in x2]
            y_tr, na_tr = np.hstack((y[ranges[0][0]:n[0]], y[n[1]:ranges[-1][-1]])), np.hstack(
                (na[ranges[0][0]:n[0]], na[n[1]:ranges[-1][-1]]))
            model.fit(x=x1_tr + x2_tr + [na_tr], y=y_tr, epochs=ep, verbose=0,
                      validation_data=(x1_val + x2_val + [na_val], y_val), batch_size=100, shuffle=True, callbacks=[checkpoint])
            X_test, y_test = test[0][0] + test[0][1] + [test[1]], test[2]
            score = model.evaluate(X_test, y_test, verbose=0)
            if step_new == 'final2':
                if ndx==0:
                    best_score = score
                    model.save_weights(name_weights1[:-3] + 'h5')
                    with open(name_weights1, 'wb') as fl:
                        pickle.dump(model.get_weights(), fl)
                else:
                    if score < best_score:
                        model.save_weights(name_weights1[:-3] + 'h5')
                        best_score = score 
                        with open(name_weights1, 'wb') as fl:
                            pickle.dump(model.get_weights(), fl)
            else:
                with open(name_weights1, 'wb') as fl:
                    pickle.dump(model.get_weights(), fl)
                model.save_weights(name_weights1[:-3] + 'h5')
            print(score)
        return model

    model, layer_custom = neuralnetwork([len1, len2], [dim1, dim2], [0, 0], 0.4)
    layer_custom[0].set_weights(weights[0])
    layer_custom[1].set_weights(weights[1])
 
    X, y = train[0][0] + train[0][1] + [np.array(train[1])], train[2]
    X_test, y_test = test[0][0] + test[0][1] + [test[1]], test[2]

    name_weights0 = weights_dir + 'base_00_initial_weights.h5'
    checkpoint = ModelCheckpoint(name_weights0, save_best_only=True, mode='auto', save_weights_only=True)
    model.fit(x=X, y = y, epochs=50, verbose=2, validation_split=0.1, batch_size=200, callbacks=[checkpoint], shuffle=True)

    n1, n2 = 50, 40
    lr = 0.0005
    model = step_train([0, 1], [0, 0], n1, 'main', 'initial', lr)
    model = step_train([1, 0], [0, 1], n1, 'main', 'main', lr)
    model = step_train([1, 1], [1, 0], n1, 'main', 'main', lr)

    model = step_train([0, 1], [1, 1], n2, 'final', 'main', lr)
    model = step_train([1, 0], [0, 1], n2, 'final2', 'final', lr)

    score = model.evaluate(X_test, y_test)
    print(score)

def first_layers(l, layer, layer_a, inputs):
    layer_noise = GaussianNoise(stddev=0.008)
    xl = []
    for rdx in range(l):
        xn = layer_noise(inputs[rdx])
        x1t = layer(xn)
        #x1t = layer(inputs[rdx])
        xt = layer_a(x1t)
        xl.append(xt)
    xs = Add()(xl)
    return xs

def neuralnetwork(l, n, w_t, lr):
    get_custom_objects().update({'custom_activation1': Activation(custom_layers.custom_activation1),'custom_activation2': Activation(custom_layers.custom_activation2)})
 
    inputs = [Input(shape=(n[0],)) for i in range(l[0])] + [Input(shape=(n[1],)) for i in range(l[1])] + [Input(shape=(1,))]
    layer_col = Sequential([Lambda(lambda x: K.sum(x, axis=1)), Lambda(lambda x: K.reshape(x, (-1, 1)))])
    layer_n, layer_s = Dense(1), Dense(1, use_bias=False)

    layer_cutoff1, layer_cutoff2 = custom_layers.MyLayer(n[0]), custom_layers.MyLayer(n[1])
    layer_cutoff1_non_trainable, layer_cutoff2_non_trainable = custom_layers.MyLayer(n[0], trainable =False), custom_layers.MyLayer(n[1], trainable = False)

    layer_f1 = Dense(1, use_bias=False, kernel_initializer='zeros')
    layer_f2 = Dense(1, use_bias=False, kernel_initializer='zeros')
    layer_f1_non_trainable = Dense(1, use_bias=False, kernel_initializer='zeros', trainable = False)
    layer_f2_non_trainable = Dense(1, use_bias=False, kernel_initializer='zeros', trainable = False)

    if w_t[0] == 0:
        layer11, layer12 = layer_cutoff1_non_trainable, layer_cutoff2_non_trainable
        layer_a = Activation('custom_activation2')
    else:
        layer11, layer12 = layer_cutoff1, layer_cutoff2
        layer_a = Activation('custom_activation1')

    if w_t[1] == 0:
        layer21, layer22 = layer_f1_non_trainable, layer_f2_non_trainable
    else:
        layer21, layer22 = layer_f1, layer_f2


    x1s = first_layers(l[0], layer11, layer_a, inputs[:l[0]])
    x2s = first_layers(l[1], layer12, layer_a, inputs[l[0]:-1])

    x1f = layer21(x1s)
    x2f = layer22(x2s)
 
    xs = Concatenate(axis=1)([x1s, x2s])
    xc = layer_col(xs)
    xd = Lambda(lambda x: x[0] / x[1])([xc, inputs[-1]])
    
    xn = layer_n(xc)
    xv = layer_s(xd)
    x = Add()([x1f, x2f, xn, xv])
    out = (x)

    model = Model(inputs=inputs, outputs=out)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model, [layer11, layer12]

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
    X = df.iloc[:, :]/5
    del df
    return X, cols

def read_table2(dir):
    df = pd.read_csv(dir, sep=' ')
    df = df.drop(['Unnamed: 0'], axis=1)
    be, la = df['exp_binding_energy'], df['num_ligand_atoms']
    return be, la, df

def make_x_matrix(cols_base, x, cols):
    x = x.values
    w = np.full((x.shape[0], len(cols_base)), 100.0)
    for adx, a in enumerate(cols_base):
        if a in cols:
            w[:, adx] = x[:,cols.index(a)]
    del x
    return w

dir='data_files/NNdata/'
table(dir)
