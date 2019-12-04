import pandas as pd, numpy as np, os, sys, pickle, custom_layers
from keras.models import Sequential, Model
from keras.layers import Dense, Add, Input, Lambda, Activation, Concatenate, Subtract, GaussianNoise
from keras import backend as K
import tensorflow as tf
from keras import regularizers
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import Constant

flat = lambda x: [i for j in x for i in j]
toint = lambda j: int(j) if j.isdigit() else int(j[:-1])
tolist = np.ndarray.tolist
unq = lambda x: np.array(np.unique(x))
np.set_printoptions(threshold=sys.maxsize, suppress=True)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def make_ranges(k, n):
    ranges = list(range(0, k, n))
    ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 1)] + [[ranges[-1], k]]
    if ranges[-1][1] - ranges[-1][0] < n / 2:
        ranges = ranges[:-2] + [[ranges[-2][0], ranges[-1][1]]]
    return ranges

def find_min_max_ints(dim, x1, x2, empty_v, max_v, r):
    
    mins1 = [min([i[:, x].min() for i in x1]) for x in range(dim)]
    maxs1 = [max([max([j for j in i[:, x] if j!=empty_v]) if i[:, x].mean()!=empty_v else max_v for i in x1]) for x in range(dim)]
    mins2 = [min([i[:, x].min() for i in x2]) for x in range(dim)]
    maxs2 = [max([max([j for j in i[:, x] if j!=empty_v]) if i[:, x].mean()!=empty_v else max_v for i in x2]) for x in range(dim)]
    mins = [min(mins1[idx], mins2[idx]) for idx in range(dim)]
    maxs = [max(maxs1[idx], maxs2[idx]) if max(maxs1[idx], maxs2[idx]) != max_v else min(maxs1[idx], maxs2[idx]) for idx in range(dim)]

    rl = [(maxs[i]-mins[i])/(r+1) for i in range(dim)]
    print([round((maxs[i]-mins[i]), 3) for i in range(dim)])
    weights = []
    for x in range(r):
        dist = [mins[i] + rl[i]*(x+1) for i in range(dim)]
        weights.append(dist)

    return weights

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

    r=4
    empty_v, max_v = 20.0, 1.0
    w = []
    for k in range(2):
        w.append(find_min_max_ints(ref_tables_m[k][0].shape[1], ref_tables_m[k], test_tables_m[k], empty_v, max_v, r))
 
    train = [ref_tables_m, nar, yr]
    test = [test_tables_m, nat, yt]
    optmize_weights_whole(w, train, test, r)
    #check_weights(test, dft, apts_list, r)

def check_weights(test, dfv, apts, r):

    len1, len2, dim1, dim2 = len(test[0][0]), len(test[0][1]), test[0][0][0].shape[1], test[0][1][0].shape[1]
    model,  custom_layersl = neuralnetwork([len1, len2], [dim1, dim2], [1, 0], 0.1, r)
    name_weights = 'data_files/weights/' + str(r) + '10_final_weights.h5'
    model.load_weights(name_weights)
    #with open("data_files/weights/wn_8_weights.pkl", "rb") as fl:
    #    w = pickle.load(fl)
    #model.set_weights(w)

    w = model.get_weights()
    #print(w)
    print([i.shape for i in w])
    w_apts = {}
    for idx, i in enumerate(apts[0]):
        values_d, values_w = [], []
        for k in range(r):
            d = w[k][0][idx]*5
            we = w[k+2*r][idx][0]
            values_d.append(d if d <= 5.0 else 5.0)
            values_w.append(we)
        w_apts[i] =[values_d, values_w]

    for idx, i in enumerate(apts[1]):
        values_d, values_w = [], []
        for k in range(r):
            d = w[k+r][0][idx]*5
            we = w[k+3*r][idx][0]
            values_d.append(d if d <= 5.0 else 5.0)
            values_w.append(we)
        w_apts[i] =[values_d, values_w]
    n1 = []
    for k in range(r):
        n1.append([w[4*r+2*k][0][0], w[4*r+2*k+1][0]])
    w_apts['n1'] = n1
    n2 = []
    for k in range(r):
        n2.append(w[6*r+k][0][0])
    w_apts['n2'] = n2

        #w_apts['n2'] = [w[6*k][0][0]]
    
    #print(w_apts['n1'], w_apts['n2'])
    #print([i.shape for i in w])
    
    with open('data_files/weights/' + str(r) + '_cutoffs.pkl', "wb") as fl:
        pickle.dump(w_apts, fl)
 
    X_test, y_test = test[0][0] + test[0][1] + [test[1]], test[2]
    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test)
    print(score)

    df = pd.DataFrame(columns=['#code', 'score'])
    df['#code'] = dfv['complex_name']
    df['score'] = np.round(y_pred / -1.365, 2)
    df.to_csv('data_files/MyScore_SF3.dat', sep='\t', mode='w')

    df2 = pd.DataFrame(columns=['y_pred', 'y_test', 'dev'])
    df2['y_pred'] = y_pred.reshape(-1)
    df2['y_test'] = y_test
    df2['err'] = df2['y_test'] - df2['y_pred']

    k = df2['err'].values
    sigma, mu, aue, mue = np.std(k), np.median(k), abs(k).mean(), abs(k).max()
    print([round(i, 4) for i in [aue, mue, mu, sigma]])

    #plots.distribution_sigma(k, mu, sigma, 0)


def optmize_weights_whole(weights, train, test, r):
    len1, len2 = len(train[0][0]), len(train[0][1]) #, len(train[0][2])
    dim1, dim2 = train[0][0][0].shape[1], train[0][1][0].shape[1] #, train[0][2][0].shape[1]
    weights_dir = "data_files/weights/"

    def step_train(w_t, w_t_prev, ep, step_new, step_old, lr):
        print(w_t)
        x, na, y, nb = train[0], train[1], train[2], 500
        ranges = make_ranges(x[0][0].shape[0], nb)

        model, custom_layersl = neuralnetwork([len1, len2], [dim1, dim2], w_t, lr, r)
        if 'initial' not in step_old:
            name_weights0 = weights_dir + str(r) + ''.join(
                [str(j) for j in w_t_prev]) + '_' + step_old + '_weights.pkl'
            with open(name_weights0, 'rb') as fl:
                weights = pickle.load(fl)
            model.set_weights(weights)
        else:
            name_weights0 = weights_dir + str(r) +'_00_initial_weights.h5'
            model.load_weights(name_weights0)
        name_weights1 = weights_dir + str(r) + ''.join([str(j) for j in w_t]) + "_" + step_new + '_weights.pkl'
        for ndx, n in enumerate(ranges):
            x1_val, x2_val = [i[n[0]:n[1]] for i in x[0]], [i[n[0]:n[1]] for i in x[1]]
            y_val, na_val = y[n[0]:n[1]], na[n[0]:n[1]]
            x1_tr = [np.vstack((i[ranges[0][0]:n[0]], i[n[1]:ranges[-1][-1]])) for i in x[0]]
            x2_tr = [np.vstack((i[ranges[0][0]:n[0]], i[n[1]:ranges[-1][-1]])) for i in x[1]]
            #x3_tr = [np.vstack((i[ranges[0][0]:n[0]], i[n[1]:ranges[-1][-1]])) for i in x[2]]
            y_tr, na_tr = np.hstack((y[ranges[0][0]:n[0]], y[n[1]:ranges[-1][-1]])), np.hstack(
                (na[ranges[0][0]:n[0]], na[n[1]:ranges[-1][-1]]))
            X = x1_tr + x2_tr + [na_tr]
            X_val = x1_val + x2_val + [na_val]
            model.fit(x=X, y=y_tr, epochs=ep, verbose=0,
                      validation_data=(X_val, y_val), batch_size=100, shuffle=True,
                      callbacks=[checkpoint])
            X_test, y_test = test[0][0] + test[0][1] + [test[1]], test[2]
            score = model.evaluate(X_test, y_test, verbose=0)
            if step_new == 'final2':
                model.save_weights(name_weights1[:-4] + str(ndx) + '.h5')
                if ndx == 0:
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

    print(len1, len2, dim1, dim2)
    model, cutoff_layers = neuralnetwork([len1, len2], [dim1, dim2], [0, 0], 0.2, r)
    print([i.shape for i in model.get_weights()])
    for x in range(len(cutoff_layers)):
        for idx, cutoff_layer in enumerate(cutoff_layers[x]):
            cutoff_layer.set_weights([np.array([weights[x][idx]])])


    X, y = flat(train[0]) + [train[1]], train[2]
    X_test, y_test = flat(test[0]) + [test[1]], test[2]

    name_weights0 = weights_dir + str(r) + '_00_initial_weights.h5'
    checkpoint = ModelCheckpoint(name_weights0, save_best_only=True, mode='auto', save_weights_only=True)
    model.fit(x=X, y=y, epochs=50, verbose=2, validation_split=0.1, batch_size=200, callbacks=[checkpoint], shuffle=True)

    n1, n2 = 20, 15
    lr1, lr2 = 0.0005, 0.0003
    model = step_train([0, 1], [0, 0], n1, 'main', 'initial', lr1)
    model = step_train([1, 0], [0, 1], n1, 'main', 'main', lr1)
    model = step_train([1, 1], [1, 0], n1, 'main', 'main', lr2)

    model = step_train([0, 1], [1, 1], n2, 'final', 'main', lr2)
    model = step_train([1, 0], [0, 1], n2, 'final2', 'final', lr2)

    score = model.evaluate(X_test, y_test)
    print(score)

def first_layers_i(l, layer, layer_a,  inputs, k):
    layer_noise = GaussianNoise(stddev=0.001)
    xl = [[]]*k
    for rdx in range(l):
        xn = layer_noise(inputs[rdx])
        xt = [l(xn) for ldx, l in enumerate(layer)]
        xa = [layer_a(x) for x in xt]
        xl = [x + [xa[xdx]] for xdx, x in enumerate(xl)]
    xs = [Add()(l) for l in xl]
    xs1 = [xs[0]] + [Subtract()([xs[i], xs[i - 1]]) for i in range(1, len(xs))]
    return xs1

def neuralnetwork(l, n, w_t, lr, k):
    print(k)
    get_custom_objects().update({'custom_activation1': Activation(custom_layers.custom_activation1),'custom_activation2': Activation(custom_layers.custom_activation2)})
    rl = regularizers.l1(0.001)
    inputs = flat([[Input(shape=(n[x],)) for i in range(l[x])] for x in range(len(l))]) + [Input(shape=(1,))]
    layer_col = Sequential([Lambda(lambda x: K.sum(x, axis=1)), Lambda(lambda x: K.reshape(x, (-1, 1)))])

    layers_ni, layers_si = [Dense(1, kernel_initializer=Constant(-0.03)) for i in range(k)], [Dense(1, use_bias=False) for i in range(k)]

    layersi_cutoff = [[custom_layers.MyLayer(dim) for i in range(k)] for dim in n]
    layersi_cutoff_non_trainable = [[custom_layers.MyLayer(dim, trainable=False) for i in range(k)] for dim in n]

    layersi_f = [[Dense(1, use_bias=False, kernel_initializer='zeros', kernel_regularizer=rl) for i in range(k)] for x in range(len(l))]
    layersi_f_non_trainable =[[Dense(1, use_bias=False, trainable=False) for i in range(k)] for x in range(len(l))]


    if w_t[0] == 0:
        layers11 = layersi_cutoff_non_trainable
        layer_a = Activation('custom_activation2')
    else:
        layers11 = layersi_cutoff
        layer_a = Activation('custom_activation1')
    if w_t[1] == 0:
        layers21 = layersi_f_non_trainable
    else:
        layers21 = layersi_f
    print(l, len(inputs[l[0]:l[1] + l[0]]))

    x1s = first_layers_i(l[0], layers11[0], layer_a, inputs[:l[0]], k)
    x2s = first_layers_i(l[1], layers11[1], layer_a, inputs[l[0]:l[1] + l[0]], k)

    x1f = Add()([layers21[0][tdx](t) for tdx, t in enumerate(x1s)])
    x2f = Add()([layers21[1][tdx](t) for tdx, t in enumerate(x2s)])

    xs = [Concatenate(axis=1)([x1s[i], x2s[i]]) for i in range(k)]
    xc = [layer_col(x) for x in xs]
    xd = [Lambda(lambda x: x[0] / x[1])([t, inputs[-1]]) for t in xc]
    xn = Add()([layers_ni[tdx](t) for tdx, t in enumerate(xc)])
    xv = Add()([layers_si[tdx](t) for tdx, t in enumerate(xd)])

    x = Add()([x1f, x2f, xn, xv])
    out = (x)
    model = Model(inputs=inputs, outputs=out)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer=adam)
    model2 = Model(inputs=inputs, outputs=xc)
    return model, layers11

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
