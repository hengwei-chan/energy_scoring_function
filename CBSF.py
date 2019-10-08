import pickle, pandas as pd, numpy as np, math, functions as f, plots

r=7

with open('data_files/weights/SF' + str(r) + '_weights.pkl', 'rb') as ff:
    w = pickle.load(ff)

with open('data_files/initial_structural_data/casf2016.pkl', 'rb') as ff:
    test_data = pickle.load(ff)

if r==1:
    w = [w]
data_pred = []
for cpx in test_data:
    int_ergies = []
    pairs = [[i[2], i[3]] for i in cpx[0]]
    for dx, dict in enumerate(w):
        p_int = [[pair, dict[pair[-1]][1]] for pair in pairs if (pair[-1] in dict) and (pair[0]<=dict[pair[-1]][0])]
        pairs = [pair for pair in pairs if pair not in [i[0] for i in p_int]]
        #print(dx, len(p_int))
        g1 = len(p_int)*dict['a1b1'][0]+dict['a1b1'][1]
        g2 = len(p_int)*dict['a2b2'][0]/cpx[1]+dict['a2b2'][1]
        g3 = sum([i[1] for i in p_int])
        g_int = g1 + g2 + g3
        int_ergies.append(g_int)
    g = round(sum(int_ergies), 2)
    data_pred.append([cpx[-1], round(g/-1.365, 2)])

df = pd.DataFrame(columns=['#code', 'score'])
df['#code'] = np.array([i[0] for i in data_pred])
df['score'] = np.array([i[1] for i in data_pred])
#df.to_csv('data_files/Score_test/MyScore_SF' + str(r) + '.dat', sep='\t', mode='w')

be = f.readfile("data_files/scoring/SF" + str(r) + ".out", "l")[1:-35]
be = [i.split() for i in be]
be = np.array([[float(i[2]), float(i[3])] for i in be])*-1.365
df2 = pd.DataFrame(columns=['y_pred', 'y_test', 'dev'])
df2['y_pred'] = be[:, 1]
df2['y_test'] = be[:, 0]
df2['err'] = df2['y_test'] - df2['y_pred']

k = df2['err'].values
sigma, mu, aue, mue = np.std(k), np.median(k), abs(k).mean(), abs(k).max()
print([round(i, 4) for i in [aue, mue, mu, sigma]])

#plots.distribution_sigma(k, mu, sigma, 0)
plots.density_scatter(df2['y_test'].values, df2['y_pred'].values, 3)

#print(pairs)
