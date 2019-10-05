import numpy as np, matplotlib.pyplot as plt

def density_scatter(x, y, k):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=10, edgecolor='', color = 'blue')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    textstr1 = ['A. SF1 (Basic Version)',
               'B. SF3',
               'C. SF5',
               'D. CBSF (SF7)']
    textstr2 = 'Scoring Power:'
    textstr3 = ['\n'.join(['MAE = 1.480', r'$\mathrm{median} = 0.0$', 'R = 0.781', r'$\sigma = 1.881$']),
                '\n'.join(['MAE = 1.272', r'$\mathrm{median} = 0.191$', 'R = 0.839', r'$\sigma = 1.614$']),
                '\n'.join(['MAE = 1.127', r'$\mathrm{median} = 0.218$', 'R = 0.870', r'$\sigma = 1.462$']),
                '\n'.join(['MAE = 1.163', r'$\mathrm{median} = 0.014$', 'R = 0.864', r'$\sigma = 1.491$'])]

    # ['\n'.join(['MAE = 1.504', r'$\mathrm{median} = -0.111$', 'R = 0.780', r'$\sigma = 1.896$']),
    # '\n'.join(['MAE = 1.276', r'$\mathrm{median} = 0.061$', 'R = 0.836', r'$\sigma = 1.638$']),
    # '\n'.join(['MAE = 1.091', r'$\mathrm{median} = 0.215$', 'R = 0.877', r'$\sigma = 1.434$']),
    # '\n'.join(['MAE = 1.052', r'$\mathrm{median} = 0.111$', 'R = 0.886', r'$\sigma = 1.378$']),

    textstr4 = 'Ranking Power:'
    textstr5 = [ '\n'.join(['SP = 0.661', r'$\tau = 0.581$', 'PI = 0.634']),
                '\n'.join(['SP = 0.741', r'$\tau = 0.653$', 'PI = 0.578']),
                '\n'.join(['SP = 0.782', r'$\tau = 0.702$', 'PI = 0.625']),
                '\n'.join(['SP = 0.831', r'$\tau = 0.756$', 'PI = 0.757'])]

    #['\n'.join(['SP = 0.637', r'$\tau = 0.553$', 'PI = 0.641']),
    # '\n'.join(['SP = 0.735', r'$\tau = 0.656$', 'PI = 0.655']),
    # '\n'.join(['SP = 0.812', r'$\tau = 0.727$', 'PI = 0.658']),
    # '\n'.join(['SP = 0.850', r'$\tau = 0.778$', 'PI = 0.773']),

    ax.text(0.05, 0.95, textstr2, transform=ax.transAxes, fontsize=12, verticalalignment='top', fontname="Arial",
            fontweight='bold')
    ax.text(0.05, 0.89, textstr3[k], transform=ax.transAxes, fontsize=12, verticalalignment='top', fontname="Arial")
    ax.text(0.60, 0.25, textstr4, transform=ax.transAxes, fontsize=12, verticalalignment='top', fontname="Arial",
            fontweight='bold')
    ax.text(0.60, 0.19, textstr5[k], transform=ax.transAxes, fontsize=12, verticalalignment='top', fontname="Arial")
    plt.xlim(-18, 0)
    plt.ylim(-18, 0)
    ax.set_xlabel('Experimental Binding Free Energy (kcal/mol)', fontname="Arial", fontsize=12)
    ax.set_ylabel('Calculated Binding Free Energy (kcal/mol)', fontname="Arial", fontsize=12)
    plt.title(textstr1[k], fontname="Arial Bold", fontsize=12, pad = 15.0, fontweight = 'bold')
    ax.set_aspect('equal', 'box')
    plt.xticks(np.arange(-18, max(x) + 2, 2.0))
    plt.savefig('data_files/figures/SF_' + str(k*2+1) + '.pdf', transparent=True)
    plt.show()

def distribution_sigma(col, mu, sigma, r):
    from matplotlib.mlab import normpdf
    fig, axs = plt.subplots(1, 1, tight_layout=True)
    n, bins, patches = axs.hist(col, bins=31, edgecolor='black', fill=True, color='skyblue')
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    y = normpdf(bincenters, mu, sigma)
    axs.plot(bincenters, y*(max(n)/max(y)), linewidth=3, color = 'r', alpha = 0.5,)
    plt.xticks(size=18)
    plt.yticks(size=18)
    axs.set_xlabel(r'$w_{1j}$, kcal/mol', fontname="Arial", fontsize=18)
    axs.set_ylabel('Count', fontname="Arial", fontsize=18)
    plt.savefig('data_files/figures/' + str(r) + 'dist.png', transparent=True)
    plt.show()

