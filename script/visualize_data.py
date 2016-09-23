import numpy as np
import matplotlib.pyplot as plt


def grid_and_legend(loc="upper right"):
    plt.legend(loc=loc)
    ax = plt.gca()
    ax.grid(True)


def plot_elastic_const(const_name):
    plt.plot(pair_interaction['d'], pair_interaction[const_name], marker='^', label='pair')
    plt.plot(pair_interaction['d'], triplet_interaction[const_name], marker='v', label='triplet')
    plt.plot(pair_interaction['d'], pair_interaction[const_name] + triplet_interaction[const_name],
             marker='*', label='total')
    plt.plot(etalon['d'], etalon[const_name], marker='o', label='etalon')
    grid_and_legend()


if __name__ == "__main__":
    etalon = np.loadtxt(
        'quadcell_r1_sigma2_for_comparison.txt',
        dtype={
            'names': ('d', 'Cxx', 'Bxxerr', 'Byy', 'Byyerr', 'Bxxxx', 'Bxxxxerr', 'Byyxx', 'Byyxxerr', 'Bxyxy',
                      'Bxyxyerr', 'Cxxxx', 'Cxxxxerr', 'Cxxyy', 'Cxxyyerr', 'Cxyxy', 'Cxyxyerr', 'nu', 'nuerr'),
            'formats': ('f8',) * 19
        }
    )
    etalon = etalon[etalon['d'] < 5.1]
    data = np.loadtxt(
        'quad_crystal_elastic_consts.out',
        dtype={
            'names': ('d', 'interaction', 'order', 'Cxx', 'Cxxxx', 'Cxxyy', 'Cxyxy', 'Bxxxx', 'Byyxx', 'Bxyxy'),
            'formats': ('f4', 'i1', 'i1', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        }
    )
    pair_interaction = data[(data['interaction'] == 2) & (data['order'] == 5)]
    triplet_interaction = data[(data['interaction'] == 3) & (data['order'] == 5)]

    plt.figure(1, figsize=(12, 8))
    for const_name in ['Cxx', 'Cxxxx', 'Cxxyy', 'Cxyxy']:
       plt.plot(pair_interaction['d'], triplet_interaction[const_name] / pair_interaction[const_name], label=const_name)
    grid_and_legend(loc='lower right')

    plt.figure(2, figsize=(12, 8))
    for const_name in ['Cxx', 'Cxxxx']:
        plt.plot(
            pair_interaction['d'],
            (pair_interaction[const_name] + triplet_interaction[const_name] - etalon[const_name]) / etalon[const_name],
            marker='*', label=const_name
        )
    grid_and_legend()

    fig_num = 3
    for const_name in ['Cxx', 'Cxxxx', 'Cxxyy', 'Cxyxy', 'Bxxxx', 'Byyxx', 'Bxyxy']:
        plt.figure(fig_num, figsize=(12, 8))
        plot_elastic_const(const_name)
        fig_num += 1

    # fig.savefig('triplet_interaction_relative_strength.svg')
    plt.show()
