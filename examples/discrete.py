import numpy as np

from bayes_opt import DiscreteBayesianOptimization



# hadoop/aggregation/large
DATASET1 = {
    # CPU type, cores, GB/core, network
    (1, 2, 2, 1): 523.679,
    (1, 4, 2, 2): 287.641,
    (1, 8, 2, 3): 173.738,
    (2, 2, 2, 1): 630.482,
    (2, 4, 2, 2): 337.147,
    (2, 8, 2, 3): 199.4,
    (3, 2, 4, 1): 581.97,
    (3, 4, 4, 2): 335.27,
    (3, 8, 4, 3): 195.896,
    (4, 2, 4, 1): 661.496,
    (4, 4, 4, 2): 375.389,
    (4, 8, 4, 3): 210.646,
    (5, 2, 8, 1): 644.187,
    (5, 4, 8, 2): 338.26,
    (5, 8, 8, 3): 197.065,
    (6, 2, 8, 1): 664.88,
    (6, 4, 8, 2): 362.476,
    (6, 8, 8, 3): 213.116
}

# hadoop/sort/small
DATASET2 = {
    # CPU type, cores, GB/core, network
    (1, 2, 2, 1): 1614.257,
    (1, 4, 2, 2): 1129.564,
    (1, 8, 2, 3): 860.443,
    (2, 2, 2, 1): 1351.110,
    (2, 4, 2, 2): 178.406,
    (2, 8, 2, 3): 106.945,
    (3, 2, 4, 1): 740.017,
    (3, 4, 4, 2): 624.492,
    (3, 8, 4, 3): 94.593,
    (4, 2, 4, 1): 319.206,
    (4, 4, 4, 2): 165.311,
    (4, 8, 4, 3): 109.993,
    (5, 2, 8, 1): 372.842,
    (5, 4, 8, 2): 173.445,
    (5, 8, 8, 3): 76.688,
    (6, 2, 8, 1): 321.718,
    (6, 4, 8, 2): 169.839,
    (6, 8, 8, 3): 85.824
}

DATASET = DATASET1


def evaluate(cpu_type, core_count, memory_size, network_type):
    return -DATASET[(cpu_type, core_count, memory_size, network_type)]


def main():
    gp_params = {"alpha": 1e-5}
    bo = DiscreteBayesianOptimization(
        evaluate,
        ['cpu_type', 'core_count', 'memory_size', 'network_type'],
        list(DATASET.keys()),
        verbose=1
    )
    # maximize:
    # init_points=2, n_iter=25, acq="ucb", kappa=1, **gp_params
    bo.maximize(init_points=3, n_iter=3, acq="ei", xi=0.1, **gp_params)
    print(bo.res['max'])
    print(bo.res['all'])
    print("optimal:", -bo.res['max']['max_val'])
    print("steps:", len(bo.res['all']['params']))

if __name__ == '__main__':
    main()
