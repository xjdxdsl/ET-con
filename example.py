import pandas
from learn_structure_cache import hill_climbing_cache
from export import get_adjacency_matrix_from_et

# Load ASIA dataframe
file_name = "data/asia.csv"
data = pandas.read_csv(file_name)
var_classes = [['yes','no'] for _ in range(8)]

# ----LEARNING BAYESIAN NETWORKS WITH BOUNDED TREEWIDTH---- #
# Learn elimination tree (ET) with hc-et, using a tw bound of 3 and BIC as the objective score
et = hill_climbing_cache(data, metric = 'bic', tw_bound = 3, custom_classes=var_classes)
# Learn ET with hc-et-poly, using a tw bound of 3 and BIC as the objective score
et2 = hill_climbing_cache(data, metric = 'bic', tw_bound = 3, custom_classes=var_classes, add_only=True)

# Get adjacency matrix of the Bayesian network encoded by the ET et
adj_mat = get_adjacency_matrix_from_et(et)



