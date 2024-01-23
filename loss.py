def calculate_correlation(x):
    return x.T.corrcoef().triu(diagonal=1).norm()