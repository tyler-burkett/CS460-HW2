import math
import statistics as stat

def not_nan(a, b, feature):
    return math.isnan(a[feature] * b[feature])


def cos_similarity(a, b, features):
    ab_sum = sum(a[feature] * b[feature] for feature in features if not_nan(a, b, feature))
    a_square = sum(a[feature] ** 2 for feature in features if not_nan(a, b, feature))
    b_square = sum(b[feature] ** 2 for feature in features if not_nan(a, b, feature))
    try:
        return ab_sum / (math.sqrt(a_square) * math.sqrt(b_square))
    except ArithmeticError:
        return 0


def mean_square_error(a, b, features):
    used_features = [not_nan(a, b, feature) for feature in features]
    return stat.mean((b[feature] - a[feature])**2 for feature in features if used_features)
