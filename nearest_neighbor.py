import math


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


class Neighborhood:
    def __init__(self, k, prediction_func=None):
        self.k = k
        self.prediction = prediction_func

    def fit(self, data):
        self.data = data
        self.features = data.columns

    def predict(self, examples):
        # Copy examples
        results = examples.copy(deep=True)

        # Find k nearest neighbors to each example, and predict missing features
        for example in results:
            # Calculate distance between example and neighbors
            data_len = len(self.data)
            similarity = [(i, cos_similarity(example, self.data.iloc[i, :])) for i in range(data_len)]

            # Sort based on the highest score and pick the top k scores
            similarity.sort(reverse=True, key=lambda x: x[1])
            k_nearest = self.data.iloc[similarity[0:self.k, 0], :]

            # Go through each feature and fill in missing values
            for feature in example.columns:
                if example[feature] == float("nan"):
                    examples[feature] = self.prediction(feature, k_nearest)

        return results
