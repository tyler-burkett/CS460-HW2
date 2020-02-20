from calculations import cos_similarity


class Neighborhood:
    def __init__(self, k):
        self.k = k

    def fit(self, data):
        self.data = data
        self.features = data.columns

    def predict(self, example, feature, default=float("nan")):
        # Copy examples and hide real feature value
        result = example.copy(deep=True)
        result[feature] = float("nan")

        # Calculate distance between example and neighbors
        data_len = len(self.data)
        similarity = [(i, cos_similarity(example, self.data.iloc[i, :], self.features)) for i in range(data_len)]

        # Sort based on the highest score and pick the top k scores
        similarity.sort(reverse=True, key=lambda x: x[1])
        k_nearest = self.data.iloc[similarity[0:self.k, 0], :]

        # Compute prediction based on mean
        avg = k_nearest[feature].mean()
        return avg if avg != float("nan") else default
