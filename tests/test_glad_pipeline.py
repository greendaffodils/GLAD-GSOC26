from glad_network.data import load_benchmark_dataset
from glad_network.model import GladConfig, GladNetwork


def test_glad_network_beats_random_guess_threshold():
    data = load_benchmark_dataset(test_size=0.25, random_state=7)
    model = GladNetwork(GladConfig(hidden_layer_sizes=(64,), max_iter=150, random_state=7)).fit(
        data.X_train, data.y_train
    )

    score = model.score(data.X_test, data.y_test)
    assert score > 0.85
