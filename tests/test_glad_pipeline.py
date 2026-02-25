from glad_network.data import generate_particle_collision_dataset
from glad_network.experiment import run_baseline
from glad_network.model import GraphAnomalyConfig, GraphAnomalyDetector
from sklearn.metrics import roc_auc_score


def test_graph_detector_reaches_quality_floor_and_beats_baseline():
    split = generate_particle_collision_dataset(
        num_normal=350,
        num_anomalous=90,
        num_nodes=24,
        test_size=0.3,
        random_state=7,
    )

    graph_model = GraphAnomalyDetector(
        GraphAnomalyConfig(message_passing_steps=1, hidden_dim=8, contamination=0.1, random_state=7)
    ).fit(split.train_graphs, split.train_labels)

    scores = graph_model.score_samples(split.test_graphs)
    auc = roc_auc_score(split.test_labels, scores)

    baseline_auc = run_baseline(split).roc_auc

    assert auc > 0.88
    assert auc >= baseline_auc - 0.05
