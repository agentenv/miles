"""segment_ids (CompactionRL cross-segment GAE) flow from Sample.metadata into train shards. No Ray needed."""

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from tests.fast.ray.rollout.conftest import make_args, make_sample


def _convert(samples):
    args = make_args()
    return args, convert_samples_to_train_data(args, samples, {}, None, None)


def test_no_segment_ids_key_when_absent():
    _, data = _convert([make_sample(index=0), make_sample(index=1)])
    assert "segment_ids" not in data


def test_segment_ids_emitted_and_sharded():
    a = make_sample(index=0, response_length=4)
    a.metadata = {"segment_ids": [0, 0, 1, 1]}
    b = make_sample(index=1, response_length=3)
    _, data = _convert([a, b])
    assert data["segment_ids"] == [[0, 0, 1, 1], [0, 0, 0]]
    args = make_args(balance_data=False)
    shards = split_train_data_by_dp_raw(args, data, dp_size=1)
    assert shards[0]["segment_ids"] == [[0, 0, 1, 1], [0, 0, 0]]


def test_tokens_after_and_gae_length_emitted_and_sharded():
    a = make_sample(index=0, response_length=4)
    a.metadata = {"tokens_after": 7, "gae_length": 11}
    b = make_sample(index=1, response_length=3)
    b.metadata = {"tokens_after": 0}
    _, data = _convert([a, b])
    assert data["tokens_after"] == [7, 0]
    assert data["gae_length"] == [11, 3]  # missing -> own response length
    assert "segment_ids" not in data
    args = make_args(balance_data=False)
    shards = split_train_data_by_dp_raw(args, data, dp_size=1)
    assert shards[0]["tokens_after"] == [7, 0] and shards[0]["gae_length"] == [11, 3]


def test_no_tokens_after_key_when_absent():
    _, data = _convert([make_sample(index=0)])
    assert "tokens_after" not in data and "gae_length" not in data


def test_positive_example_success_emitted_and_sharded():
    a = make_sample(index=0)
    a.metadata = {"success": True}
    b = make_sample(index=1)
    b.metadata = {"is_correct": False}
    c = make_sample(index=2)  # missing -> -1, rejected when the LM loss uses it
    _, data = _convert([a, b, c])
    assert data["positive_example_success"] == [1, 0, -1]
    args = make_args(balance_data=False)
    shards = split_train_data_by_dp_raw(args, data, dp_size=1)
    assert shards[0]["positive_example_success"] == [1, 0, -1]


def test_no_positive_example_success_key_when_absent():
    _, data = _convert([make_sample(index=0)])
    assert "positive_example_success" not in data
