from __future__ import annotations

import pytest
from pydantic import ValidationError

from featurepipe.registry import DefinitionError, Registry
from featurepipe.spec import FeatureSpec, FeatureView


def spec(**overrides) -> FeatureSpec:
    base = {
        "name": "txn_count_24h",
        "agg": "count",
        "window": "24h",
        "dtype": "int",
        "empty_window": "zero",
    }
    return FeatureSpec(**{**base, **overrides})


def view(**overrides) -> FeatureView:
    base = {
        "name": "v",
        "entity": "account_id",
        "event_time": "event_at",
        "recorded_time": "recorded_at",
        "ttl": "36h",
        "materialize_every": "10m",
        "max_staleness": "20m",
        "features": (spec(),),
    }
    return FeatureView(**{**base, **overrides})


def test_a_count_takes_no_column():
    with pytest.raises(ValidationError, match="count takes no column"):
        spec(column="amount")


def test_a_sum_needs_one():
    with pytest.raises(ValidationError, match="needs a column"):
        spec(name="amount_sum_24h", agg="sum", dtype="float")


def test_a_mean_is_a_float():
    with pytest.raises(ValidationError, match="a mean is a float"):
        spec(name="amount_mean_7d", agg="mean", column="amount", dtype="int")


def test_a_distinct_count_is_an_int():
    with pytest.raises(ValidationError, match="is an int"):
        spec(name="countries_7d", agg="distinct_count", column="country", dtype="float")


@pytest.mark.parametrize("agg", ["mean", "max"])
def test_there_is_no_mean_or_max_of_nothing(agg):
    with pytest.raises(ValidationError, match="empty_window"):
        spec(
            name=f"amount_{agg}_7d",
            agg=agg,
            column="amount",
            dtype="float",
            empty_window="zero",
        )


def test_a_feature_name_has_to_be_usable_as_a_column():
    with pytest.raises(ValidationError, match="not usable as a column name"):
        spec(name="txn count 24h")


def test_an_unknown_key_in_a_definition_is_not_ignored():
    with pytest.raises(ValidationError):
        FeatureSpec(
            name="x", agg="count", window="1h", dtype="int", empty_window="zero", windw="2h"
        )


def test_an_empty_window_means_zero_for_a_count_and_unknown_for_a_mean():
    assert spec().empty_value == 0
    assert spec(name="s", agg="sum", column="amount", dtype="float").empty_value == 0.0
    mean = spec(name="m", agg="mean", column="amount", dtype="float", empty_window="unknown")
    assert mean.empty_value is None


def test_a_duration_that_does_not_parse_is_refused():
    with pytest.raises(ValidationError):
        view(ttl="soon")


def test_a_view_needs_a_feature():
    with pytest.raises(ValidationError, match="nothing to serve"):
        view(features=())


def test_a_feature_declared_twice_in_one_view_is_refused():
    with pytest.raises(ValidationError, match="declared twice"):
        view(features=(spec(), spec()))


def test_the_longest_window_and_the_columns_come_off_the_feature_set():
    both = view(
        features=(
            spec(),
            spec(name="amount_sum_7d", agg="sum", column="amount", dtype="float", window="7d"),
        )
    )
    assert both.longest_window_seconds == 7 * 86400
    assert both.columns == ("amount",)


def test_narrowing_a_window_moves_the_field_but_not_the_key():
    original = view()
    narrowed = original.with_features((spec(window="12h"),))
    assert narrowed.version == original.version
    assert narrowed.features[0].field != original.features[0].field


def test_adding_a_feature_moves_neither_the_key_nor_an_existing_field():
    original = view()
    extended = original.with_features(
        (
            *original.features,
            spec(name="amount_sum_7d", agg="sum", column="amount", dtype="float"),
        )
    )
    assert extended.version == original.version
    assert extended.features[0].field == original.features[0].field


def test_renaming_the_recorded_time_column_rotates_the_keyspace():
    assert view(recorded_time="ingested_at").version != view().version


def test_the_empty_policy_is_part_of_a_features_identity():
    assert spec(empty_window="unknown").field != spec(empty_window="zero").field


def test_a_feature_name_may_not_appear_in_two_views(registry):
    clash = registry.view("merchant_activity").with_features(
        (registry.view("account_activity").features[0],)
    )
    with pytest.raises(ValueError, match="has to be unique across views"):
        Registry(views=(registry.view("account_activity"), clash))


def test_the_registry_names_the_file_that_will_not_load(tmp_path):
    (tmp_path / "broken.toml").write_text('name = "x"\nentity = ', encoding="utf-8")
    with pytest.raises(DefinitionError, match=r"broken\.toml"):
        Registry.from_directory(tmp_path)


def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(DefinitionError, match=r"no \.toml definitions"):
        Registry.from_directory(tmp_path)


def test_asking_for_a_view_that_is_not_there_lists_the_ones_that_are(registry):
    with pytest.raises(DefinitionError, match="account_activity, merchant_activity"):
        registry.view("nope")


def test_the_shipped_definitions_load(registry):
    assert [v.name for v in registry.views] == ["account_activity", "merchant_activity"]
    assert registry.owner_of("merchant_txn_count_1h").name == "merchant_activity"
