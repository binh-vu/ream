from ream.dataset_helper import (
    AbsoluteRangeSelection,
    DatasetQuery,
    PercentageRangeSelection,
)


def test_dataset_query():
    assert DatasetQuery.from_string("wt250") == DatasetQuery(
        "wt250", {"": PercentageRangeSelection(0, 100)}, False, None, []
    )
    assert DatasetQuery.from_string("wt250[train[:100]]") == DatasetQuery(
        "wt250", {"train": AbsoluteRangeSelection(0, 100)}, False, None, []
    )
    assert DatasetQuery.from_string("wt250[:100]") == DatasetQuery(
        "wt250", {"": AbsoluteRangeSelection(0, 100)}, False, None, []
    )
    assert DatasetQuery.from_string("wt250[90%:100%]") == DatasetQuery(
        "wt250", {"": PercentageRangeSelection(90, 100)}, False, None, []
    )
    assert DatasetQuery.from_string("wt250[90%:]") == DatasetQuery(
        "wt250", {"": PercentageRangeSelection(90, 100)}, False, None, []
    )
    assert DatasetQuery.from_string("wt250[:90%]") == DatasetQuery(
        "wt250", {"": PercentageRangeSelection(0, 90)}, False, None, []
    )
    assert DatasetQuery.from_string("wt250[train[:100]]{shuffle}") == DatasetQuery(
        "wt250", {"train": AbsoluteRangeSelection(0, 100)}, True, None, []
    )
    assert DatasetQuery.from_string("wt250[train[:100]]{shuffle(42)}") == DatasetQuery(
        "wt250", {"train": AbsoluteRangeSelection(0, 100)}, True, 42, []
    )
    assert DatasetQuery.from_string(
        "wt250[train[:100]]{shuffle(42), no-unk-col}"
    ) == DatasetQuery(
        "wt250", {"train": AbsoluteRangeSelection(0, 100)}, True, 42, ["no-unk-col"]
    )
    assert DatasetQuery.from_string("wt250{no-unk-col}") == DatasetQuery(
        "wt250", {"": PercentageRangeSelection(0, 100)}, False, None, ["no-unk-col"]
    )
