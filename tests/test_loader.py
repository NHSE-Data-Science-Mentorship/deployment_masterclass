import datetime

from case_regression_example.loader import load_latest_data


def fake_csv() -> str:
    csv = [
        "areaCode,areaName,areaType,date,"
        "newCasesByPublishDate,newCasesBySpecimenDate"
    ]
    codes = {"E06000003": "Redcar and Cleveland", "E06000014": "York"}

    for code, name in codes.items():
        cur_date = datetime.date(2020, 12, 25)
        end_date = datetime.date(2021, 1, 31)
        while cur_date < end_date:
            csv.append(
                ",".join(
                    [code, name, "utla", cur_date.isoformat(), "100", "120"]
                )
            )
            cur_date += datetime.timedelta(days=1)
    return "\n".join(csv)


def test_load(mocker):
    class MockResponse:
        def __init__(self):
            pass

        def text(self):
            return

    mock_requests = mocker.Mock(text=fake_csv())
    with mocker.patch(
        "case_regression_example.loader.requests.get",
        side_effect=lambda _: mock_requests,
    ):
        df = load_latest_data()

    assert list(df.columns) == [
        "areaCode",
        "areaName",
        "areaType",
        "date",
        "newCasesByPublishDate",
        "newCasesBySpecimenDate",
    ]
    assert df["date"].dtype == "datetime64[ns]"
    assert len(df["areaCode"].unique()) == 2
