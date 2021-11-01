import requests
import io
import pandas as pd

URL = (
    "https://api.coronavirus.data.gov.uk/v2/data?"
    "areaType=utla&metric=newCasesByPublishDate&metric=newCasesBySpecimenDate&format=csv"
)


def load_latest_data() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(requests.get(URL).text), parse_dates=["date"])
    return df
