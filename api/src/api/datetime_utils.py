import datetime


def get_current_date_time() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)
