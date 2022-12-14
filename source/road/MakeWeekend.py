from datetime import datetime as dt


def make_weekend(num_date):
    num_year = int(str(num_date)[:4])
    num_month = int(str(num_date)[4:6])
    num_day = int(str(num_date)[6:])

    num_weekday = dt(
        num_year, num_month, num_day
    ).weekday()
    return 1 if num_weekday > 4 else 0
