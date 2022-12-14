def make_season_month(num_month):
    if (num_month // 3) >= 4:
        return 0

    return num_month // 3
