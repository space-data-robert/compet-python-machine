def make_time_hour(num_hour):
    arr_bins = [3, 8, 18, 22]

    for num, num_bins in enumerate(arr_bins):
        if num_hour > num_bins:
            continue
        return num
    return 0
