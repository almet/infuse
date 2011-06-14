def split_list(list, n):
    """Split a list in N equal-sized bits.

    :param list: the list to split
    :param n: the number of wanted bits
    """
    if len(list) <= n:
        final_list = list
    else:
        final_list = []
        bit_size = len(list) / n
        for i in range(n):
            final_list.append(list[i*bit_size:(i+1)*bit_size])

    return final_list
