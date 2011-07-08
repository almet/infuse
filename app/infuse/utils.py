def split_list(original_list, n):
    """Split a list in N equal-sized bits.

    :param original_list: the list to split
    :param n: the number of wanted bits
    """
    if len(original_list) <= n:
        final_list = [original_list, ]
    else:
        final_list = []
        bit_size = len(original_list) / n
        for i in range(n):
            final_list.append(original_list[i*bit_size:(i+1)*bit_size])

    return final_list
