'''check for phrase'''

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def length_check(strs, min_l, max_l):
    if min_l <= len(strs) <= max_l:
        return True
    else:
        return False
