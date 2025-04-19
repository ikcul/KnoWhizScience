#!/usr/bin/env python3
# coding: utf-8

class StringHelper(object):
    def __init__(self):
        pass

    def get_list_from_string(self, str_of_list):
        result_list = list(map(lambda x: int(x), str_of_list.split(",")))
        return result_list

if __name__ == "__main__":
    # Only for test
    str_of_list = '337, 338, 356'
    helper = StringHelper()
    result = helper.get_list_from_string(str_of_list)
    print(result)
    if isinstance(result, list):
        print("correct")
    else:
        print("wrong")
