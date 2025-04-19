#!/usr/bin/env python3
# coding: utf-8

import os

class PrintSpace(object):

    def print_tmp_space(self, folder='/tmp'):
        # get /tmp infomation
        statvfs = os.statvfs(folder)

        # calculate space
        total_space = statvfs.f_frsize * statvfs.f_blocks
        free_space = statvfs.f_frsize * statvfs.f_bfree
        used_space = total_space - free_space

        # convert byte to MB
        total_space_mb = total_space / (1024 * 1024)
        free_space_mb = free_space / (1024 * 1024)
        used_space_mb = used_space / (1024 * 1024)

        # print used and free space
        space_string = f"Total space in {folder}: {total_space_mb:.2f} MB\nUsed space in {folder}: {used_space_mb:.2f} MB\nFree space in {folder}: {free_space_mb:.2f} MB"
        print(space_string)
        return space_string

    def get_directory_size(self, directory):
        total_size = 0
        # foreach all file and folder
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                # calculate only if the file exist. Do not calculate symbol link
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size

    def print_directory_size(self, directory):
        size_in_bytes = self.get_directory_size(directory)
        size_in_mb = size_in_bytes / (1024 * 1024)
        space_string = f"Total size of '{directory}': {size_in_mb:.2f} MB"
        print(space_string)
        return space_string

if __name__ == "__main__":
    print_space = PrintSpace()
    print_space.print_tmp_space()
    print_space.print_directory_size('/tmp/pipeline')