# Functions to properly read in different files

import numpy as np
import os, sys


def read_matrix_file(filename, delimiter = None, name_column = 0, data_start_column = 1, value_dtype = float, header = '#', strip_names = '"', column_name_replace = None, row_name_replace = None, unknown_value = 'NA', nan_value = 0):
    '''
    Reads in text file and returns names of rows, names of colums and data matrix

    Parameters
    ----------
    filename : string
        Location of file
    delimiter : string
        Delimiter between columns
    name_column: int
        Column in which name is placed, None means that no names are given
    data_start_column:
        Column
    '''
    if delimiter is None:
        if os.path.splitext(filename)[1] == '.csv':
            delimiter = ','


    f = open(filename, 'r').readlines()
    columns, rows, values = None, [], []
    if header is not None:
        if f[0][:len(header)] == header:
            columns = f[0].strip(header).strip().replace(strip_names,'').split(delimiter)

    start = 0
    if columns is not None:
        start = 1

    if name_column is None:
        nc = 0
    else:
        nc = name_column
    
    for l, line in enumerate(f):
        if l >= start:
            line = line.strip().split(delimiter)
            rows.append(line[name_column].strip(strip_names))
            ival = np.array(line[data_start_column:])
            values.append(ival)

    if name_column is None:
        rows = np.arange(len(rows)).astype(str)
    if column_name_replace is not None:
        for c, col in enumerate(columns):
            columns[c] = col.replace(column_name_replace[0], row_name_replace[1])

    if row_name_replace is not None:
        for r, row in enumerate(rows):
            rows[r] = row.replace(row_name_replace[0], row_name_replace[1])
    try:
        values = np.array(values, dtype = value_dtype)
    except:
        ValueError
        values = np.array(values)
        print("matrix could not be converted to floats")

    if (values == np.nan).any():
        print('ATTENTION nan values in data matrix.', filename)
        if nan_value is not None:
            print('nan values replaced with', nan_value)
            values = np.nan_to_num(values, nan = nan_value)

    if columns is not None:
        if len(columns) > np.shape(values)[1]:
            columns = columns[-np.shape(values)[1]:]
        columns = np.array(columns)
    
    return np.array(rows), columns, values



