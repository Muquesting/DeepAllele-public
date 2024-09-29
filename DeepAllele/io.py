# Functions to properly read in different files

import numpy as np
import os, sys

# check if str is boolean, a list, or a number, otherwise return string back
import ast
def check(inbool):
    if inbool == 'True' or inbool == 'TRUE' or inbool == 'true':
        return True
    elif inbool == 'False' or inbool == 'FALSE' or inbool == 'false':
        return False
    elif inbool == 'None' or inbool == 'NONE' or inbool == 'none':
        return None
    elif "[" in inbool or "(" in inbool:
        return ast.literal_eval(inbool)
    else:
        inbool = numbertype(inbool)
    return inbool

def inputkwargs_from_string(string, definer='=', separater = '+'):
    '''
    Takes long string as input and seperates it at "separater"
    Returns dictionary with keys from before definer and values after definer
    '''
    kwargs = {}
    if definer in string:
        if separater in string:
            adjpar = string.split(separater)
        else:
            adjpar = [string]
        for p in adjpar:
            p = p.split(definer,1)
            kwargs[p[0]] = check(p[1])
    return kwargs

def add_name_from_dict(dictionary, cutkey = 2, cutitem = 3, keysep = None):
    '''
    Transforms keys and values of dictionary into a string that can be used
    as an identifier for file name
    '''
    addname = ''
    for key in dictionary:
        if keysep is not None:
            if keysep in str(key):
                keyd = str(key).split('_')
                apname = ''
                for ap in keyd:
                    apname += ap[0]
            else:
                apname = str(key)[:cutkey]
        else:
            apname = str(key)[:cutkey]
        
        addname += apname+str(dictionary[key])[:cutitem]
    return addname

def isfloat(number):
    '''
    Check if string is float
    '''
    try:
        float(number)
    except:
        return False
    else:
        return True

def isint(x):
    '''
    Check if string is int
    '''
    try:
        int(x) 
        return True
    except:
        return False

def numbertype(inbool):
    '''
    check if string can be integer or float
    '''
    try:
        int(inbool)
    except:
        pass
    else:
        return int(inbool)
    try:
        float(inbool)
    except:
        pass
    else:
        return float(inbool)
    return inbool

def write_meme_file(pwm, pwmname, alphabet, output_file_path):
    """[summary]
    write the pwm to a meme file
    Args:
        pwm ([np.array]): n_filters * 4 * motif_length
        output_file_path ([type]): [description]
    """
    n_filters = len(pwm)
    print(f'Writing {n_filters} filters in {output_file_path}')
    meme_file = open(output_file_path, "w")
    meme_file.write("MEME version 4 \n")
    meme_file.write("ALPHABET= "+alphabet+" \n")
    meme_file.write("strands: + -\n")

    # Switch axes if necessary
    switch = True
    for p, pw in enumerate(pwm):
        if pw.shape[1] != len(alphabet):
            switch *= False
    
    for i in range(0, n_filters):
        pw = pwm[i]
        if switch:
            pw = pw.T
        meme_file.write("\n")
        meme_file.write("MOTIF %s \n" % pwmname[i])
        meme_file.write("letter-probability matrix: alength= "+str(len(alphabet))+" w= %d \n"% np.count_nonzero(np.sum(pw, axis=0)))

        for j in range(0, np.shape(pw)[-1]):
            for a in range(len(alphabet)):
                if a < len(alphabet)-1:
                    meme_file.write(str(pw[ a, j])+ "\t")
                else:
                    meme_file.write(str(pw[ a, j])+ "\n")

    meme_file.close()

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
        if header == 'auto': 
            if not isfloat(f[0].strip().split(delimiter)[-1]):
                columns = f[0].strip().replace(strip_names,'').split(delimiter)

        elif f[0][:len(header)] == header:
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
        values = np.array(values, dtype = object)
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

def readin_motif_files(pwmfile, nameline = 'Motif'):
    '''
    Motifs are saved as meme files or npz files.
    This function is a wrapper to identify the file format.
    '''
    infmt= os.path.splitext(pwmfile)[1]    
    if infmt == '.meme':
        pwm_set, pwmnames, nts = read_meme(pwmfile)
    elif infmt == '.npz':
        pf = np.load(pwmfile, allow_pickle = True)
        pwm_set, pwmnames = pf['pwms'] , pf['pwmnames']
        nts = None
        if 'nts' in pf:
            nts = pf['nts']
    else:
        print('Only accepts .meme and .npz files with keys "pwms", "pwmnames", and "nts"')
        sys.exit()

    return pwm_set, pwmnames, nts

def read_meme(pwmlist, nameline = 'MOTIF'):
    '''
    Read in meme file
    Parameters
    ----------
    pwmlist: 
        PATH to file
        
    Returns
    -------
    pwms : 
        Motifs
    names : 
        Names or identifiers of motifs
    nts : 
        Alphabet
    '''
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split()
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0] == 'ALPHABET=':
                nts = list(line[1])
            elif isinstance(numbertype(line[0]), float):
                pwm.append(line)
    if len(pwm) > 0:
        pwm = np.array(pwm, dtype = float)
        pwms.append(np.array(pwm))
        names.append(name)
    return np.array(pwms, dtype = object), np.array(names), np.array(nts)

def readtomtom(f):
    '''
    Minimalistic function to read names, targets, p-values and q-values from
    tomtom output
    '''
    obj = open(f,'r').readlines()
    names = []
    pvals =[]
    qvals = []
    target = []
    for l, line in enumerate(obj):
        if l > 0 and line[0] != '#':
            line = line.strip().split('\t')
            if len(line) > 5:
                names.append(line[0])
                target.append(line[1])
                pvals.append(line[3])
                qvals.append(line[5])
    if len(names) == 0:
        print('Could not read tomtom file')
        sys.exit()
    names = np.array(names)
    target = np.array(target)
    pvals = np.array(pvals, dtype = float)
    qvals = np.array(qvals, dtype = float)
    return names, target, pvals, qvals

