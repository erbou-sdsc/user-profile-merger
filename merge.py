#!/usr/bin/env python3

import argparse
import configparser
import csv
import json
import logging
import os
import pathlib
import re
import sqlite3
import sys

from collections import defaultdict

# Report
class Report:
    def __auto_dict(self):
        return defaultdict(self.__auto_dict)

    def __init__(self):
        self.obj = self.__auto_dict()
        self.unique = defaultdict(set)

    def addUnique(self, field, value):
        self.unique[field].add(value)

    def print(self):
        for k,v in reports.unique.items(): 
            logger.info(f'UNIQUE IN FIELD {k}: {v}')

# Decorator to register functions into a lookup table
class FunctionRegistry:
    # Class-level dictionary to store functions
    function_lookup_table = {}

    @staticmethod
    def register_function(func):
        FunctionRegistry.function_lookup_table[func.__name__] = func
        return func

    @staticmethod
    def exec(name):
        return FunctionRegistry.function_lookup_table[name]

@FunctionRegistry.register_function
def academia_Domains(output):
    """Add Academic to Domains"""
    domains = output.get('Domains',None)
    if domains is None:
        return
    domains = [x for x in list(set(output.get('Domains', '').split(','))|set(['Academic'])) if x.strip() != '']
    domains.sort()
    domains = ','.join(domains)
    if domains != output['Domains']:
        logger.info(f'Action: {output["Email Address"]} {output["Domains"]} -> {domains}')
    output['Domains'] = domains

def splitCell(cell):
    return next(csv.reader([cell], skipinitialspace=True))

def getFileAndTag(filename):
    qf = filename.strip().split(':')
    if len(qf) > 1:
        return qf[0].strip(),qf[1].strip()
    else:
        return filename,None

## Open a file from path or duplicate a file descriptor (e.g. stdout)
def open_file(f, **kwargs):
    if isinstance(f, ( str, pathlib.Path )):
        return open(f, **kwargs)
    elif isinstance(f, int):
        return os.fdopen(f, **kwargs)
    else:
        raise ValueError(f"f {type(f)} must be either a file path (str) or a file descriptor (int).")

## Merge list b into a, trying to preserve order of a as much as possible
def merge_list(a, b):
    if len(a) == 0:
       return b
    new_set = set(b) - set(a)
    merged_list = a
    b_i = 0

    while len(new_set) > 0:
        for i,m in enumerate(b[b_i:]):
            if m in new_set:
                b_i += i + 1
                after = set(b[b_i:])
                new_set -= {m}
                break
        if len(after) > 0:
            for i,n in enumerate(merged_list):
                if n in after:
                    break
            merged_list = merged_list[:i] + [m] + merged_list[i:]
        else:
            merged_list = merged_list + [m]

    assert(len(set(a+b) - set(merged_list)) == 0)
        
    return merged_list

## Validate value based on type (email, ...) and return True/False result of validation and
## a canonic form of the value that can be used to check for the unique constraint.
##
def validateValue(value, isa):
    match (isa or '').lower():
        case 'email':
            v = re.sub(r'\+.*@gmail', '@gmail', value).lower() # Google ...+...@gmail.com
            ok = re.match('^[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+\\.[a-zA-Z0-9_.-]+(\\.[a-zA-Z0-9_.-])?$', v)
        case _:
            ok = True
            v = value
    return v, ok

## Return log level from string
def getLogLevel(level : str) -> int:
    return logging.getLevelNamesMapping().get(level.upper())

## Detect encoding of file f
def detect_encoding(f, rules, args):
    encoding = args.encoding
    for r in rules.get('encode', []):
        if r.get('file') is None:
           if encoding is None:
               encoding = r.get('encoding')
        elif  re.search(r.get('file'), f):
           encoding = r.get('encoding')
           break
    if encoding is None:
        import chardet
        with open(f, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            logger.debug(f'{result}')
    else:
        logger.debug(f'Encoding: {encoding}')  # This will show you the detected encoding
    return encoding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge MailChimp lists.')
    parser.add_argument('--config', '-c', metavar='config', type=pathlib.Path, default='config.ini', help='configuration INI file')
    parser.add_argument('--list', '-f',  metavar='files', type=str, nargs='+', help='csv files - filename[:tag], if tag is provided it is added to the TAGS column')
    parser.add_argument('--encoding', '-e', metavar='code', type=str, help='default input encoding')
    parser.add_argument('--o_encoding', '-x', metavar='code', type=str, default=None, help='output encoding')
    parser.add_argument('--rules', '-r', metavar='file', type=pathlib.Path, help='conversion rules')
    parser.add_argument('--out', '-o', metavar='file', type=pathlib.Path, help='output file')
    parser.add_argument('--loglevel', '-v', metavar='level', choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help='log level')
    parser.add_argument('--logfile', '-O', metavar='file', type=pathlib.Path, help='log file')
    args = parser.parse_args()

    default_section = (args.config.name.split(':')[1:] or ['DEFAULT'])[0]
    config = configparser.ConfigParser(default_section=default_section)
    config_path = pathlib.Path().joinpath(args.config.parent, args.config.name.split(':')[0])
    config.read(config_path)

    for k in config[default_section]:
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, config[default_section][k])
    args.o_encoding = (args.o_encoding or config.get('DEFAULT', 'o_encoding', fallback=None) or 'utf-8-sig').lower()

    logger = logging.getLogger(__name__)
    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile, filemode='w', format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8', level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8', level=logging.INFO)
    logger.setLevel(args.loglevel.upper())

    if args.rules is not None:
        with open(args.rules, 'r') as ruleset:
            rules = json.load(ruleset)
    else:
        rules = {}

    ## Merge the column names from all files
    ## rename the columns according to the rules
    ## and capitalize each word of the fields and
    ##
    logger.info(f'ITERATION 1 -- Merge headers')
    column_map = dict()
    fieldnames = []
    reports = Report()
    for f in args.list:
        f,file_tag = getFileAndTag(f)
        try:
            if args.out is not None and os.path.samefile(f, args.out):
                continue
        except:
            pass
        logger.info(f'FILE {f} tag:{file_tag}')
        encoding = detect_encoding(f, rules, args)

        with open(f, 'r', encoding = encoding, newline='') as file:
            csvreader = csv.DictReader(file)

            columns = dict()
            for c in csvreader.fieldnames:
                s = c.strip()
                for r in rules.get('rename', []):
                    s = re.sub(r['field'], r['to'], s)
                s = re.sub(r'\b[a-z]', lambda m: m.group(0).upper(), s)
                if (s != c):
                    logger.debug(f'Rename field {c} to {s}')
                    pass
                columns[c] = s

            axx = set(columns.values()) - set(column_map.values())
            if (len(axx) > 0):
                logger.debug(f'ADD  {axx}')
            mxx = set(column_map.values()) - set(columns.values())
            if (len(mxx) > 0):
                logger.debug(f'NOT  {mxx}')
            column_map |= columns
            fieldnames = merge_list(fieldnames, [ column_map[c] for c in csvreader.fieldnames ])

    logger.info(f'FIELDS {fieldnames}')

    ## Merge the rows from all files
    ## Transform and validate the values accordding to the rules
    ##
    logger.info(f'ITERATION 2 -- Merge rows')
    unique_idx = dict()
    with open_file(args.out or sys.stdout.fileno(), mode='w', newline='', encoding=args.o_encoding) as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        for f in args.list:
            f,file_tag = getFileAndTag(f)
            if args.out is not None and os.path.samefile(f, args.out):
                continue
            logger.info(f'FILE {f} tag:{file_tag}')
            encoding = detect_encoding(f, rules, args)
    
            with open(f, 'r', encoding = encoding, newline='') as file:
                csvreader = csv.DictReader(file)
                for row in csvreader:
                    action_set = set()
                    logger.debug(f'{"-"*100}')
                    output = dict()
                    for field in csvreader.fieldnames:
                        renamed_field = column_map.get(field)
                        value=row[field]

                        # Clean the mess caused by people merging inputs of different encodings into same file :-/
                        try:
                            value = value.encode('latin1').decode('utf-8')
                        except (UnicodeDecodeError, TypeError):
                            try:
                               value = value.encode('utf-8').decode('utf-8')
                            except (UnicodeDecodeError, TypeError):
                               try:
                                   value = value.encode('iso-8859-1').decode('utf-8')
                               except (UnicodeDecodeError, TypeError):
                                   try:
                                       value = value.encode('windows-1252').decode('utf-8')
                                   except (UnicodeDecodeError, TypeError):
                                       logger.warning(f'Re-encoding failed for `{value}`')

                        # Transform field values according to regex rules
                        for r in rules.get('transform', []):
                            if r.get('field') is None or re.match(r.get('field'), renamed_field):
                                if r.get('re'):
                                    value = re.sub(r['re'], r['to'], value)
                                if r.get('field') and (add_elem := r.get('add')) and add_elem:
                                    valueList=set(splitCell(value))
                                    if add_elem == '@.INPUT_TAG' and file_tag:
                                        add_elem = file_tag
                                    if not add_elem.startswith('@.'):
                                        valueList.add(add_elem)
                                    value = ','.join(['"'+x.strip(' "')+'"' for x in valueList])
                        if value != row[field]:
                            logger.info(f'Transform {field} `{row[field]}` to `{value}`')

                        # Create reports
                        for r in rules.get('report', []):
                            if (f := r.get('field')) and f is not None and re.match(f, renamed_field) and (s := r.get('stats')) and s:
                                if 'unique' in s:
                                    for vv in set(splitCell(value)):
                                        reports.addUnique(renamed_field, vv.strip(' "'))
    
                        # Validate field values according to regex rules
                        for r in rules.get('validate', []):
                            if r.get('field') is None or re.match(r.get('field'), renamed_field):
                                canonic_value, ok = validateValue(value, r.get('type'))
                                if not ok or r.get('re', False) and not re.match(r.get('re'), value):
                                    lvl = getLogLevel(r.get('log', 'warning'))
                                    msg = f'Invalid {renamed_field}={value} (rule:{r.get("id","")})'
                                    logger.log(lvl, msg)
                                    if lvl == logging.CRITICAL:
                                        raise Exception(msg)
                                if r.get('unique', False):
                                    unique_key = renamed_field + '::' + canonic_value.lower().replace(" ","")
                                    if unique_idx.get(unique_key, False):
                                        lvl = getLogLevel(r.get('log', 'warning'))
                                        msg = f'Duplicate {renamed_field}={value} AND {unique_idx.get(unique_key)} (rule:{r.get("id","")})'
                                        logger.log(lvl, msg)
                                        if lvl == logging.CRITICAL:
                                            raise Exception(msg)
                                    else:
                                        unique_idx[unique_key] = value

                        # Custom actions
                        for r in rules.get('custom', []):
                            if r.get('field') is None or re.match(r.get('field'), renamed_field):
                                if r.get('re') is None or re.search(r.get('re'), value):
                                    action_set |= set([r['action']])

                        logger.debug(f'{renamed_field}={value}')
                        output[renamed_field] = value

                    for act in action_set:
                        FunctionRegistry.exec(act)(output)
 
                    logger.debug(f'INPUT: {row}')
                    logger.debug(f'OUTPUT: {output}')
                    csvwriter.writerow(output)

    reports.print()
