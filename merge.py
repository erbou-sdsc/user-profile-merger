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
from datetime import datetime


def parse_date(date_str, hint='%m/%d/%Y %H:%M'):
    if not hasattr(parse_date, '__hint'):
       parse_date.__hint = hint

    try:
        return (datetime.strptime(date_str, parse_date.__hint), parse_date.__hint)
    except ValueError:
        logger.warning(f'Cannot parse date {date_str} using format {parse_date.__hint}')

    for h in [ '%m/%d/%Y %H:%M', '%Y/%m/%d %H:%M', '%d/%m/%Y %H:%M' ]:
        try:
            date_ts = datetime.strptime(date_str, h)
            parse_date.__hint = h
            return date_ts, parse_date.__hint
        except ValueError:
            pass
    
    raise ValueError("Date format {date_str} not recognized or invalid date.")

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
def academia_Domains(output, *argv):
    """Add Academic to Domains"""
    domains = output.get('Domains',None)
    if domains is None:
        return
    domains = sorted([x for x in list(set(output.get('Domains', '').split(','))|set(['Academic'])) if x.strip() != ''])
    domains = ','.join(domains)
    if domains != output['Domains']:
        logger.info(f'Action: {output["Email Address"]} {output["Domains"]} -> {domains}')
    output['Domains'] = domains

@FunctionRegistry.register_function
def row_Merge(row1, row2, *argv):
    """
    Merge SDSC Rows, according to rules:
    - Email: Address, should be the same
    - R1: TAGS, NOTES, Domaine, Topics I Am Interested In, Source Of Contact: merge
    - R2: First Name, Last Name, Marketing Permissions: take non empty, otherwise they must be the same or take longest and warn
    - R3: Research Category, Company/Institution/Media/Other, Function, Country, Preferred Language: take most recent non-empty
    - R4: LAST_CHANGED,LEID(?),EUID(?): All from most recent LAST_CHANGED, and must not be empty
    - R5: MEMBER_RATING: ? Maximum?
    - R6: LATITUDE,LONGITUDE,GMTOFF,DSTOFF,TIMEZONE,CC,REGION: use most recent non-empty tuple
    - R7: OPTIN_TIME,OPTIN_IP: take value pair of the earliest OPTIN_TIME
    - R8: CONFIRM_TIME,CONFIRM_IP: take value pair of the most recent CONFIRM_TIME
    - Otherwise values from most recent LAST_CHANGED take precedence.
    """

    fields = set(row1.keys()) | set(row2.keys())
    def __findField(field):
        for k in fields:
            if re.match(field, k, re.I):
                return k
        return None

    date1, _ = parse_date(row1['LAST_CHANGED'])
    date2, _ = parse_date(row2['LAST_CHANGED'])

    for field in [ 'Tag', 'Note', 'Domain', 'Topic', 'Source' ]:
        k_field = __findField(field)
        if k_field is None:
            logger.debug(f'No field matches `{field}`')
            continue
        if not row1.get(k_field, ''):
            row1[k_field] = row2.get(k_field, '')
            continue
        elif not row2.get(k_field, ''):
            continue
        fieldList1=set(splitCell(row1[k_field]))
        fieldList2=set(splitCell(row2[k_field]))
        fieldList1|=fieldList2
        row1[k_field] = ','.join(['"'+x.strip(' "')+'"' for x in sorted(fieldList1)])

    for field in [ 'First +N', 'Last +N', 'Marketing +P' ]:
        k_field = __findField(field)
        if k_field is None:
            logger.debug(f'No field matches `{field}`')
            continue
        if not row1.get(k_field, ''):
            row1[k_field] = row2.get(k_field, '')
            continue
        elif not row2.get(k_field, ''):
            continue
        tok1 = re.sub('[^a-z]', '', row1.get(k_field), flags=re.I)
        tok2 = re.sub('[^a-z]', '', row2.get(k_field), flags=re.I)
        if tok1.lower() != tok2.lower() and len(tok1) > 0 and len(tok2) > 0:
            logger.warning(f'Fields `{field}`: `{row1.get(k_field)}` And `{row2.get(k_field)}` are different')
            if len(tok2) > len(tok1):
                row1[k_field] = row2[k_field].strip()

    for field in [ 'Research +C', 'Comp.*Inst', 'Function', 'Country', 'Preferred +L' ]:
        k_field = __findField(field)
        if k_field is None:
            logger.debug(f'No field matches `{field}`')
            continue
        if not row1.get(k_field, ''):
            row1[k_field] = row2.get(k_field, '')
            continue
        elif not row2.get(k_field, ''):
            continue
        if date1 < date2:
            row1[k_field] = row2[k_field]

    if date1 < date2:
       row1['LAST_CHANGED'] = row2['LAST_CHANGED']
       row1['LEID'] = row2['LEID']
       row1['EUID'] = row2['EUID']

    locations = ['LATITUDE','LONGITUDE','GMTOFF','DSTOFF','TIMEZONE','CC','REGION']
    nSet1 = set([f for f in locations if row1.get(f)])
    nSet2 = set([f for f in locations if row1.get(f)])

    if (date1 < date2 or len(nSet1) == 0) and len(nSet2) > 0:
        for f in locations:
            if f in nSet2:
                row1[f] = row2[f]
            elif row1.get(f):
                del row1[f]

    if parse_date(row1['OPTIN_TIME'])[0] > parse_date(row2['OPTIN_TIME'])[0]:
        row1['OPTIN_TIME'] = row2['OPTIN_TIME']
        if row2.get('OPTIN_IP', ''):
            row1['OPTIN_IP'] = row2['OPTIN_IP']
        else:
            del row1['OPTIN_IP']

    if parse_date(row1['CONFIRM_TIME'])[0] < parse_date(row2['CONFIRM_TIME'])[0]:
        row1['CONFIRM_TIME'] = row2['CONFIRM_TIME']
        if row2.get('CONFIRM_IP', ''):
            row1['CONFIRM_IP'] = row2['CONFIRM_IP']
        else:
            del row1['CONFIRM_IP']

    logger.debug(f'Merged ouput: {row1}')
    return row1

def splitCell(cell):
    return next(csv.reader([cell], skipinitialspace=True))

def getFileAndTag(filename):
    qf = filename.strip().split(':')
    if len(qf) > 1:
        return qf[0].strip(),qf[1].strip()
    else:
        return filename,None

def unwind(d, fk):
    while isinstance(d[fk], (str,int)):
        fk = d[k]
    return fk

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
    parser.add_argument('--config', '-c', metavar='Config', type=pathlib.Path, default='config.ini', help='configuration INI file')
    parser.add_argument('--list', '-f',  metavar='Files', type=str, nargs='+', help='csv files - filename[:tag], if tag is provided it is added to the TAGS column')
    parser.add_argument('--encoding', '-e', metavar='Code', type=str, help='default input encoding')
    parser.add_argument('--o_encoding', '-x', metavar='Code', type=str, default=None, help='output encoding')
    parser.add_argument('--rules', '-r', metavar='File', type=pathlib.Path, help='conversion rules')
    parser.add_argument('--out', '-o', metavar='File', type=pathlib.Path, help='output file')
    parser.add_argument('--loglevel', '-v', metavar='Level', choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help='log level')
    parser.add_argument('--logfile', '-O', metavar='File', type=pathlib.Path, help='log file')
    parser.add_argument('--merge', '-m', metavar='Action', default=None,  help='action for merging redundant rows, only row_Merge is supported')
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
    merged_rows = dict()
    row_id = 0
    try:
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
                        duplicateSet = set()
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
                                        value = ','.join(['"'+x.strip(' "')+'"' for x in sorted(valueList)])
                            if value.strip(''' '"''') != row[field].strip(''' '"'''):
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
                                            duplicateSet.add(int(unique_idx.get(unique_key)[1]))
                                            if lvl == logging.CRITICAL:
                                                raise Exception(msg)
                                        else:
                                            unique_idx[unique_key] = (value, row_id)
    
                            # Custom actions
                            for r in rules.get('custom', []):
                                if r.get('field') is None or re.match(r.get('field'), renamed_field):
                                    if r.get('re') is None or re.search(r.get('re'), value):
                                        action_set |= set([r['action']])
    
                            logger.debug(f'{renamed_field}={value}')
                            output[renamed_field] = value
    
                        for act in action_set:
                            act_param = act.split(':')
                            FunctionRegistry.exec(act_param[0])(output, *act_param[1:])
     
                        logger.debug(f'INPUT: {row}')
                        logger.debug(f'OUTPUT: {output}')
                        if not args.merge:
                            csvwriter.writerow(output)
                        else: 
                            merge_funct = FunctionRegistry.exec(args.merge.split(':')[0])
                            merge_args = args.merge.split(':')[1:]
                            if len(duplicateSet) > 0:
                                dup_k=sorted(duplicateSet)
                                pk1 = unwind(merged_rows, dup_k[0])
                                for ik in dup_k[1:]:
                                    pk2 = unwind(merged_rows, ik)
                                    if pk2 != pk1:
                                        logger.debug(f'Merge {merged_rows[pk2]} into {merged_rows[pk1]}')
                                        output = merge_funct(merged_rows[pk2], merged_rows[pk1], *merge_args)
                                        merged_rows[pk2] = pk1
                                logger.debug(f'Merge {output} into {merged_rows[pk1]}')
                                output = merge_funct(merged_rows[pk1], output, *merge_args)
                                logger.debug(f'Merged output: {output}')
                            else:
                                merged_rows[row_id]=output
                                row_id+=1

            if args.merge:
                for _, row in sorted(merged_rows.items()):
                    if not isinstance(row, (str, int)):
                        csvwriter.writerow(row)
        reports.print()
    except Exception as e:
        logger.exception(e)
    finally:
        pass
