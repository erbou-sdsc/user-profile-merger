# User Profile Merger

Consolidate user profiles for event campaign coordination

## Usage

```
./merge.py --list *.csv  --rules rules.json --encoding 'iso-8859-1' --out out.csv --log debug --logfile out.log

--list: a list of input csv files to be merged.
--out: the output of the merged files (default to terminal).
--rules: an optional json file with a set of data processing rules.
--encoding: optional default encoding, if not specified the tool will try to detect the most likely encoding.
--log: log level (debug, info, warning, error, critical).
--logfile: log output (default to terminal).
```

## Rule set

Transformation rules are expressed in the JSON rule file. The rules consist of five main sections, other sections are ignored:

- Encode: a default encoding used for the files.
- Rename: set of regex substitution rules used to rename the table headers.
- Transform: set of regex substitution rules used to convert individual record values.
- Validate: set of predefined rules or regex substitution rules used to validate individual record values.
- Custom: set of custom actions, used to invoke registered python code on given fields.

### Encoding rules

Specify the _encoding_ of files that matches the given regex _file_ pattern.
Patterns must matches from the start of the file path name.

```
{
    "encode": [
       { "encoding": null, "comment": "use null as default to autmatically detect the encoding." },
       { "file": "(?i:.*\\.utf8)$",   "encoding": "UTF-8" }
    ],
}
```

### Renaming rules

Rename headers matches the given regex _field_ pattern.
Patterns must matches from the start of the field name.

```
    "rename": [
       { "field": "(?i:Name)$", "to": "Names", "comment": "rename Name to Names" }
    ],
```

### Transformation rules

Substitute into _to_ values in column _field_ (or all) that match the _re_ pattern.
Use back references in _to_, e.g. "\\1" to insert group captures.

```
    "transform": [
       { "field": "(?i:First Name)", "re": "^(?i:\\s*dr\\.|\\s*prof\\.)+\\s+", "to": "" },
       { "re": "^\\s+", "to": "" },
       { "re": "\\s+$", "to": "" }
    ],
```

### Validation rules

Ensure that the values in the _field_ column either match the provided _re_ pattern or conform to the predefined type rule
(in the example below, both rules are equivalent). Validation occurs after the transformation.
Use the _log_ attribute to specify the log level for invalid entries (setting to 'critical' will raise an uncaught exception).

```
    "validate": [
       { "field": "(?i:Email)", "type": "email", "log": "warning", "unique": true, "id": "email:1" },
       { "field": "(?i:Email)", "re": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9_.-]+\\.[a-zA-Z0-9_.-]+(\\.[a-zA-Z0-9_.-])?$", "log": "warning", "unique": true, "id": "email:2" }
    ],
```

### Custom rules

Perform custom actions on field values that match the _re_ expression (or on all fields if _field_ is null, and all values if _re_ is null).
Actions registered under the _action_ name in the Python code are invoked with the row, represented as a dictionary, after transformations and validations have been applied.

```
    "custom": [
       { "field": "(?i:Email)", "re": "(?i:ethz\\.ch|epfl\\.ch|unil\\.ch)", "action": "academia_Domains", "id": "academia:1" }
    ],
```

## Notes
- The tool preserves gmail aliases such as username+tag@gmail.com, however it ignores the +tag when checking for uniqueness.
- The tool reports duplicates, but does nothing about them
- We frequently receive input files that combine records from multiple sources, all merged under a single encoding. As a result, records created with different encodings may appear garbled. The tool attempts to automatically correct these encoding issues, but this process is ad-hoc and has not been thoroughly tested with all possible encoding combinations.
