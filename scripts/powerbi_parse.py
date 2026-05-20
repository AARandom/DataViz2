import json
import re
from pathlib import Path

import pandas as pd


TRAIN_LINE_HINTS = {
    "Williamstown",
    "Werribee",
    "Upfield",
    "Sunbury",
    "Stony Point",
    "Sandringham",
    "Pakenham",
    "Mernda",
    "Lilydale",
    "Hurstbridge",
    "Glen Waverly",
    "Glen Waverley",
    "Frankston",
    "Cranbourne",
    "Craigieburn",
    "Belgrave",
    "Alamein",
}

OUTPUT_COLUMNS = [
    ("G0", "Line (Alamein, Belgrave, etc)"),
    ("M0", "% TT delivered (Train)"),
    ("M1", "% ontime (Train|dest)"),
    ("M2", "% cancelled (Train)"),
    ("M3", "% short (Train)"),
    ("M4", "% bypass (Train|Loops)"),
    ("M5", "% skip services (Train)"),
    ("M6", "# scheduled (Train)"),
    ("M7", "# scheduled (Train|dest)"),
    ("M8", "# ontime (Train|dest)"),
    ("M9", "# cancelled (Train)"),
    ("M10", "# short (Train)"),
    ("M11", "# bypass (Train|Loops)"),
    ("M12", "# skip services (Train)"),
]


def _iter_dicts(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _iter_dsr_objects(payload):
    for candidate in _iter_dicts(payload):
        dsr = candidate.get("dsr")
        if isinstance(dsr, dict):
            yield dsr


def _normalize_schema(row):
    schema = row.get("S")
    if not isinstance(schema, list):
        return []

    columns = []
    for index, field in enumerate(schema):
        if isinstance(field, dict):
            name = field.get("N") or field.get("Name") or field.get("Value")
        else:
            name = None
        columns.append(name or f"field_{index}")
    return columns


def _row_to_record(row):
    record = {}
    columns = _normalize_schema(row)

    value_dict = row.get("ValueDict")
    if isinstance(value_dict, dict):
        if columns and all(f"M{i}" in value_dict for i in range(len(columns))):
            for index, column_name in enumerate(columns):
                record[column_name] = value_dict.get(f"M{index}")
        else:
            record.update(value_dict)

    for index, column_name in enumerate(columns):
        if column_name in row:
            record[column_name] = row[column_name]
        elif f"M{index}" in row:
            record[column_name] = row[f"M{index}"]

    if not record:
        for key, value in row.items():
            if key not in {"S", "ValueDict"} and not re.fullmatch(r"DM\d+", key):
                record[key] = value

    return record


def _columns_from_schema(schema):
    columns = []
    for index, field in enumerate(schema):
        if isinstance(field, dict):
            name = field.get("N") or field.get("Name") or field.get("Value")
        else:
            name = None
        columns.append(name or f"field_{index}")
    return columns


def _records_from_c_rows(items):
    schema = None
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("S"), list):
            schema = item.get("S")
            break

    if not schema:
        return []

    columns = _columns_from_schema(schema)
    records = []
    for item in items:
        if not isinstance(item, dict) or "C" not in item:
            continue

        values = item.get("C")
        if not isinstance(values, list):
            continue

        if values and not isinstance(values[0], str):
            # Skip subtotal/metadata rows where the first column isn't a label.
            continue

        record = {}
        for index, column_name in enumerate(columns):
            if index < len(values):
                record[column_name] = values[index]
        if record:
            records.append(record)

    return records


def _iter_row_records(node):
    if isinstance(node, dict):
        if isinstance(node.get("S"), list) and (
            "ValueDict" in node or any(re.fullmatch(r"M\d+", key) for key in node)
        ):
            yield _row_to_record(node)

        for key, value in node.items():
            if re.fullmatch(r"DM\d+", key) and isinstance(value, list):
                yield from _iter_row_records(value)
            elif key == "PH" and isinstance(value, list):
                yield from _iter_row_records(value)
            elif key == "ValueDict" and isinstance(value, dict):
                continue
    elif isinstance(node, list):
        if any(isinstance(item, dict) and "C" in item for item in node):
            for record in _records_from_c_rows(node):
                yield record
            return

        for item in node:
            yield from _iter_row_records(item)


def _contains_train_line_hint(records):
    for record in records:
        for value in record.values():
            if isinstance(value, str) and value in TRAIN_LINE_HINTS:
                return True
    return False


def extract_dataframes(payload):
    tables = []

    for dsr_index, dsr in enumerate(_iter_dsr_objects(payload)):
        ds_list = dsr.get("DS")
        if not isinstance(ds_list, list):
            continue

        for ds_index, dataset in enumerate(ds_list):
            if not isinstance(dataset, dict):
                continue

            records = list(_iter_row_records(dataset))
            if not records:
                continue

            frame = pd.DataFrame(records)
            frame.insert(0, "source_dsr_index", dsr_index)
            frame.insert(1, "source_ds_index", ds_index)
            tables.append(frame)

    return tables


def extract_train_lines_dataframe(payload):
    tables = extract_dataframes(payload)
    for frame in tables:
        if _contains_train_line_hint(frame.to_dict("records")):
            if "G0" not in frame.columns:
                return frame.reset_index(drop=True)

            frame = frame[frame["G0"].isin(TRAIN_LINE_HINTS)].copy()
            if frame.empty:
                return pd.DataFrame()

            renamed = {}
            for source, target in OUTPUT_COLUMNS:
                if source in frame.columns:
                    renamed[source] = target
                else:
                    frame[source] = pd.NA
                    renamed[source] = target

            ordered = [source for source, _ in OUTPUT_COLUMNS]
            frame = frame[ordered].rename(columns=renamed)
            return frame.reset_index(drop=True)

    return pd.DataFrame()


def load_payload(json_path):
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    payload = load_payload("powerbi_raw_data.json")
    frame = extract_train_lines_dataframe(payload)

    if frame.empty:
        print("No train-line table was found in the captured data.")
        return

    frame.to_csv("powerbi_train_lines.csv", index=False)
    print("Saved cleaned table to powerbi_train_lines.csv")


if __name__ == "__main__":
    main()
