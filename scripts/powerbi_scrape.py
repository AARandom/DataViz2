import base64
import json
import time
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


def _iter_row_records(node):
    if isinstance(node, dict):
        if isinstance(node.get("S"), list) and (
            "ValueDict" in node or any(re.fullmatch(r"M\d+", key) for key in node)
        ):
            yield _row_to_record(node)

        for key, value in node.items():
            if re.fullmatch(r"DM\d+", key) and isinstance(value, list):
                for item in value:
                    yield from _iter_row_records(item)
            elif key == "PH" and isinstance(value, list):
                for item in value:
                    yield from _iter_row_records(item)
            elif key == "ValueDict" and isinstance(value, dict):
                continue
    elif isinstance(node, list):
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
            return frame.reset_index(drop=True)

    return pd.DataFrame()


def _is_powerbi_domain(url):
    return any(domain in url for domain in ("powerbi", "analysis", "pbidedicated"))


def _is_powerbi_data_response(response):
    url = response.url.lower()
    if not _is_powerbi_domain(url):
        return False

    content_type = response.headers.get("content-type", "").lower()
    return (
        "json" in content_type
        or "application/problem+json" in content_type
        or "querydata" in url
        or "visualdata" in url
    )

def intercept_pbi_data():
    from playwright.sync_api import sync_playwright

    # The target Power BI dashboard URL
    url = "https://app.powerbi.com/view?r=eyJrIjoiNWQ1MDkwMDEtMDA2NS00ODhlLWFjNzgtMjhhOTM0M2U3OWYxIiwidCI6IjUwOTRjN2E3LTA3NDgtNDY2ZS05NDFlLTcyODgyYzMwOTdiYSJ9"
    
    # Lists to store intercepted payloads and request metadata
    captured_data = []
    request_log = []

    with sync_playwright() as p:
        # Launching in non-headless mode so you can see it working.
        # Once it works, you can change this to headless=True
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page()

        def handle_request(request):
            url = request.url.lower()
            if not _is_powerbi_domain(url):
                return

            request_log.append(
                {
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                }
            )

        # This function listens to the Power BI responses that usually carry the visual data.
        def handle_response(response):
            if not _is_powerbi_data_response(response):
                return

            try:
                body = response.json()
                payload_type = "json"
            except Exception:
                try:
                    body = json.loads(response.text())
                    payload_type = "json"
                except Exception:
                    raw_bytes = response.body()
                    body = {
                        "content_type": response.headers.get("content-type"),
                        "body_base64": base64.b64encode(raw_bytes[:20000]).decode("ascii"),
                        "body_truncated": len(raw_bytes) > 20000,
                    }
                    payload_type = "binary"

            captured_data.append(
                {
                    "url": response.url,
                    "status": response.status,
                    "payload_type": payload_type,
                    "data": body,
                }
            )
            print(f"✅ Intercepted a data packet from: {response.url[:80]}...")

        # Attach the listener to the page
        page.on("request", handle_request)
        page.on("response", handle_response)

        print("Loading Power BI dashboard and listening for data requests...")
        page.goto(url, wait_until="domcontentloaded")
        
        # Wait a bit longer to ensure the dashboard fully loads and all data is fetched
        print("Waiting for visual containers to populate...")
        page.wait_for_timeout(50000)
        
        browser.close()

        # Save the intercepted network data to a JSON file
        output_file = "powerbi_raw_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(captured_data, f, indent=4)

        with open("powerbi_capture_log.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "url": item.get("url"),
                        "status": item.get("status"),
                        "has_data": bool(item.get("data")),
                    }
                    for item in captured_data
                ],
                f,
                indent=4,
            )

        with open("powerbi_request_log.json", "w", encoding="utf-8") as f:
            json.dump(request_log, f, indent=4)
        
        print(f"🎉 Scraping complete. Raw data saved to {output_file}.")
        print(f"Captured {len(captured_data)} candidate Power BI responses.")
        print("Provide this JSON file to your coding agent to parse out the rows and columns.")


def load_train_lines_dataframe(json_path="powerbi_raw_data.json"):
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return extract_train_lines_dataframe(payload)


def main():
    # 1. RUN THE SCRAPER FIRST
    print("--- STEP 1: Intercepting Network Data ---")
    print("IMPORTANT: When the browser opens, quickly click the navigation arrows ")
    print("at the bottom to reach Page 4 (Metropolitan train: Performance snapshot)!")
    intercept_pbi_data()

    # 2. PARSE THE DATA SECOND
    print("--- STEP 2: Parsing the JSON ---")
    frame = load_train_lines_dataframe()
    if frame.empty:
        print("No train-line table was found. Did you make it to Page 4 in time?")
        return

    print(frame)
    frame.to_csv("powerbi_train_lines.csv", index=False)
    print("Saved cleaned table to powerbi_train_lines.csv")

if __name__ == "__main__":
    main()