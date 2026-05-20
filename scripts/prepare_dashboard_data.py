from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "cleaned"


def load_station_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    fields = [field["id"] for field in payload["fields"]]
    df = pd.DataFrame(payload["records"], columns=fields)

    numeric_cols = [
        "Stop_lat",
        "Stop_long",
        "Pax_annual",
        "Pax_weekday",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def map_shape_line_id(shape_id: str) -> tuple[str, str]:
    parts = shape_id.split("-")
    code = ""
    if len(parts) >= 2:
        code = parts[1].strip()
    code = code or shape_id

    line_map = {
        "ALM": "Alamein",
        "BEL": "Belgrave",
        "CRA": "Craigieburn",
        "CRB": "Cranbourne",
        "FRK": "Frankston",
        "GLN": "Glen Waverley",
        "HUR": "Hurstbridge",
        "LIL": "Lilydale",
        "MER": "Mernda",
        "PAK": "Pakenham",
        "SND": "Sandringham",
        "SUN": "Sunbury",
        "UPF": "Upfield",
        "WER": "Werribee",
        "WIL": "Williamstown",
        "STP": "Stony Point",
    }

    return code, line_map.get(code, code)


def find_powerbi_file() -> Optional[Path]:
    candidates = [
        BASE_DIR / "scripts" / "powerbi_train_lines.csv",
        DATA_DIR / "vista" / "powerbi_train_lines_apr2025_to_apr2026.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def categorize_fare_type(faretype: str, agegroup: str) -> str:
    faretype_lower = (faretype or "").strip().lower()
    agegroup_value = (agegroup or "").strip()

    min_age = None
    if "->" in agegroup_value:
        left = agegroup_value.split("->")[0]
        if left.isdigit():
            min_age = int(left)
    elif agegroup_value.endswith("+"):
        left = agegroup_value[:-1]
        if left.isdigit():
            min_age = int(left)

    if "senior" in faretype_lower or (min_age is not None and min_age >= 65):
        return "Senior"
    if any(term in faretype_lower for term in ["concession", "student", "child", "youth"]):
        return "Concession"
    if any(term in faretype_lower for term in ["full", "adult"]):
        return "Full Fare"
    return "Other"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    station_2024 = load_station_json(
        DATA_DIR / "station_entry_and_exits" / "station_entry_2024_2025.json"
    )
    station_2019 = load_station_json(
        DATA_DIR / "station_entry_and_exits" / "station_entry_2019_2020.json"
    )

    station_columns = {
        "Stop_ID": "stop_id",
        "Stop_name": "station_name",
        "Stop_lat": "station_latitude",
        "Stop_long": "station_longitude",
        "Pax_weekday": "daily_passenger_volume",
        "Pax_annual": "annual_passenger_count",
        "Fin_year": "year",
    }

    station_2024_clean = station_2024.rename(columns=station_columns)[
        list(station_columns.values())
    ]
    station_2019_clean = station_2019.rename(columns=station_columns)[
        list(station_columns.values())
    ]

    station_2024_clean.to_csv(OUTPUT_DIR / "station_volumes_2024_2025.csv", index=False)
    station_2019_clean.to_csv(OUTPUT_DIR / "station_volumes_2019_2020.csv", index=False)

    station_change = station_2019.merge(
        station_2024,
        on="Stop_ID",
        suffixes=("_2019", "_2024"),
    )
    station_change = station_change.assign(
        annual_change=station_change["Pax_annual_2024"] - station_change["Pax_annual_2019"],
        annual_change_pct=(
            (station_change["Pax_annual_2024"] - station_change["Pax_annual_2019"])
            / station_change["Pax_annual_2019"].replace(0, pd.NA)
        ),
    )

    station_change_output = station_change[
        [
            "Stop_ID",
            "Stop_name_2019",
            "Stop_lat_2019",
            "Stop_long_2019",
            "Pax_annual_2019",
            "Pax_annual_2024",
            "annual_change",
            "annual_change_pct",
        ]
    ].rename(
        columns={
            "Stop_ID": "stop_id",
            "Stop_name_2019": "station_name",
            "Stop_lat_2019": "station_latitude",
            "Stop_long_2019": "station_longitude",
            "Pax_annual_2019": "annual_passenger_count_2019",
            "Pax_annual_2024": "annual_passenger_count_2024",
        }
    )
    station_change_output.to_csv(
        OUTPUT_DIR / "station_volume_change_2019_2024.csv", index=False
    )

    shapes_path = DATA_DIR / "metropolitan_train_gtfs_schedule" / "shapes.txt"
    shapes_df = pd.read_csv(
        shapes_path,
        usecols=[
            "shape_id",
            "shape_pt_lat",
            "shape_pt_lon",
            "shape_pt_sequence",
        ],
    )
    line_codes = (
        shapes_df["shape_id"]
        .astype(str)
        .str.split("-", n=2, expand=True)
        .iloc[:, 1]
        .fillna(shapes_df["shape_id"].astype(str))
    )
    shapes_df["train_line_id"] = line_codes
    shapes_df["train_line_name"] = line_codes.map(
        lambda code: map_shape_line_id(code)[1]
    )
    shapes_output = shapes_df.rename(
        columns={
            "shape_pt_lat": "latitude",
            "shape_pt_lon": "longitude",
            "shape_pt_sequence": "sequence_order",
        }
    )[["train_line_id", "train_line_name", "latitude", "longitude", "sequence_order"]]

    shapes_output.to_csv(OUTPUT_DIR / "train_shapes.csv", index=False)

    powerbi_path = find_powerbi_file()
    if powerbi_path is None:
        raise FileNotFoundError("powerbi_train_lines.csv not found in scripts or data/vista.")

    performance_df = pd.read_csv(
        powerbi_path,
        engine="python",
        on_bad_lines="skip",
    )
    performance_output = performance_df.rename(
        columns={
            "Line (Alamein, Belgrave, etc)": "train_line_name",
            "% ontime (Train|dest)": "on_time_percentage",
            "% cancelled (Train)": "cancellation_percentage",
        }
    )[["train_line_name", "on_time_percentage", "cancellation_percentage"]]

    performance_output.to_csv(OUTPUT_DIR / "train_line_performance.csv", index=False)

    stops_path = DATA_DIR / "vista" / "stops_vista_2024_2025.csv"
    stops_usecols = [
        "persid",
        "mainmode",
        "starthour",
        "travdow",
        "dayType",
        "stoppoststratweight",
    ]
    stops_df = pd.read_csv(stops_path, usecols=stops_usecols)
    stops_train = stops_df[stops_df["mainmode"] == "Train"].copy()

    stops_train["time_of_day_hour"] = pd.to_numeric(
        stops_train["starthour"], errors="coerce"
    )
    stops_train["day_of_week"] = stops_train["travdow"].fillna(
        stops_train["dayType"]
    )
    stops_train["passenger_count"] = pd.to_numeric(
        stops_train["stoppoststratweight"], errors="coerce"
    )

    hourly_output = (
        stops_train.groupby("time_of_day_hour", dropna=True)["passenger_count"]
        .sum()
        .reset_index()
    )
    hourly_output.to_csv(OUTPUT_DIR / "vista_train_by_hour.csv", index=False)

    heatmap_output = (
        stops_train.groupby(["day_of_week", "time_of_day_hour"], dropna=True)[
            "passenger_count"
        ]
        .sum()
        .reset_index()
    )
    heatmap_output.to_csv(OUTPUT_DIR / "vista_train_by_day_hour.csv", index=False)

    persons_path = DATA_DIR / "vista" / "person_vista_2024_2025.csv"
    persons_usecols = ["persid", "faretype", "agegroup"]
    persons_df = pd.read_csv(persons_path, usecols=persons_usecols)

    fare_joined = stops_train.merge(persons_df, on="persid", how="left")
    fare_joined["fare_category"] = fare_joined.apply(
        lambda row: categorize_fare_type(row.get("faretype"), row.get("agegroup")), axis=1
    )

    fare_output = (
        fare_joined.groupby("fare_category", dropna=False)["passenger_count"]
        .sum()
        .reset_index()
    )
    total_passengers = fare_output["passenger_count"].sum()
    fare_output["percentage"] = fare_output["passenger_count"] / total_passengers
    fare_output.to_csv(OUTPUT_DIR / "vista_train_fare_type_share.csv", index=False)

    journey_files = [
        (DATA_DIR / "vista" / "journey_to_work_vista_2024_2025.csv", "Work"),
        (DATA_DIR / "vista" / "journey_to_education_vista_2024_2025.csv", "Education"),
    ]

    journey_frames = []
    for path, label in journey_files:
        journeys = pd.read_csv(path, usecols=[
            "main_journey_mode",
            "start_time",
            "start_LGA",
            "end_LGA",
            "journey_weight",
        ])
        journeys = journeys[journeys["main_journey_mode"] == "Train"].copy()
        journeys["start_time"] = pd.to_numeric(journeys["start_time"], errors="coerce")
        journeys = journeys[
            journeys["start_time"].between(360, 540, inclusive="both")
        ]

        journeys["direction"] = "Other"
        journeys.loc[
            (journeys["start_LGA"] != "Melbourne (C)")
            & (journeys["end_LGA"] == "Melbourne (C)"),
            "direction",
        ] = "Inbound"
        journeys.loc[
            (journeys["start_LGA"] == "Melbourne (C)")
            & (journeys["end_LGA"] != "Melbourne (C)"),
            "direction",
        ] = "Outbound"

        journeys["journey_type"] = label
        journeys["passenger_count"] = pd.to_numeric(
            journeys["journey_weight"], errors="coerce"
        )
        journey_frames.append(journeys)

    journey_all = pd.concat(journey_frames, ignore_index=True)
    journey_output = (
        journey_all.groupby(["journey_type", "direction"], dropna=False)[
            "passenger_count"
        ]
        .sum()
        .reset_index()
    )
    journey_output.to_csv(OUTPUT_DIR / "vista_train_am_peak_direction.csv", index=False)

    print(f"Wrote cleaned files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
