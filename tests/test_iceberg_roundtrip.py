"""Spike/regression: prove the no-Docker Iceberg-at-Bronze round-trip works.

PyIceberg (SQLite catalog + local FS) writes a Bronze table; both PyIceberg and
DuckDB read it back; and a re-run via `overwrite` does not duplicate rows.
"""
import pyarrow as pa

from pantryiq.lakehouse.catalog import get_catalog, scan_with_duckdb


def _sample() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "name": pa.array(["a", "b", "c", "d", "e"], pa.string()),
        }
    )


def test_iceberg_bronze_roundtrip(tmp_path):
    catalog = get_catalog(tmp_path / "lakehouse")
    catalog.create_namespace("bronze")
    data = _sample()
    table = catalog.create_table("bronze.spike_demo", schema=data.schema)

    # Write, then read back with BOTH engines.
    table.append(data)
    table = catalog.load_table("bronze.spike_demo")
    assert table.scan().to_arrow().num_rows == 5
    assert scan_with_duckdb(table).num_rows == 5

    # Idempotent re-run: overwrite must NOT duplicate rows.
    table.overwrite(data)
    table = catalog.load_table("bronze.spike_demo")
    assert table.scan().to_arrow().num_rows == 5
    assert scan_with_duckdb(table).num_rows == 5
