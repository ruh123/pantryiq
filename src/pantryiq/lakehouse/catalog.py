"""Local Iceberg catalog for the Bronze layer (no-Docker path).

PyIceberg writes; a SQLite catalog + a local-filesystem warehouse hold the tables.
DuckDB reads back via the table's *current* metadata file (see `scan_with_duckdb`) —
which is how Silver/Gold (dbt-duckdb) will consume Bronze. Promote the SQLite catalog
to a REST catalog before deploy; the table data + metadata carry over unchanged.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.table import Table

DEFAULT_WAREHOUSE = Path("data/lakehouse")


def get_catalog(warehouse: Path | str = DEFAULT_WAREHOUSE, name: str = "pantryiq") -> SqlCatalog:
    """Return a SQLite-backed Iceberg catalog rooted at `warehouse` (created if missing)."""
    warehouse = Path(warehouse)
    warehouse.mkdir(parents=True, exist_ok=True)
    return SqlCatalog(
        name,
        uri=f"sqlite:///{warehouse / 'catalog.db'}",
        warehouse=warehouse.resolve().as_uri(),
    )


def scan_with_duckdb(table: Table, con: duckdb.DuckDBPyConnection | None = None) -> pa.Table:
    """Read an Iceberg table with DuckDB via its current metadata file.

    Resolving the metadata path from the table object (rather than guessing the
    latest file on disk) is what makes DuckDB reads work against a PyIceberg SQLite
    catalog, which DuckDB cannot attach to directly.
    """
    con = con or duckdb.connect()
    con.execute("INSTALL iceberg; LOAD iceberg;")
    metadata = table.metadata_location.replace("file://", "")
    return con.execute("SELECT * FROM iceberg_scan(?)", [metadata]).to_arrow_table()
