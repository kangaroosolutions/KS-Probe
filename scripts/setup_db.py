#!/usr/bin/env python3
"""
Initialize the KS-Probe database schema.

Creates all tables (idempotent — safe to run multiple times).
Uses SQLite by default; set DATABASE_URL env var for PostgreSQL.

Usage:
    python scripts/setup_db.py
    python scripts/setup_db.py --db-url postgresql://user:pass@localhost/ks_probe_results
"""
import argparse
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from ks_probe.db.engine import get_engine, init_db
from ks_probe.core.config import get_db_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize KS-Probe database")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    args = parser.parse_args()

    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    db_url = get_db_url()
    print(f"Initializing database: {db_url}")

    init_db(db_url)

    engine = get_engine(db_url)
    from ks_probe.db.schema import Base
    table_names = list(Base.metadata.tables.keys())
    print(f"Tables created: {table_names}")
    print("Database setup complete.")


if __name__ == "__main__":
    main()
