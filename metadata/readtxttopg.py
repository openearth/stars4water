
"""
Load a pg_dump COPY-format text file into an existing PostgreSQL table.

The input file's first line holds the COPY statement (with column list),
the remaining lines are the tab-separated data rows in the same format,
so they can be streamed straight into the database via COPY FROM STDIN.

After loading, any missing title is filled from the CI_Citation title
(gco:CharacterString) found in the MD_DataIdentification block of the
XML stored in the 'data' column.
"""
import configparser
import xml.etree.ElementTree as ET

import psycopg2

txt = r"C:\projectinfo\eu\stars4water\metadata_s4w_dump\metadatatable_utf8.txt"
cfg = r"C:\develop\stars4water\metadata\configuration.txt"

config = configparser.ConfigParser()
config.read(cfg)
pg = config["PostGIS"]

conn = psycopg2.connect(
    host=pg["host"],
    dbname=pg["database"],
    user=pg["user"],
    password=pg["password"],
)

# namespace-agnostic xpath: matches both gmd: (19139) and mri:/mrc: (19115-3) variants
XPATH = ".//{*}MD_DataIdentification//{*}CI_Citation//{*}title//{*}CharacterString"


def fill_missing_titles(conn):
    with conn:
        with conn.cursor() as sel_cur, conn.cursor() as upd_cur:
            sel_cur.execute(
                "SELECT id, data FROM public.metadata WHERE title IS NULL OR title = ''"
            )
            rows = sel_cur.fetchall()

            updated = 0
            for row_id, data in rows:
                if not data:
                    continue
                try:
                    root = ET.fromstring(data)
                except ET.ParseError as e:
                    print(f"id {row_id}: could not parse xml - {e}")
                    continue

                title_el = root.find(XPATH)
                if title_el is None or not title_el.text or not title_el.text.strip():
                    continue

                upd_cur.execute(
                    "UPDATE public.metadata SET title = %s WHERE id = %s",
                    (title_el.text.strip(), row_id),
                )
                updated += 1

    print(f"Updated {updated} of {len(rows)} rows")


try:
    with open(txt, "r", encoding="utf-8") as f:
        copy_sql = f.readline().strip()
        if not copy_sql.upper().startswith("COPY"):
            raise ValueError(f"Expected a COPY statement on the first line, got: {copy_sql}")

        with conn:
            with conn.cursor() as cur:
                cur.copy_expert(copy_sql, f)

    print("Data successfully loaded into public.metadata")

    fill_missing_titles(conn)
finally:
    conn.close()

