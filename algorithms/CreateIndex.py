import sqlite3
from pathlib import Path
import pandas as pd
from collections import defaultdict

from utils.FreebaseIndex import FreebaseIndex
from utils.IOOperations import IOOperations


class CreateIndex:

    def __init__(self, index_path, literals_index_path, graphs_path, graph_prefix_filename="bridge_enhanced_graph"):
        self.index_path = Path(index_path)
        self.graphs_path = graphs_path
        self.literals_idx = FreebaseIndex(literals_index_path)
        self.graph_prefix_filename = graph_prefix_filename

    def build_bridge_type_outgoing_index(self):
        """
        Build a compact SQLite index using dictionary-encoding:
          strings(id INTEGER PRIMARY KEY, s TEXT UNIQUE)
          types(type_id INTEGER PRIMARY KEY, name_id INTEGER) WITHOUT ROWID
          neighs(type_id INTEGER, predicate_id INTEGER, object_id INTEGER,
                 PRIMARY KEY(type_id, predicate_id, object_id)) WITHOUT ROWID

        - type_id is the type string's id in `strings`
        - name_id is the label string's id in `strings`
        - predicate_id/object_id are ids of predicate string and object label string
        """

        enhanced_dir = Path(self.graphs_path)


        if not self.index_path.exists():
            self.index_path.mkdir()

        db_path = self.index_path / "type_index.sql"

        # ---- open sqlite + schema (PRIMARY KEYs only) ----
        # Using WITHOUT ROWID for types and neighs to save space and improve lookup speed, 
        # since we don't need the implicit rowid.
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS strings (
              id INTEGER PRIMARY KEY,
              s  TEXT NOT NULL UNIQUE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS types (
              type_id INTEGER PRIMARY KEY,
              name_id INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS neighs (
              type_id      INTEGER NOT NULL,
              predicate_id INTEGER NOT NULL,
              object_id    INTEGER NOT NULL,
              PRIMARY KEY (type_id, predicate_id, object_id)
            ) WITHOUT ROWID
        """)
        conn.commit()

        def norm_key(k: str) -> str:
            return IOOperations.sanitize(k)

        TYPE_KEY_N = norm_key("type.object.type")

        # cache for string to id mapping to minimize DB queries; we rely on UNIQUE constraint to avoid duplicates
        sid_cache = {}

        def get_sid(s: str) -> int:
            if s in sid_cache:
                return sid_cache[s]
            # Compatible two-step (works without RETURNING)
            cur.execute("INSERT OR IGNORE INTO strings(s) VALUES (?)", (s,))
            cur.execute("SELECT id FROM strings WHERE s = ?", (s,))
            sid = cur.fetchone()[0]
            sid_cache[s] = sid
            return sid

        # iterate over enhanced graphs and accumulate type and neighbor info for bridge nodes
        edge_files = sorted(enhanced_dir.glob(f"{self.graph_prefix_filename}.edges.*.parquet"))
        graph_ids = [p.name.split(".")[2] for p in edge_files if len(p.name.split(".")) >= 4]

        # batch buffers and seen sets to avoid duplicates and keep memory usage bounded; 
        # we rely on PRIMARY KEY constraints to avoid duplicates in DB
        seen_types = set()  # int type_id
        seen_neighs = set()  # (type_id, predicate_id, object_id)
        batch_types = []  # (type_id, name_id)
        batch_neighs = []  # (type_id, predicate_id, object_id)

        for gid in graph_ids:
            edges_path = enhanced_dir / f"{self.graph_prefix_filename}.edges.{gid}.parquet"
            nodes_path = enhanced_dir / f"{self.graph_prefix_filename}.nodes.{gid}.parquet"
            try:
                edges = pd.read_parquet(edges_path, engine="fastparquet")
                nodes = pd.read_parquet(nodes_path, engine="fastparquet")
            except Exception:
                continue

            # normalize keys
            if "key" in edges.columns:
                edges = edges.copy()
                edges["key"] = edges["key"].astype(str).map(norm_key)

            # bridges: rely on is_bridge flag
            if "is_bridge" not in nodes.columns:
                continue
            bridges = nodes.loc[nodes["is_bridge"].fillna(False).astype(bool), ["node", "name", "types"]].copy()
            if bridges.empty:
                continue

            # quick index of edges by source
            edges_by_src = defaultdict(list)
            for s, t, k in zip(edges["source"].astype(str),
                               edges["target"].astype(str),
                               edges["key"].astype(str)):
                edges_by_src[s].append((t, k))

            for _, row in bridges.iterrows():
                mid = str(row["node"])
                type_label_from_bridge = row["name"] if isinstance(row["name"], str) and row["name"].strip() else None

                # find associated type string
                type_str = None
                types_val = row.get("types", None)
                if isinstance(types_val, (list, tuple)) and len(types_val) > 0:
                    cand = list(types_val)
                    type_targets = {t for (t, k) in edges_by_src.get(mid, []) if k == TYPE_KEY_N}
                    chosen = next((t for t in cand if t in type_targets), cand[0])
                    type_str = chosen
                else:
                    for t, k in edges_by_src.get(mid, []):
                        if k == TYPE_KEY_N:
                            type_str = t
                            break
                if not isinstance(type_str, str) or not type_str:
                    continue

                # encode strings to IDs
                type_id_int = get_sid(type_str)
                type_name_str = type_label_from_bridge or type_str
                name_id_int = get_sid(type_name_str)

                if type_id_int not in seen_types:
                    seen_types.add(type_id_int)
                    batch_types.append((type_id_int, name_id_int))

                # collect outgoing neighbor (predicate, object_label) and encode
                for tgt, key in edges_by_src.get(mid, []):
                    if key == TYPE_KEY_N:
                        continue
                    pred_id = get_sid(key)

                    # resolve target label: prefer nodes.name; else raw id
                    obj_label = None
                    if "node" in nodes.columns and "name" in nodes.columns:
                        rows = nodes.loc[nodes["node"].astype(str) == str(tgt), "name"]
                        if not rows.empty and isinstance(rows.iloc[0], str) and rows.iloc[0].strip():
                            obj_label = rows.iloc[0]
                    if not obj_label:
                        obj_label = str(tgt)
                    obj_id = get_sid(obj_label)

                    trip = (type_id_int, pred_id, obj_id)
                    if trip not in seen_neighs:
                        seen_neighs.add(trip)
                        batch_neighs.append(trip)

            # periodic flush to keep memory bounded
            if len(batch_neighs) > 100_000:
                cur.executemany(
                    "INSERT INTO types(type_id, name_id) VALUES(?, ?) "
                    "ON CONFLICT(type_id) DO UPDATE SET name_id=excluded.name_id",
                    batch_types
                )
                cur.executemany(
                    "INSERT OR IGNORE INTO neighs(type_id, predicate_id, object_id) VALUES(?, ?, ?)",
                    batch_neighs
                )
                conn.commit()
                batch_types.clear()
                batch_neighs.clear()

        # final flush
        if batch_types:
            cur.executemany(
                "INSERT INTO types(type_id, name_id) VALUES(?, ?) "
                "ON CONFLICT(type_id) DO UPDATE SET name_id=excluded.name_id",
                batch_types
            )
        if batch_neighs:
            cur.executemany(
                "INSERT OR IGNORE INTO neighs(type_id, predicate_id, object_id) VALUES(?, ?, ?)",
                batch_neighs
            )
        conn.commit()
        conn.close()
