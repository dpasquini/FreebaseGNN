import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from utils.DBManager import DBManager


class FreebaseIndex:
    def __init__(self, db_path):
        self.db_path = Path(db_path)

    def _conn(self):
        return DBManager.get_db(self.db_path)

    def get_diversified_sample(self, connections, max_sample_size=15):
        """
        Creates a diversified sample from a list of connection tuples.

        This function prioritizes variety by taking items from each predicate group
        in a round-robin fashion until the max sample size is reached.
        """
        # Remove duplicate (predicate, object) pairs to start with variety.
        # Sorting makes the output deterministic.
        unique_connections = sorted(list(set(connections)))

        # If we're already under the limit, no sampling is needed.
        if len(unique_connections) <= max_sample_size:
            return unique_connections

        # Group connections by their predicate to ensure we sample from each type.
        grouped_by_predicate = defaultdict(list)
        for pred, obj in unique_connections:
            grouped_by_predicate[pred].append(obj)

        # Perform round-robin sampling to build the final list.
        sample = []
        predicates = list(grouped_by_predicate.keys())
        # Find the length of the largest group to set the loop boundary.
        max_len = max(len(v) for v in grouped_by_predicate.values())

        for i in range(max_len):
            # Stop if the sample is full.
            if len(sample) >= max_sample_size:
                break
            for pred in predicates:
                # Add an item from each predicate group if it exists at this index.
                if i < len(grouped_by_predicate[pred]):
                    obj = grouped_by_predicate[pred][i]
                    sample.append((pred, obj))
                    if len(sample) >= max_sample_size:
                        break
        return sample

    def get_bridge_neighs(self, entity_id, predicate=None):
        """
        Retrieves outgoing neighbors for a given entity ID from the compact SQLite index.
        
        Parameters:
            - entity_id: the Freebase entity ID for which to retrieve neighbors
            - predicate: if specified, filters neighbors to only those with this predicate
        Returns:
            - if predicate is None:
                list[(p, o)] for outgoing non-type edges, without duplicates
            - else:
                list[o] where p == predicate, without duplicates
            - None on error / not found
        """
        try:
            cursor = self._conn().cursor()
            cursor.execute(
                "SELECT value FROM freebase_index WHERE key = ?",
                (entity_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            connections = json.loads(row[0])

            if predicate is None:
                seen = set()
                result = []
                for c in connections:
                    if c.get('dir') != 'out':
                        continue
                    if c.get('p') == 'type.object.type':
                        continue

                    key = (c.get('p'), c.get('o'))
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(key)
                return result

            else:
                seen = set()
                result = []
                for c in connections:
                    if c.get('dir') != 'out':
                        continue
                    if c.get('p') != predicate:
                        continue

                    o = c.get('o')
                    if o in seen:
                        continue
                    seen.add(o)
                    result.append(o)
                return result

        except sqlite3.Error as e:
            print(f"Database error in get_bridge_neighs for {entity_id}: {e}", file=sys.stderr)
            return None


    def get_types_for_node(self, entity_id):
        """
        Retrieves the types for a given entity ID from the compact SQLite index.
        Parameters:
            - entity_id: the Freebase entity ID for which to retrieve types

        Returns:
            - list[(p, o)] where p == 'type.object.type', without duplicates.
        """
        try:
            cursor = self._conn().cursor()
            cursor.execute(
                "SELECT value FROM freebase_index WHERE key = ?",
                (entity_id,),
            )
            row = cursor.fetchone()
            if not row:
                return []

            connections = json.loads(row[0])

            seen = set()
            types = []
            for c in connections:
                if c.get('dir') != 'out':
                    continue
                if c.get('p') != 'type.object.type':
                    continue

                key = (c.get('p'), c.get('o'))
                if key in seen:
                    continue
                seen.add(key)
                types.append(key)

            return types

        except sqlite3.Error as e:
            print(f"Database error in get_types_for_node for {entity_id}: {e}", file=sys.stderr)
            return []


    def get_literals_for_node(self, entity_id):
        """
        The function prioritizes 'type.object.name' and 'common.topic.alias' literals, 
        but if those are not available, it will try to find any English literal as a fallback.

        Parameters:
            - entity_id: the Freebase entity ID for which to retrieve literals
        Returns:
            - A tuple (predicate, literal_value) if a suitable literal is found, or None if no literals are found or an error occurs.
        """
        try:
            cursor = self._conn().cursor()
            cursor.execute(
                "SELECT value FROM freebase_index WHERE key = ?",
                (entity_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            connections = json.loads(row[0])
            literals = defaultdict(list)
            for c in connections:
                literals[c['p']].append(c['o'])

            # same priority logic as your code
            if literals.get('type.object.name'):
                return "type.object.name", literals['type.object.name'][0]
            if literals.get('common.topic.alias'):
                return "type.object.name", literals['common.topic.alias'][0]
            if literals.get('en'):
                return "type.object.name", literals['en'][0].replace('_', ' ')

            # otherwise, try to detect first English literal
            for values in literals.values():
                for element in values:
                    try:
                        if isinstance(element, str) and detect(element) == 'en':
                            return "type.object.name", element
                    except LangDetectException:
                        continue

            return None

        except sqlite3.Error as e:
            print(f"Database error in get_literals_for_node for {entity_id}: {e}", file=sys.stderr)
            return None

    def get_neighs_for_type(self, type_str: str, sample_size: int = 15):
        """
        Resolve a type string (e.g., 'american_football.football_coach') to its label
        and retrieve all (predicate, object_label) neighbors from the compact SQLite index.

        Returns:
            (label: str, neighs: list[dict])  or  None if type not found.

        Schema expected:
            strings(id INTEGER PRIMARY KEY, s TEXT UNIQUE)
            types(type_id INTEGER PRIMARY KEY, name_id INTEGER)
            neighs(type_id INTEGER, predicate_id INTEGER, object_id INTEGER,
                   PRIMARY KEY(type_id, predicate_id, object_id))
        """
        try:
            cursor = self._conn().cursor()

            # 1) Resolve type_str -> type_id
            cursor.execute("SELECT id FROM strings WHERE s = ?", (type_str,))
            row = cursor.fetchone()
            if not row:
                return None
            type_id = row[0]

            # 2) Get human label for the type (from types.name_id -> strings.s)
            cursor.execute("""
                SELECT s.s
                FROM types t
                JOIN strings s ON s.id = t.name_id
                WHERE t.type_id = ?
            """, (type_id,))
            row = cursor.fetchone()
            type_label = row[0] if row and isinstance(row[0], str) and row[0].strip() else type_str

            # 3) Fetch all neighbors, decode predicate/object labels
            cursor.execute("""
                SELECT sp.s AS predicate, so.s AS object
                FROM neighs n
                JOIN strings sp ON sp.id = n.predicate_id
                JOIN strings so ON so.id = n.object_id
                WHERE n.type_id = ?
                ORDER BY sp.s, so.s
            """, (type_id,))
            rows = cursor.fetchall()

            neighs = [(p, o) for (p, o) in rows]

            # 4) Optional sampling
            if sample_size is not None and sample_size > 0 and len(neighs) > sample_size:
                    # your helper expects tuples most likely
                    neighs = self.get_diversified_sample(neighs, max_sample_size=sample_size)
            else:
                neighs = neighs[:sample_size]

            return type_label, neighs

        except sqlite3.Error as e:
            print(f"SQLite error in get_neighs_for_type('{type_str}'): {e}")
            return None

    def get_entities(self, entity_id, predicate=None):
        """Looks up connections for an entity ID."""
        try:
            return self.get_bridge_neighs(entity_id, predicate)
        except sqlite3.Error as e:
            print(f"Database error in get_entities for {entity_id}: {e}", file=sys.stderr)