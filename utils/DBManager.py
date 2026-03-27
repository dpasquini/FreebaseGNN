import sqlite3


class DBManager:
    _cache = {}

    @classmethod
    def get_db(cls, path):
        key = str(path)
        if key not in cls._cache:
            conn = sqlite3.connect(key)
            conn.row_factory = sqlite3.Row
            cls._cache[key] = conn
        return cls._cache[key]

    @classmethod
    def close_all(cls):
        for conn in cls._cache.values():
            try:
                conn.close()
            except Exception:
                pass
        cls._cache.clear()
