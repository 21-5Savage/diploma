# SQLite CLI Basics

This is a concise, practical cheat sheet for the `sqlite3` command‑line tool.

## 1) Start SQLite

Open or create a database file:

```bash
sqlite3 path/to/db.sqlite
```

If the file doesn't exist, SQLite creates it.

## 2) Helpful Dot Commands

Dot commands are specific to the CLI (not SQL):

```bash
.help                -- show help
.open FILE           -- open a database
.databases           -- list attached databases
.tables              -- list tables
.schema TABLE        -- show schema for a table
.quit                -- exit
.read FILE.sql       -- execute SQL from file
.mode                -- set output mode (list, column, csv, etc.)
.headers on|off      -- toggle headers
```

## 3) Basic SQL

Create a table:

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT
);
```

Insert rows:

```sql
INSERT INTO users (name, created_at)
VALUES ('Ada', '2024-01-01');
```

Query rows:

```sql
SELECT * FROM users;
```

Update rows:

```sql
UPDATE users SET name = 'Ada Lovelace' WHERE id = 1;
```

Delete rows:

```sql
DELETE FROM users WHERE id = 1;
```

## 4) Output Formatting

Show results in columns with headers:

```bash
.mode column
.headers on
SELECT * FROM users;
```

Export to CSV:

```bash
.mode csv
.headers on
.output users.csv
SELECT * FROM users;
.output stdout
```

## 5) Importing CSV

```bash
.mode csv
.import users.csv users
```

For CSV with headers, you can skip the header row manually:

```bash
.mode csv
.import --skip 1 users.csv users
```

## 6) Run SQL From a File

```bash
sqlite3 db.sqlite < schema.sql
```

Or inside SQLite:

```bash
.read schema.sql
```

## 7) Transactions

```sql
BEGIN;
INSERT INTO users (name) VALUES ('Grace');
INSERT INTO users (name) VALUES ('Linus');
COMMIT;
```

Rollback on error:

```sql
ROLLBACK;
```

## 8) Indexes

```sql
CREATE INDEX idx_users_name ON users(name);
```

Check query plan:

```sql
EXPLAIN QUERY PLAN SELECT * FROM users WHERE name = 'Ada';
```

## 9) Attach Another Database

```sql
ATTACH DATABASE 'other.sqlite' AS other;
SELECT * FROM other.some_table;
DETACH DATABASE other;
```

## 10) Backup

Inside the CLI:

```bash
.backup backup.sqlite
```

Or from shell:

```bash
sqlite3 db.sqlite ".backup backup.sqlite"
```

---

If you want, I can tailor this to your project schema or add examples that match your dataset.
