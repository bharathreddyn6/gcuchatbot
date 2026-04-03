"""
Complaint Management System — Database Layer
SQLite-backed persistence for complaints and comments.
"""

import sqlite3
import uuid
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "complaints.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the complaints database schema."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            complaint_id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            image_path TEXT,
            location TEXT,
            category TEXT,
            department TEXT,
            status TEXT NOT NULL DEFAULT 'Pending',
            student_name TEXT,
            student_email TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS complaint_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT NOT NULL,
            author TEXT,
            author_role TEXT DEFAULT 'student',
            message TEXT NOT NULL,
            image_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (complaint_id) REFERENCES complaints(complaint_id)
        )
    """)

    conn.commit()
    conn.close()
    print("[INFO] Complaints DB initialized.")


def _generate_id() -> str:
    return "CMP-" + uuid.uuid4().hex[:8].upper()


def create_complaint(
    description: str,
    category: str,
    department: str,
    student_name: str = "",
    student_email: str = "",
    location: str = "",
    image_path: str = None,
) -> Dict[str, Any]:
    """Insert a new complaint. Returns the created record."""
    now = datetime.now().isoformat()
    complaint_id = _generate_id()

    conn = get_connection()
    conn.execute(
        """INSERT INTO complaints
           (complaint_id, description, image_path, location, category, department,
            status, student_name, student_email, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (complaint_id, description, image_path, location, category, department,
         "Pending", student_name, student_email, now, now),
    )
    conn.commit()

    row = conn.execute(
        "SELECT * FROM complaints WHERE complaint_id=?", (complaint_id,)
    ).fetchone()
    conn.close()
    return dict(row)


def get_complaint(complaint_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single complaint by ID including its comments."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM complaints WHERE complaint_id=?", (complaint_id,)
    ).fetchone()

    if not row:
        conn.close()
        return None

    complaint = dict(row)

    comments = conn.execute(
        "SELECT * FROM complaint_comments WHERE complaint_id=? ORDER BY created_at",
        (complaint_id,),
    ).fetchall()
    complaint["comments"] = [dict(c) for c in comments]
    conn.close()
    return complaint


def update_complaint_status(complaint_id: str, status: str) -> bool:
    """Update status of a complaint. Returns True on success."""
    valid = {"Pending", "In Progress", "Resolved", "Rejected"}
    if status not in valid:
        return False
    now = datetime.now().isoformat()
    conn = get_connection()
    cur = conn.execute(
        "UPDATE complaints SET status=?, updated_at=? WHERE complaint_id=?",
        (status, now, complaint_id),
    )
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def add_comment(
    complaint_id: str,
    message: str,
    author: str = "Student",
    author_role: str = "student",
    image_path: str = None,
) -> Dict[str, Any]:
    """Add a comment/message to a complaint. Returns inserted row."""
    now = datetime.now().isoformat()
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO complaint_comments
           (complaint_id, author, author_role, message, image_path, created_at)
           VALUES (?,?,?,?,?,?)""",
        (complaint_id, author, author_role, message, image_path, now),
    )
    row_id = cur.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM complaint_comments WHERE id=?", (row_id,)).fetchone()
    conn.close()
    return dict(row)


def get_complaints_by_dept(department: str) -> List[Dict[str, Any]]:
    """Fetch all complaints assigned to a department."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM complaints WHERE department=? ORDER BY created_at DESC",
        (department,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_complaints(status: str = None) -> List[Dict[str, Any]]:
    """Admin: fetch all complaints optionally filtered by status."""
    conn = get_connection()
    if status:
        rows = conn.execute(
            "SELECT * FROM complaints WHERE status=? ORDER BY created_at DESC", (status,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM complaints ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> Dict[str, Any]:
    """Return summary statistics for the admin panel."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    by_status = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) as cnt FROM complaints GROUP BY status"
    ).fetchall():
        by_status[row[0]] = row[1]

    by_category = {}
    for row in conn.execute(
        "SELECT category, COUNT(*) as cnt FROM complaints GROUP BY category"
    ).fetchall():
        by_category[row[0]] = row[1]

    conn.close()
    return {
        "total": total,
        "by_status": by_status,
        "by_category": by_category,
    }


# Auto-init on import
init_db()
