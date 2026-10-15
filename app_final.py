# ============================================================
# NEUROGUARDIAN AI
# Flask Backend + SQLite Database
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

import sqlite3
import os
import json
import uuid
from datetime import datetime, timezone
import traceback


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database"
)

DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "neuroguardian.db"
)

os.makedirs(
    DATABASE_DIR,
    exist_ok=True
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# CURRENT TIME
# ============================================================

def current_time():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():

    connection = get_db()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT DEFAULT '',
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)


    # --------------------------------------------------------
    # ADD PHONE COLUMN TO OLD DATABASE
    # --------------------------------------------------------

    try:

        cursor.execute("""
            ALTER TABLE users
            ADD COLUMN phone TEXT DEFAULT ''
        """)

    except sqlite3.OperationalError:

        # Phone column already exists.
        pass


    # --------------------------------------------------------
    # REPORTS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            user_email TEXT NOT NULL,
            module TEXT NOT NULL,
            date TEXT NOT NULL,
            tier TEXT DEFAULT 'neutral',
            title TEXT DEFAULT 'Report',
            summary TEXT DEFAULT '',
            details TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)


    # --------------------------------------------------------
    # REPORTS TABLE MIGRATION
    # --------------------------------------------------------
    # The current API uses a TEXT report id. Older project
    # databases may have an INTEGER id or another old schema.
    # In that case we keep the old table as a backup and create
    # a fresh compatible reports table.
    # --------------------------------------------------------

    cursor.execute("PRAGMA table_info(reports)")
    report_info = cursor.fetchall()

    report_schema = {
        row["name"]: row
        for row in report_info
    }

    required_columns = {
        "id",
        "user_email",
        "module",
        "date",
        "tier",
        "title",
        "summary",
        "details",
        "created_at"
    }

    id_column = report_schema.get("id")

    id_is_compatible = (
        id_column is not None
        and str(id_column["type"]).upper() == "TEXT"
        and int(id_column["pk"] or 0) == 1
    )

    schema_is_complete = required_columns.issubset(
        set(report_schema.keys())
    )

    if not id_is_compatible:

        backup_name = (
            "reports_legacy_"
            + datetime.now(timezone.utc).strftime(
                "%Y%m%d%H%M%S"
            )
        )

        print()
        print("Old/incompatible reports table detected.")
        print("Backing it up as:", backup_name)

        cursor.execute(
            f'ALTER TABLE reports RENAME TO "{backup_name}"'
        )

        cursor.execute("""
            CREATE TABLE reports (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                module TEXT NOT NULL,
                date TEXT NOT NULL,
                tier TEXT DEFAULT 'neutral',
                title TEXT DEFAULT 'Report',
                summary TEXT DEFAULT '',
                details TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)

        report_schema = {}
        schema_is_complete = True

    elif not schema_is_complete:

        required_definitions = {
            "user_email": "TEXT DEFAULT ''",
            "module": "TEXT DEFAULT 'unknown'",
            "date": "TEXT DEFAULT ''",
            "tier": "TEXT DEFAULT 'neutral'",
            "title": "TEXT DEFAULT 'Report'",
            "summary": "TEXT DEFAULT ''",
            "details": "TEXT DEFAULT '{}'",
            "created_at": "TEXT DEFAULT ''"
        }

        for column_name, column_definition in required_definitions.items():

            if column_name not in report_schema:

                try:

                    cursor.execute(
                        f'ALTER TABLE reports ADD COLUMN "{column_name}" {column_definition}'
                    )

                    print(
                        "Added missing reports column:",
                        column_name
                    )

                except sqlite3.OperationalError as migration_error:

                    print(
                        "REPORT COLUMN MIGRATION ERROR:",
                        column_name,
                        migration_error
                    )


    # --------------------------------------------------------
    # REPORT USER INDEX
    # --------------------------------------------------------

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_reports_user_email
        ON reports(user_email)
    """)


    # --------------------------------------------------------
    # REPORT MODULE INDEX
    # --------------------------------------------------------

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_reports_module
        ON reports(module)
    """)


    connection.commit()

    connection.close()


# ============================================================
# ERROR RESPONSE
# ============================================================

def error_response(
    message,
    status_code=400
):

    return jsonify({
        "success": False,
        "message": message
    }), status_code


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "success": True,

        "application":
            "NeuroGuardian AI",

        "backend":
            "Flask",

        "database":
            "SQLite",

        "status":
            "running"

    })


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/api/register",
    methods=["POST"]
)
def register():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return error_response(
                "No registration data received."
            )


        # ----------------------------------------------------
        # GET FORM DATA
        # ----------------------------------------------------

        name = str(
            data.get(
                "name",
                ""
            )
        ).strip()


        email = str(
            data.get(
                "email",
                ""
            )
        ).strip().lower()


        phone = str(
            data.get(
                "phone",
                ""
            )
        ).strip()


        password = str(
            data.get(
                "password",
                ""
            )
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not name:

            return error_response(
                "Full name is required."
            )


        if not email:

            return error_response(
                "Email address is required."
            )


        if not password:

            return error_response(
                "Password is required."
            )


        if len(password) < 8:

            return error_response(
                "Password must be at least 8 characters long."
            )


        # ----------------------------------------------------
        # PHONE VALIDATION
        # ----------------------------------------------------

        if phone:

            if (
                len(phone) != 10
                or not phone.isdigit()
                or phone[0] not in "6789"
            ):

                return error_response(
                    "Please enter a valid 10-digit Indian mobile number."
                )


        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        connection = get_db()

        cursor = connection.cursor()


        # ----------------------------------------------------
        # CHECK EXISTING EMAIL
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (email,)
        )

        existing_user = cursor.fetchone()


        if existing_user:

            connection.close()

            return error_response(
                "An account with this email already exists.",
                409
            )


        # ----------------------------------------------------
        # HASH PASSWORD
        # ----------------------------------------------------

        password_hash = generate_password_hash(
            password
        )


        created_at = current_time()


        # ----------------------------------------------------
        # INSERT USER
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                phone,
                password,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                email,
                phone,
                password_hash,
                created_at
            )
        )


        user_id = cursor.lastrowid


        connection.commit()

        connection.close()


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "message":
                "Account created successfully.",

            "user": {

                "id":
                    user_id,

                "name":
                    name,

                "email":
                    email,

                "phone":
                    phone

            }

        }), 201


    except sqlite3.IntegrityError:

        return error_response(
            "An account with this email already exists.",
            409
        )


    except Exception as error:

        print(
            "REGISTER ERROR:",
            error
        )

        return error_response(
            "Registration failed. Please try again.",
            500
        )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/api/login",
    methods=["POST"]
)
def login():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return error_response(
                "No login data received."
            )


        email = str(
            data.get(
                "email",
                ""
            )
        ).strip().lower()


        password = str(
            data.get(
                "password",
                ""
            )
        )


        if not email:

            return error_response(
                "Email address is required."
            )


        if not password:

            return error_response(
                "Password is required."
            )


        # ----------------------------------------------------
        # FIND USER
        # ----------------------------------------------------

        connection = get_db()

        cursor = connection.cursor()


        cursor.execute(
            """
            SELECT
                id,
                name,
                email,
                phone,
                password,
                created_at
            FROM users
            WHERE email = ?
            """,
            (email,)
        )


        user = cursor.fetchone()


        connection.close()


        if not user:

            return error_response(
                "Invalid email or password.",
                401
            )


        # ----------------------------------------------------
        # VERIFY PASSWORD
        # ----------------------------------------------------

        password_valid = check_password_hash(
            user["password"],
            password
        )


        if not password_valid:

            return error_response(
                "Invalid email or password.",
                401
            )


        # ----------------------------------------------------
        # LOGIN SUCCESS
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "message":
                "Login successful.",

            "user": {

                "id":
                    user["id"],

                "name":
                    user["name"],

                "email":
                    user["email"],

                "phone":
                    user["phone"] or ""

            }

        })


    except Exception as error:

        print(
            "LOGIN ERROR:",
            error
        )

        return error_response(
            "Login failed. Please try again.",
            500
        )


# ============================================================
# GET USER
# ============================================================

@app.route(
    "/api/user",
    methods=["GET"]
)
def get_user():

    try:

        email = request.args.get(
            "email",
            ""
        ).strip().lower()


        if not email:

            return error_response(
                "Email is required."
            )


        connection = get_db()

        cursor = connection.cursor()


        cursor.execute(
            """
            SELECT
                id,
                name,
                email,
                phone,
                created_at
            FROM users
            WHERE email = ?
            """,
            (email,)
        )


        user = cursor.fetchone()


        connection.close()


        if not user:

            return error_response(
                "User not found.",
                404
            )


        return jsonify({

            "success": True,

            "user": {

                "id":
                    user["id"],

                "name":
                    user["name"],

                "email":
                    user["email"],

                "phone":
                    user["phone"] or "",

                "created_at":
                    user["created_at"]

            }

        })


    except Exception as error:

        print(
            "USER ERROR:",
            error
        )

        return error_response(
            "Could not retrieve user.",
            500
        )


# ============================================================
# SAVE REPORT
# ============================================================

@app.route(
    "/api/reports",
    methods=["POST"]
)
def save_report():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return error_response(
                "No report data received."
            )


        # ----------------------------------------------------
        # USER EMAIL
        # ----------------------------------------------------

        user_email = str(
            data.get(
                "user",
                data.get(
                    "email",
                    ""
                )
            )
        ).strip().lower()


        if not user_email:

            return error_response(
                "User email is required."
            )


        # ----------------------------------------------------
        # REPORT ID
        # ----------------------------------------------------

        report_id = str(
            data.get(
                "id",
                ""
            )
        ).strip()


        if not report_id:

            report_id = (
                str(
                    int(
                        datetime.now(
                            timezone.utc
                        ).timestamp() * 1000
                    )
                )
                + "-"
                + uuid.uuid4().hex[:8]
            )


        # ----------------------------------------------------
        # REPORT FIELDS
        # ----------------------------------------------------

        module = str(
            data.get(
                "module",
                "unknown"
            )
        ).strip()


        report_date = str(
            data.get(
                "date",
                current_time()
            )
        ).strip()


        tier = str(
            data.get(
                "tier",
                "neutral"
            )
        ).strip()


        title = str(
            data.get(
                "title",
                "Report"
            )
        ).strip()


        summary = str(
            data.get(
                "summary",
                ""
            )
        ).strip()


        details = data.get(
            "details",
            {}
        )


        # ----------------------------------------------------
        # DETAILS → JSON
        # ----------------------------------------------------

        if isinstance(
            details,
            str
        ):

            try:

                json.loads(
                    details
                )

                details_json = details

            except json.JSONDecodeError:

                details_json = json.dumps({
                    "text": details
                })

        else:

            details_json = json.dumps(
                details,
                ensure_ascii=False
            )


        created_at = current_time()


        # ----------------------------------------------------
        # SAVE REPORT
        # ----------------------------------------------------

        connection = get_db()

        cursor = connection.cursor()


        cursor.execute(
            """
            INSERT OR REPLACE INTO reports
            (
                id,
                user_email,
                module,
                date,
                tier,
                title,
                summary,
                details,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                user_email,
                module,
                report_date,
                tier,
                title,
                summary,
                details_json,
                created_at
            )
        )


        connection.commit()

        connection.close()


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "message":
                "Report saved successfully.",

            "report": {

                "id":
                    report_id,

                "module":
                    module,

                "date":
                    report_date,

                "tier":
                    tier,

                "title":
                    title,

                "summary":
                    summary,

                "details":
                    details,

                "user":
                    user_email

            }

        })


    except Exception as error:

        print(
            "SAVE REPORT ERROR:",
            error
        )

        traceback.print_exc()

        return error_response(
            "Could not save report.",
            500
        )


# ============================================================
# GET ALL REPORTS
# ============================================================

@app.route(
    "/api/reports",
    methods=["GET"]
)
def get_reports():

    try:

        user_email = request.args.get(
            "email",
            request.args.get(
                "user",
                ""
            )
        ).strip().lower()


        if not user_email:

            return error_response(
                "User email is required."
            )


        connection = get_db()

        cursor = connection.cursor()


        cursor.execute(
            """
            SELECT
                id,
                user_email,
                module,
                date,
                tier,
                title,
                summary,
                details,
                created_at
            FROM reports
            WHERE user_email = ?
            ORDER BY date DESC
            """,
            (user_email,)
        )


        rows = cursor.fetchall()


        connection.close()


        reports = []


        for row in rows:

            try:

                details = json.loads(
                    row["details"] or "{}"
                )

            except Exception:

                details = (
                    row["details"]
                    or {}
                )


            reports.append({

                "id":
                    row["id"],

                "module":
                    row["module"],

                "date":
                    row["date"],

                "tier":
                    row["tier"],

                "title":
                    row["title"],

                "summary":
                    row["summary"],

                "details":
                    details,

                "user":
                    row["user_email"],

                "created_at":
                    row["created_at"]

            })


        return jsonify({

            "success": True,

            "count":
                len(reports),

            "reports":
                reports

        })


    except Exception as error:

        print(
            "GET REPORTS ERROR:",
            error
        )

        return error_response(
            "Could not retrieve reports.",
            500
        )


# ============================================================
# GET SINGLE REPORT
# ============================================================

@app.route(
    "/api/reports/<report_id>",
    methods=["GET"]
)
def get_single_report(
    report_id
):

    try:

        user_email = request.args.get(
            "email",
            request.args.get(
                "user",
                ""
            )
        ).strip().lower()


        if not user_email:

            return error_response(
                "User email is required."
            )


        connection = get_db()

        cursor = connection.cursor()


        cursor.execute(
            """
            SELECT
                id,
                user_email,
                module,
                date,
                tier,
                title,
                summary,
                details,
                created_at
            FROM reports
            WHERE id = ?
            AND user_email = ?
            """,
            (
                report_id,
                user_email
            )
        )


        row = cursor.fetchone()


        connection.close()


        if not row:

            return error_response(
                "Report not found.",
                404
            )


        try:

            details = json.loads(
                row["details"] or "{}"
            )

        except Exception:

            details = (
                row["details"]
                or {}
            )


        return jsonify({

            "success": True,

            "report": {

                "id":
                    row["id"],

                "module":
                    row["module"],

                "date":
                    row["date"],

                "tier":
                    row["tier"],

                "title":
                    row["title"],

                "summary":
                    row["summary"],

                "details":
                    details,

                "user":
                    row["user_email"],

                "created_at":
                    row["created_at"]

            }

        })


    except Exception as error:

        print(
            "GET SINGLE REPORT ERROR:",
            error
        )

        return error_response(
            "Could not retrieve the report.",
            500
        )


# ============================================================
# DELETE REPORT
# ============================================================

@app.route(
    "/api/reports/<report_id>",
    methods=["DELETE"]
)
def delete_report(
    report_id
):

    try:

        user_email = request.args.get(
            "email",
            request.args.get(
                "user",
                ""
            )
        ).strip().lower()


        if not user_email:

            return error_response(
                "User email is required."
            )


        connection = get_db()

        cursor = connection.cursor()


        cursor.execute(
            """
            DELETE FROM reports
            WHERE id = ?
            AND user_email = ?
            """,
            (
                report_id,
                user_email
            )
        )


        deleted = cursor.rowcount


        connection.commit()

        connection.close()


        if deleted == 0:

            return error_response(
                "Report not found.",
                404
            )


        return jsonify({

            "success": True,

            "message":
                "Report deleted successfully."

        })


    except Exception as error:

        print(
            "DELETE REPORT ERROR:",
            error
        )

        return error_response(
            "Could not delete report.",
            500
        )


# ============================================================
# SYNC OLD LOCALSTORAGE REPORTS
# ============================================================

@app.route(
    "/api/reports/sync",
    methods=["POST"]
)
def sync_reports():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return error_response(
                "No synchronization data received."
            )


        user_email = str(
            data.get(
                "email",
                data.get(
                    "user",
                    ""
                )
            )
        ).strip().lower()


        reports = data.get(
            "reports",
            []
        )


        if not user_email:

            return error_response(
                "User email is required."
            )


        if not isinstance(
            reports,
            list
        ):

            return error_response(
                "Reports must be provided as an array."
            )


        connection = get_db()

        cursor = connection.cursor()


        saved_count = 0


        for report in reports:

            if not isinstance(
                report,
                dict
            ):

                continue


            report_id = str(
                report.get(
                    "id",
                    ""
                )
            ).strip()


            if not report_id:

                report_id = uuid.uuid4().hex


            module = str(
                report.get(
                    "module",
                    "unknown"
                )
            ).strip()


            report_date = str(
                report.get(
                    "date",
                    current_time()
                )
            ).strip()


            tier = str(
                report.get(
                    "tier",
                    "neutral"
                )
            ).strip()


            title = str(
                report.get(
                    "title",
                    "Report"
                )
            ).strip()


            summary = str(
                report.get(
                    "summary",
                    ""
                )
            ).strip()


            details = report.get(
                "details",
                {}
            )


            details_json = json.dumps(
                details,
                ensure_ascii=False
            )


            cursor.execute(
                """
                INSERT OR REPLACE INTO reports
                (
                    id,
                    user_email,
                    module,
                    date,
                    tier,
                    title,
                    summary,
                    details,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    user_email,
                    module,
                    report_date,
                    tier,
                    title,
                    summary,
                    details_json,
                    current_time()
                )
            )


            saved_count += 1


        connection.commit()

        connection.close()


        return jsonify({

            "success": True,

            "message":
                "Reports synchronized successfully.",

            "saved":
                saved_count

        })


    except Exception as error:

        print(
            "SYNC REPORTS ERROR:",
            error
        )

        return error_response(
            "Could not synchronize reports.",
            500
        )


# ============================================================
# SERVE FRONTEND
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return send_from_directory(
        BASE_DIR,
        "Login.html"
    )


@app.route(
    "/<path:filename>",
    methods=["GET"]
)
def serve_frontend(
    filename
):

    if filename.startswith(
        "api/"
    ):

        return error_response(
            "API endpoint not found.",
            404
        )


    file_path = os.path.join(
        BASE_DIR,
        filename
    )


    if not os.path.isfile(
        file_path
    ):

        return error_response(
            "File not found.",
            404
        )


    return send_from_directory(
        BASE_DIR,
        filename
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    init_database()


    print()
    print("=" * 46)
    print("       NEUROGUARDIAN AI BACKEND")
    print("=" * 46)
    print()
    print("Database:")
    print(DATABASE_PATH)
    print()
    print("Server:")
    print("http://127.0.0.1:5000")
    print()
    print("Health Check:")
    print(
        "http://127.0.0.1:5000/api/health"
    )
    print()
    print("=" * 46)
    print()


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )