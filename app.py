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

from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F


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

# Maximum upload size for image-based CT/MRI screening.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# MEDICAL IMAGE AI MODELS (LAZY LOADED)
# ============================================================
# MRI model: 4-class brain tumor MRI screening.
# CT model: single axial brain CT stroke screening.
# These are research/educational models, not clinical diagnosis.

MRI_MODEL_NAME = "Alia-Mohammed/resnet-50-finetuned-brain-tumor"
CT_MODEL_NAME = "Sahende/teknofest_ct_stroke_binary"

mri_classifier = None
mri_processor = None
mri_model = None
ct_classifier = None


def get_mri_classifier():
    """
    Load the MRI model explicitly so image preprocessing and class mapping
    are controlled by the model's own Hugging Face configuration.
    A pipeline fallback is kept for compatibility with older Transformers
    versions.
    """
    global mri_processor, mri_model, mri_classifier

    if mri_model is not None and mri_processor is not None:
        return "direct"

    if mri_classifier is not None:
        return "pipeline"

    print("Loading Brain MRI AI model...")

    direct_error = None

    try:
        mri_processor = AutoImageProcessor.from_pretrained(
            MRI_MODEL_NAME
        )

        mri_model = AutoModelForImageClassification.from_pretrained(
            MRI_MODEL_NAME
        )

        mri_model.eval()

        print("✓ Brain MRI AI model loaded using direct image classification.")
        print("MRI labels:", getattr(mri_model.config, "id2label", {}))
        return "direct"

    except Exception as error:
        direct_error = error
        print("Direct MRI model loading failed:", repr(error))
        traceback.print_exc()
        mri_processor = None
        mri_model = None

    try:
        mri_classifier = pipeline(
            "image-classification",
            model=MRI_MODEL_NAME
        )

        print("✓ Brain MRI AI model loaded using pipeline fallback.")
        return "pipeline"

    except Exception as pipeline_error:
        print("Pipeline MRI model loading also failed:", repr(pipeline_error))
        traceback.print_exc()
        raise RuntimeError(
            "MRI model could not be loaded. "
            f"Direct loader error: {direct_error}. "
            f"Pipeline error: {pipeline_error}"
        ) from pipeline_error


def run_mri_prediction(image, top_k=4):
    """Run real image classification and return pipeline-style predictions."""
    loader_type = get_mri_classifier()

    if loader_type == "pipeline":
        return mri_classifier(image, top_k=top_k)

    inputs = mri_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = mri_model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)[0]

    values, indices = torch.topk(
        probabilities,
        k=min(top_k, probabilities.shape[-1])
    )

    id2label = getattr(mri_model.config, "id2label", {}) or {}
    predictions = []

    for score, index in zip(values.tolist(), indices.tolist()):
        label = id2label.get(index, id2label.get(str(index), f"LABEL_{index}"))
        predictions.append({
            "label": str(label),
            "score": float(score)
        })

    return predictions


def get_ct_classifier():

    global ct_classifier

    if ct_classifier is None:

        print("Loading Brain CT stroke AI model...")

        ct_classifier = pipeline(
            "image-classification",
            model=CT_MODEL_NAME
        )

        print("✓ Brain CT stroke AI model loaded.")

    return ct_classifier


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
    # Older versions of the project may already have a reports
    # table with fewer/different columns. CREATE TABLE IF NOT
    # EXISTS does not modify an existing table, so we check the
    # actual schema and add any missing compatible columns.
    #
    # If the old table does not contain the required "id" column,
    # it is backed up and a fresh compatible reports table is
    # created. The old table is NOT deleted.
    # --------------------------------------------------------

    cursor.execute("PRAGMA table_info(reports)")

    report_info = cursor.fetchall()

    existing_report_columns = {
        row["name"]
        for row in report_info
    }

    required_report_columns = {
        "id": "TEXT",
        "user_email": "TEXT DEFAULT ''",
        "module": "TEXT DEFAULT 'unknown'",
        "date": "TEXT DEFAULT ''",
        "tier": "TEXT DEFAULT 'neutral'",
        "title": "TEXT DEFAULT 'Report'",
        "summary": "TEXT DEFAULT ''",
        "details": "TEXT DEFAULT '{}'",
        "created_at": "TEXT DEFAULT ''"
    }

    # The id column is essential because it is the primary key
    # used by the current API.
    if "id" not in existing_report_columns:

        backup_name = (
            "reports_legacy_"
            + datetime.now(timezone.utc).strftime(
                "%Y%m%d%H%M%S"
            )
        )

        print(
            "Old reports table detected without id column."
        )

        print(
            "Backing it up as:",
            backup_name
        )

        cursor.execute(
            f"ALTER TABLE reports RENAME TO '{backup_name}'"
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

        existing_report_columns = set()

    # Add compatible missing columns.
    for column_name, column_definition in required_report_columns.items():

        if column_name == "id":
            continue

        if column_name not in existing_report_columns:

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
# CT / MRI IMAGE AI SCREENING
# ============================================================
# IMPORTANT:
# - This block is ONLY for CT/MRI image prediction.
# - Existing migraine / brain disease / other screening routes
#   elsewhere in this file are not changed.
# - CT model: single brain CT slice -> Stroke / No Stroke.
# - MRI model: brain MRI image -> Glioma / Meningioma /
#   No Tumor / Pituitary.
# - These are research/educational screening models, NOT
#   clinical diagnostic tools.

MRI_MODEL_NAME = "amjadfqs/vit-base-patch16-224-in21k-finetuned-brain-tumor"
CT_MODEL_NAME = "Sahende/teknofest_ct_stroke_binary"

mri_processor = None
mri_model = None
ct_processor = None
ct_model = None


def normalize_model_label(label):
    """Make model labels comparable."""
    return (
        str(label or "")
        .lower()
        .strip()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


def load_mri_model():
    """Load the real MRI brain-tumor image-classification model once."""
    global mri_processor, mri_model

    if mri_processor is not None and mri_model is not None:
        return

    print("Loading Brain MRI AI model...")

    mri_processor = AutoImageProcessor.from_pretrained(
        MRI_MODEL_NAME
    )

    mri_model = AutoModelForImageClassification.from_pretrained(
        MRI_MODEL_NAME
    )

    mri_model.eval()

    print("✓ Brain MRI AI model loaded.")
    print(
        "MRI labels:",
        getattr(mri_model.config, "id2label", {})
    )


def load_ct_model():
    """Load the real brain-CT stroke classification model once."""
    global ct_processor, ct_model

    if ct_processor is not None and ct_model is not None:
        return

    print("Loading Brain CT AI model...")

    ct_processor = AutoImageProcessor.from_pretrained(
        CT_MODEL_NAME
    )

    ct_model = AutoModelForImageClassification.from_pretrained(
        CT_MODEL_NAME
    )

    ct_model.eval()

    print("✓ Brain CT AI model loaded.")
    print(
        "CT labels:",
        getattr(ct_model.config, "id2label", {})
    )


def predict_mri_image(image):
    """Return all MRI class probabilities, sorted highest first."""
    load_mri_model()

    inputs = mri_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = mri_model(**inputs)
        probabilities = F.softmax(
            outputs.logits,
            dim=-1
        )[0]

    id2label = getattr(
        mri_model.config,
        "id2label",
        {}
    ) or {}

    results = []

    for index, score in enumerate(
        probabilities.tolist()
    ):
        label = id2label.get(
            index,
            id2label.get(
                str(index),
                f"LABEL_{index}"
            )
        )

        results.append({
            "label": str(label),
            "score": float(score)
        })

    return sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )


def predict_ct_image(image):
    """Return all CT class probabilities, sorted highest first."""
    load_ct_model()

    inputs = ct_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = ct_model(**inputs)
        probabilities = F.softmax(
            outputs.logits,
            dim=-1
        )[0]

    id2label = getattr(
        ct_model.config,
        "id2label",
        {}
    ) or {}

    results = []

    for index, score in enumerate(
        probabilities.tolist()
    ):
        label = id2label.get(
            index,
            id2label.get(
                str(index),
                f"LABEL_{index}"
            )
        )

        results.append({
            "label": str(label),
            "score": float(score)
        })

    return sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )


def save_ct_mri_prediction(
    user_email,
    module,
    title,
    tier,
    summary,
    details
):
    """
    Save CT/MRI AI result directly into the existing reports table.
    Does not modify the users/report schema.
    """
    user_email = str(
        user_email or ""
    ).strip().lower()

    if not user_email:
        return None

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

    connection = get_db()

    try:
        connection.execute(
            """
            INSERT INTO reports (
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
                current_time(),
                tier,
                title,
                summary,
                json.dumps(
                    details,
                    ensure_ascii=False
                ),
                current_time()
            )
        )

        connection.commit()

        return report_id

    finally:
        connection.close()


def get_user_email_from_request():
    """
    Accept the logged-in user's email from the CT/MRI page.
    This is used only for saving the imaging result.
    """
    email = request.form.get(
        "user_email",
        ""
    )

    if not email:
        email = request.headers.get(
            "X-User-Email",
            ""
        )

    return str(
        email or ""
    ).strip().lower()


# ============================================================
# MRI IMAGE AI SCREENING
# ============================================================

@app.route(
    "/api/analyze-mri",
    methods=["POST"]
)
def analyze_mri():

    try:

        if "file" not in request.files:
            return error_response(
                "No MRI image was uploaded."
            )

        uploaded_file = request.files["file"]

        if not uploaded_file.filename:
            return error_response(
                "Please select an MRI image."
            )

        allowed_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp"
        }

        extension = os.path.splitext(
            uploaded_file.filename
        )[1].lower()

        if extension not in allowed_extensions:
            return error_response(
                "Please upload an MRI image as JPG, JPEG, PNG, WEBP or BMP."
            )

        try:
            image = Image.open(
                uploaded_file.stream
            ).convert("RGB")
        except Exception:
            return error_response(
                "The uploaded MRI image could not be opened. Please upload a valid image.",
                400
            )

        predictions = predict_mri_image(image)

        if not predictions:
            return error_response(
                "The MRI AI model did not return a prediction.",
                500
            )

        best = predictions[0]

        raw_label = normalize_model_label(
            best["label"]
        )

        confidence = round(
            best["score"] * 100,
            2
        )

        label_map = {
            "glioma": "Glioma",
            "meningioma": "Meningioma",
            "notumor": "No Tumor",
            "pituitary": "Pituitary Tumor"
        }

        prediction = label_map.get(
            raw_label,
            str(best["label"])
        )

        tumor_labels = {
            "glioma",
            "meningioma",
            "pituitary"
        }

        if raw_label == "notumor":

            level = "green"

            possible_condition = (
                "The MRI screening model did not identify "
                "a tumor pattern in this image."
            )

            recommendation = (
                "No tumor pattern was identified by this "
                "screening model. This does not rule out "
                "other neurological conditions. Clinical "
                "evaluation is recommended if symptoms are present."
            )

        elif raw_label in tumor_labels:

            level = (
                "red"
                if confidence >= 80
                else "yellow"
            )

            possible_condition = (
                f"{prediction} pattern detected"
            )

            recommendation = (
                f"The MRI screening model identified a "
                f"{prediction}-like pattern with "
                f"{confidence}% model confidence. "
                "This is a screening result, not a diagnosis. "
                "A qualified radiologist/neurologist should "
                "review the complete MRI study."
            )

        else:

            level = "yellow"

            possible_condition = (
                "Unrecognized MRI model output"
            )

            recommendation = (
                "The MRI model returned a class that this "
                "application does not have a clinical label for. "
                "Please have the MRI reviewed by a qualified professional."
            )

        top_predictions = []

        for item in predictions:

            item_label = normalize_model_label(
                item["label"]
            )

            readable_label = label_map.get(
                item_label,
                str(item["label"])
            )

            top_predictions.append({
                "label": readable_label,
                "confidence": round(
                    item["score"] * 100,
                    2
                )
            })

        details = {
            "Scan Type": "MRI",
            "File Name": uploaded_file.filename,
            "Prediction": prediction,
            "Confidence": confidence,
            "Possible Condition": possible_condition,
            "Top Predictions": top_predictions,
            "Recommendation": recommendation,
            "Model": MRI_MODEL_NAME,
            "Note": (
                "Single-image research screening. "
                "Not a medical diagnosis."
            )
        }

        report_id = save_ct_mri_prediction(
            user_email=get_user_email_from_request(),
            module="ct_mri",
            title="MRI AI Screening",
            tier=level,
            summary=(
                f"{prediction} - "
                f"{confidence}% confidence"
            ),
            details=details
        )

        return jsonify({
            "success": True,
            "module": "MRI AI Screening",
            "scan_type": "MRI",
            "prediction": prediction,
            "possible_condition": possible_condition,
            "confidence": confidence,
            "level": level,
            "top_predictions": top_predictions,
            "recommendation": recommendation,
            "report_id": report_id,
            "database_saved": bool(report_id),
            "model": MRI_MODEL_NAME,
            "disclaimer": (
                "Research/educational AI screening only. "
                "This result is not a medical diagnosis and "
                "must be confirmed by a qualified radiologist "
                "or neurologist."
            )
        })

    except Exception as error:

        print(
            "MRI AI ANALYSIS ERROR:",
            repr(error)
        )

        traceback.print_exc()

        return error_response(
            "MRI AI analysis failed. "
            f"{type(error).__name__}: {error}",
            500
        )


# ============================================================
# CT IMAGE AI SCREENING
# ============================================================

@app.route(
    "/api/analyze-ct",
    methods=["POST"]
)
def analyze_ct():

    try:

        if "file" not in request.files:
            return error_response(
                "No CT image was uploaded."
            )

        uploaded_file = request.files["file"]

        if not uploaded_file.filename:
            return error_response(
                "Please select a CT image."
            )

        allowed_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp"
        }

        extension = os.path.splitext(
            uploaded_file.filename
        )[1].lower()

        if extension not in allowed_extensions:
            return error_response(
                "Please upload a CT image as JPG, JPEG, PNG, WEBP or BMP."
            )

        try:
            image = Image.open(
                uploaded_file.stream
            ).convert("RGB")
        except Exception:
            return error_response(
                "The uploaded CT image could not be opened. Please upload a valid image.",
                400
            )

        predictions = predict_ct_image(image)

        if not predictions:
            return error_response(
                "The CT AI model did not return a prediction.",
                500
            )

        best = predictions[0]

        raw_label = normalize_model_label(
            best["label"]
        )

        confidence = round(
            best["score"] * 100,
            2
        )

        # The CT model is binary:
        # 0 = No Stroke
        # 1 = Stroke.
        is_stroke = (
            raw_label in {
                "1",
                "label1",
                "stroke",
                "positive",
                "10"
            }
            or "stroke" in raw_label
        )

        if is_stroke:

            prediction = "Stroke"
            possible_condition = (
                "Stroke-like pattern detected in the CT image."
            )

            level = (
                "red"
                if confidence >= 80
                else "yellow"
            )

            recommendation = (
                f"The CT screening model identified a "
                f"stroke-like pattern with {confidence}% "
                "model confidence. If the patient has sudden "
                "weakness, facial drooping, speech difficulty, "
                "vision changes, severe dizziness or loss of "
                "balance, seek emergency medical care immediately. "
                "A qualified radiologist/doctor must confirm the scan."
            )

        else:

            prediction = "No Stroke"
            possible_condition = (
                "No stroke pattern was identified by this "
                "CT screening model."
            )

            level = "green"

            recommendation = (
                "The CT screening model did not identify a "
                "stroke pattern in this image. This does not "
                "rule out other brain or neurological conditions. "
                "Clinical evaluation is recommended when symptoms "
                "are present."
            )

        top_predictions = []

        for item in predictions:

            item_label = normalize_model_label(
                item["label"]
            )

            if (
                item_label in {
                    "1",
                    "label1",
                    "stroke",
                    "positive",
                    "10"
                }
                or "stroke" in item_label
            ):
                readable_label = "Stroke"
            else:
                readable_label = "No Stroke"

            top_predictions.append({
                "label": readable_label,
                "confidence": round(
                    item["score"] * 100,
                    2
                )
            })

        details = {
            "Scan Type": "CT",
            "File Name": uploaded_file.filename,
            "Prediction": prediction,
            "Confidence": confidence,
            "Possible Condition": possible_condition,
            "Top Predictions": top_predictions,
            "Recommendation": recommendation,
            "Model": CT_MODEL_NAME,
            "Note": (
                "Single axial brain CT slice research screening. "
                "Not a medical diagnosis."
            )
        }

        report_id = save_ct_mri_prediction(
            user_email=get_user_email_from_request(),
            module="ct_mri",
            title="CT AI Stroke Screening",
            tier=level,
            summary=(
                f"{prediction} - "
                f"{confidence}% confidence"
            ),
            details=details
        )

        return jsonify({
            "success": True,
            "module": "CT AI Stroke Screening",
            "scan_type": "CT",
            "prediction": prediction,
            "possible_condition": possible_condition,
            "confidence": confidence,
            "level": level,
            "top_predictions": top_predictions,
            "recommendation": recommendation,
            "report_id": report_id,
            "database_saved": bool(report_id),
            "model": CT_MODEL_NAME,
            "disclaimer": (
                "Research/educational AI screening only. "
                "This result is not a medical diagnosis and "
                "must be confirmed by a qualified radiologist "
                "or neurologist."
            )
        })

    except Exception as error:

        print(
            "CT AI ANALYSIS ERROR:",
            repr(error)
        )

        traceback.print_exc()

        return error_response(
            "CT AI analysis failed. "
            f"{type(error).__name__}: {error}",
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