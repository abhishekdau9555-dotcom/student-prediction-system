import os
import time
import sqlite3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
from reportlab.lib       import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from sklearn.tree import DecisionTreeClassifier
from werkzeug.utils import secure_filename

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "ai_student_secret_2024"   # Session ke liye zaroori

UPLOAD_FOLDER = "static/uploads"
GRAPH_FOLDER  = "static/graphs"
PDF_FOLDER    = "static/reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

for folder in (UPLOAD_FOLDER, GRAPH_FOLDER, PDF_FOLDER):
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# ── Login Credentials (Admin + Teachers) ─────────────────────────────────────
USERS = {
    "admin":   {"password": "admin123",   "role": "Admin",   "name": "Administrator",
                "email": "admin@example.com"},
    "teacher": {"password": "teacher123", "role": "Teacher", "name": "Class Teacher",
                "email": "teacher@example.com"},
    "hod":     {"password": "hod123",     "role": "HOD",     "name": "Head of Department",
                "email": "hod@example.com"},
}

# ── Email / SMTP Configuration (Feature 5) ───────────────────────────────────
# IMPORTANT: Yaha apna Gmail address aur App Password daalo.
# App Password kaise banaye:
#   1. Google Account -> Security -> 2-Step Verification ON karo
#   2. "App Passwords" search karo -> naya app password generate karo
#   3. Wo 16-digit password yaha SENDER_PASSWORD mein daalo (apna normal Gmail password NAHI)
SMTP_SERVER       = "smtp.gmail.com"
SMTP_PORT         = 587
SENDER_EMAIL      = "abhishekdau9555@gmail.com"   # <-- aapka Gmail
SENDER_PASSWORD   = "your_16_digit_app_password"   # <-- App Password yaha daalo (neeche steps dekho)
EMAIL_ENABLED     = True   # False kar do agar email feature temporarily band karni ho

# ── Prediction History (RAM cache — synced with SQLite on every request) ────
history = []

# ── SQLite Database Setup (Feature 6) ────────────────────────────────────────
DB_PATH = "students.db"

def init_db():
    """Create the students table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT,
            email           TEXT,
            course          TEXT,
            branch          TEXT,
            semester        TEXT,
            photo           TEXT,
            attendance      INTEGER,
            internal        INTEGER,
            assignment      INTEGER,
            maths           INTEGER,
            cn              INTEGER,
            ml              INTEGER,
            communication   INTEGER,
            result          TEXT,
            risk            TEXT,
            percentage      REAL,
            grade           TEXT,
            advice          TEXT,
            graph_path      TEXT,
            predicted_by    TEXT,
            placement_score REAL,
            placement_category TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_record_to_db(record: dict) -> int:
    """Insert a prediction record into SQLite. Returns the new row id."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO students (
            name, email, course, branch, semester, photo,
            attendance, internal, assignment, maths, cn, ml, communication,
            result, risk, percentage, grade, advice, graph_path, predicted_by,
            placement_score, placement_category
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record.get("name"), record.get("email"), record.get("course"),
        record.get("branch"), record.get("semester"), record.get("photo"),
        record.get("attendance"), record.get("internal"), record.get("assignment"),
        record.get("maths"), record.get("cn"), record.get("ml"),
        record.get("communication"), record.get("result"), record.get("risk"),
        record.get("percentage"), record.get("grade"), record.get("advice"),
        record.get("graph_path"), record.get("predicted_by"),
        record.get("placement_score"), record.get("placement_category"),
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def load_history_from_db(search_query: str = "") -> list:
    """
    Load all student records from SQLite, most recent first.
    If search_query is given, filters by name/email/branch/course (case-insensitive).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if search_query:
        like = f"%{search_query}%"
        cur.execute("""
            SELECT * FROM students
            WHERE name LIKE ? OR email LIKE ? OR branch LIKE ? OR course LIKE ?
            ORDER BY id DESC
        """, (like, like, like, like))
    else:
        cur.execute("SELECT * FROM students ORDER BY id DESC")

    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_record_by_id(record_id: int):
    """Fetch a single student record by its database id."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# Initialize the database and load any existing records into RAM cache
init_db()
history = load_history_from_db()


# ── Load Dataset & Train Model ────────────────────────────────────────────────
data = pd.read_csv("data/students.csv")

FEATURES = ["Attendance", "Internal", "Assignment",
            "Maths", "CN", "ML", "Communication"]
X = data[FEATURES]
y = data["Result"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    """Decorator — agar login nahi hai toh /login par bhejo."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please login to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """Decorator — sirf Admin/HOD role dashboard dekh sakta hai."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please login to access this page.", "warning")
            return redirect(url_for("login"))
        if session.get("role") not in ("Admin", "HOD"):
            flash("Access denied. Only Admin/HOD can view the Dashboard.", "warning")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated

def calculate_dashboard_stats():
    """Calculate all statistics needed for the admin dashboard from history."""
    total = len(history)

    if total == 0:
        return {
            "total_students": 0, "pass_count": 0, "fail_count": 0,
            "pass_rate": 0, "avg_percentage": 0,
            "high_risk": 0, "medium_risk": 0, "low_risk": 0,
            "grade_labels": [], "grade_counts": [],
            "branch_labels": [], "branch_avg": [],
            "trend_labels": [], "trend_values": [],
            "top_students": [], "at_risk_students": [],
            "avg_placement_score": 0,
            "placement_labels": [], "placement_counts": [],
            "placement_ready_count": 0,
        }

    pass_count = sum(1 for r in history if r["result"] == "Pass")
    fail_count = total - pass_count
    pass_rate  = round((pass_count / total) * 100, 1)
    avg_pct    = round(sum(r["percentage"] for r in history) / total, 1)

    high_risk   = sum(1 for r in history if r["risk"] == "High Risk")
    medium_risk = sum(1 for r in history if r["risk"] == "Medium Risk")
    low_risk    = sum(1 for r in history if r["risk"] == "Low Risk")

    # Grade distribution
    grade_order  = ["A+", "A", "B", "C", "D"]
    grade_counts_map = {g: 0 for g in grade_order}
    for r in history:
        if r["grade"] in grade_counts_map:
            grade_counts_map[r["grade"]] += 1
    grade_labels = grade_order
    grade_counts = [grade_counts_map[g] for g in grade_order]

    # Branch-wise average percentage
    branch_data = {}
    for r in history:
        b = r.get("branch", "Unknown") or "Unknown"
        branch_data.setdefault(b, []).append(r["percentage"])
    branch_labels = list(branch_data.keys())
    branch_avg    = [round(sum(v)/len(v), 1) for v in branch_data.values()]

    # Trend — percentage over each submission (last 15)
    recent = history[-15:]
    trend_labels = [r["name"][:10] for r in recent]
    trend_values = [r["percentage"] for r in recent]

    # Top 5 performers
    top_students = sorted(history, key=lambda r: r["percentage"], reverse=True)[:5]

    # At-risk students (High Risk only, latest 5)
    at_risk_students = [r for r in history if r["risk"] == "High Risk"][-5:]

    # Placement readiness stats (Feature 4)
    placement_scores = [r["placement_score"] for r in history if "placement_score" in r]
    avg_placement_score = round(sum(placement_scores) / len(placement_scores), 1) if placement_scores else 0

    placement_category_order = ["Excellent", "Good", "Needs Improvement", "Not Ready"]
    placement_counts_map = {c: 0 for c in placement_category_order}
    for r in history:
        cat = r.get("placement_category")
        if cat in placement_counts_map:
            placement_counts_map[cat] += 1
    placement_labels = placement_category_order
    placement_counts = [placement_counts_map[c] for c in placement_category_order]

    placement_ready_count = placement_counts_map["Excellent"] + placement_counts_map["Good"]

    return {
        "total_students": total, "pass_count": pass_count, "fail_count": fail_count,
        "pass_rate": pass_rate, "avg_percentage": avg_pct,
        "high_risk": high_risk, "medium_risk": medium_risk, "low_risk": low_risk,
        "grade_labels": grade_labels, "grade_counts": grade_counts,
        "branch_labels": branch_labels, "branch_avg": branch_avg,
        "trend_labels": trend_labels, "trend_values": trend_values,
        "top_students": top_students, "at_risk_students": at_risk_students,
        "avg_placement_score": avg_placement_score,
        "placement_labels": placement_labels, "placement_counts": placement_counts,
        "placement_ready_count": placement_ready_count,
    }

def compute_grade(pct: float) -> str:
    if pct >= 90: return "A+"
    if pct >= 80: return "A"
    if pct >= 70: return "B"
    if pct >= 60: return "C"
    return "D"

def compute_advice(attendance: int, maths: int, result: str) -> str:
    if attendance < 75:  return "Improve your Attendance immediately."
    if maths < 50:       return "Practice Mathematics regularly."
    if result == "Pass": return "Excellent Performance! Keep it up."
    return "Counselling recommended. Speak to your mentor."

def compute_risk(attendance: int) -> str:
    if attendance < 60: return "High Risk"
    if attendance < 75: return "Medium Risk"
    return "Low Risk"

def compute_placement_score(internal, assignment, maths, cn, ml,
                             communication, attendance) -> dict:
    """
    Calculate Placement Readiness Score (0-100) using a weighted formula:
      - Technical Skills (40%)   -> average of CN + ML
      - Academic Strength (25%) -> average of Maths + Internal + Assignment
      - Communication (20%)     -> Communication marks
      - Consistency (15%)       -> Attendance

    Returns a dict with the overall score, category, color, advice,
    and the individual component breakdown (for the gauge/chart).
    """
    technical_avg  = (cn + ml) / 2
    academic_avg   = (maths + internal + assignment) / 3
    communication_score = communication
    consistency_score   = attendance

    placement_score = round(
        (technical_avg        * 0.40) +
        (academic_avg         * 0.25) +
        (communication_score  * 0.20) +
        (consistency_score    * 0.15),
        1
    )
    placement_score = min(max(placement_score, 0), 100)

    if placement_score >= 85:
        category = "Excellent"
        color    = "#198754"   # green
        badge    = "🟢"
        advice   = "Placement Ready! You're a strong candidate — start applying to top companies."
    elif placement_score >= 70:
        category = "Good"
        color    = "#0d6efd"   # blue
        badge    = "🔵"
        advice   = "Almost Ready. Polish your weaker areas before placement drives begin."
    elif placement_score >= 50:
        category = "Needs Improvement"
        color    = "#ffc107"   # yellow
        badge    = "🟡"
        advice   = "Needs Improvement. Focus on technical skills and mock interviews."
    else:
        category = "Not Ready"
        color    = "#dc3545"   # red
        badge    = "🔴"
        advice   = "Not Ready Yet. Dedicate serious time to technical and communication skills."

    # Identify weakest area for targeted advice
    components = {
        "Technical Skills":  round(technical_avg, 1),
        "Academic Strength": round(academic_avg, 1),
        "Communication":     round(communication_score, 1),
        "Consistency":       round(consistency_score, 1),
    }
    weakest_area = min(components, key=components.get)

    return {
        "score":         placement_score,
        "category":      category,
        "color":         color,
        "badge":         badge,
        "advice":        advice,
        "weakest_area":  weakest_area,
        "components":    components,
    }

def save_graph(maths, cn, ml, internal, assignment, attendance) -> str:
    subjects = ["Maths", "CN", "ML", "Internal", "Assignment", "Attendance"]
    marks    = [maths, cn, ml, internal, assignment, attendance]

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
    bars = ax.bar(subjects, marks, color=bar_colors, edgecolor="white", linewidth=0.8)

    ax.set_title("Student Marks Analysis", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Marks / Score")
    ax.set_ylim(0, 115)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(75, color="green",  linestyle="--", linewidth=0.8, alpha=0.6, label="Good (75)")
    ax.axhline(50, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="Average (50)")
    ax.legend(fontsize=8)

    for bar, val in zip(bars, marks):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2, str(val),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    filename  = f"graph_{int(time.time() * 1000)}.png"
    full_path = os.path.join(GRAPH_FOLDER, filename)
    fig.tight_layout()
    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return full_path


# ── Smart PDF Generator ───────────────────────────────────────────────────────
def generate_pdf(student_data: dict) -> str:
    filename  = f"report_{int(time.time() * 1000)}.pdf"
    full_path = os.path.join(PDF_FOLDER, filename)

    width, height = A4
    doc = SimpleDocTemplate(
        full_path, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm,   bottomMargin=2*cm,
    )
    story  = []
    styles = getSampleStyleSheet()
    cw     = width - 3*cm

    def st(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    s_white_title = st("WT",  fontSize=18, textColor=colors.white,
                        alignment=TA_CENTER, fontName="Helvetica-Bold")
    s_white_sub   = st("WS",  fontSize=10, textColor=colors.HexColor("#cce0ff"),
                        alignment=TA_CENTER)
    s_section     = st("SEC", fontSize=13, textColor=colors.HexColor("#1a3a5c"),
                        fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4)
    s_label       = st("LBL", fontName="Helvetica-Bold",
                        textColor=colors.HexColor("#1a3a5c"), fontSize=10)
    s_value       = st("VAL", fontSize=10)
    s_center      = st("CTR", fontSize=10, alignment=TA_CENTER)
    s_advice      = st("ADV", fontSize=11, fontName="Helvetica-Bold",
                        textColor=colors.HexColor("#7d4e00"))
    s_hdr         = st("MH",  fontName="Helvetica-Bold",
                        textColor=colors.white, alignment=TA_CENTER, fontSize=10)

    # Header
    hdr = Table([
        [Paragraph("AI Based Student Prediction &amp; Counselling System", s_white_title)],
        [Paragraph("Automated Academic Performance Report", s_white_sub)],
    ], colWidths=[cw])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#1a3a5c")),
        ("TOPPADDING",    (0,0),(-1,-1), 16),
        ("BOTTOMPADDING", (0,0),(-1,-1), 16),
        ("LEFTPADDING",   (0,0),(-1,-1), 20),
    ]))
    story += [hdr, Spacer(1, 0.4*cm)]

    # Student Info + Photo
    story.append(Paragraph("Student Information", s_section))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a3a5c"), spaceAfter=6))

    info_rows = [
        ["Name",     student_data.get("name","—"),     "Course",    student_data.get("course","—")],
        ["Email",    student_data.get("email","—"),     "Branch",    student_data.get("branch","—")],
        ["Semester", student_data.get("semester","—"), "Date",      time.strftime("%d %B %Y")],
    ]
    info_data = [[Paragraph(r[0],s_label), Paragraph(str(r[1]),s_value),
                  Paragraph(r[2],s_label), Paragraph(str(r[3]),s_value)]
                 for r in info_rows]

    iw = (cw - 3.5*cm) / 4
    info_tbl = Table(info_data, colWidths=[iw*0.7, iw*1.3, iw*0.7, iw*1.3])
    info_tbl.setStyle(TableStyle([
        ("GRID",         (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND",   (0,0),(0,-1),  colors.HexColor("#eaf0fb")),
        ("BACKGROUND",   (2,0),(2,-1),  colors.HexColor("#eaf0fb")),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
    ]))

    photo_cell = Paragraph("No Photo", s_center)
    pf = student_data.get("photo","")
    if pf:
        pp = os.path.join(UPLOAD_FOLDER, pf)
        if os.path.exists(pp):
            try:
                photo_cell = RLImage(pp, width=2.8*cm, height=3.2*cm)
            except Exception:
                pass

    combo = Table([[info_tbl, photo_cell]], colWidths=[cw-3.5*cm, 3.5*cm])
    combo.setStyle(TableStyle([
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING", (1,0),(1,0),   10),
    ]))
    story += [combo, Spacer(1, 0.3*cm)]

    # Marks Table
    story.append(Paragraph("Academic Marks", s_section))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a3a5c"), spaceAfter=6))

    def row_bg(val):
        if val >= 75: return colors.HexColor("#d4edda")
        if val >= 50: return colors.HexColor("#fff3cd")
        return colors.HexColor("#f8d7da")

    subjects = [
        ("Attendance (%)",         student_data.get("attendance",    0)),
        ("Internal Marks",         student_data.get("internal",      0)),
        ("Assignment",             student_data.get("assignment",    0)),
        ("Mathematics",            student_data.get("maths",         0)),
        ("Computer Networks (CN)", student_data.get("cn",            0)),
        ("Machine Learning (ML)",  student_data.get("ml",            0)),
        ("Communication",          student_data.get("communication", 0)),
    ]

    m_data = [[Paragraph("Subject", s_hdr), Paragraph("Score", s_hdr),
               Paragraph("Out of", s_hdr), Paragraph("Status", s_hdr)]]
    bgs = []
    for subj, val in subjects:
        status = "Good" if val >= 75 else ("Average" if val >= 50 else "Poor")
        m_data.append([Paragraph(subj, s_value), Paragraph(str(val), s_center),
                        Paragraph("100", s_center), Paragraph(status, s_center)])
        bgs.append(row_bg(val))

    mw = cw / 4
    m_tbl = Table(m_data, colWidths=[mw*1.8, mw*0.8, mw*0.6, mw*0.8])
    m_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#1a3a5c")),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), bgs),
    ]))
    story += [m_tbl, Spacer(1, 0.3*cm)]

    # Result Summary
    story.append(Paragraph("Result Summary", s_section))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a3a5c"), spaceAfter=6))

    result     = student_data.get("result",     "—")
    percentage = student_data.get("percentage", 0)
    grade      = student_data.get("grade",      "—")
    risk       = student_data.get("risk",       "—")

    r_color  = colors.HexColor("#d4edda") if result == "Pass" else colors.HexColor("#f8d7da")
    rk_color = (colors.HexColor("#f8d7da") if "High"   in risk else
                colors.HexColor("#fff3cd") if "Medium" in risk else
                colors.HexColor("#d4edda"))

    s_res_val = st("RV", fontSize=14, fontName="Helvetica-Bold",
                   textColor=colors.HexColor("#198754") if result=="Pass"
                             else colors.HexColor("#dc3545"),
                   alignment=TA_CENTER)

    sw = cw / 4
    sum_data = [
        [Paragraph("Prediction", s_label), Paragraph(result, s_res_val),
         Paragraph("Percentage", s_label), Paragraph(f"{percentage}%", s_center)],
        [Paragraph("Grade",      s_label), Paragraph(grade, s_center),
         Paragraph("Risk Level", s_label), Paragraph(risk, s_center)],
    ]
    sum_tbl = Table(sum_data, colWidths=[sw*1.1, sw*0.9, sw*1.1, sw*0.9])
    sum_tbl.setStyle(TableStyle([
        ("GRID",         (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND",   (0,0),(0,-1),  colors.HexColor("#eaf0fb")),
        ("BACKGROUND",   (2,0),(2,-1),  colors.HexColor("#eaf0fb")),
        ("BACKGROUND",   (1,0),(1,0),   r_color),
        ("BACKGROUND",   (3,1),(3,1),   rk_color),
        ("TOPPADDING",   (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story += [sum_tbl, Spacer(1, 0.2*cm)]

    # Progress Bar
    bar_color = (colors.HexColor("#198754") if percentage >= 75 else
                 colors.HexColor("#ffc107") if percentage >= 50 else
                 colors.HexColor("#dc3545"))
    filled_w = max(cw * (percentage / 100), 1.2*cm)
    s_pct = st("PCT", fontSize=9, textColor=colors.white, fontName="Helvetica-Bold")
    bar_fill = Table([[Paragraph(f"  {percentage}%", s_pct)]],
                     colWidths=[filled_w], rowHeights=[18])
    bar_fill.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), bar_color),
        ("TOPPADDING",    (0,0),(-1,-1), 2),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ]))
    story.append(Paragraph("Overall Performance", s_label))
    story.append(Spacer(1, 0.1*cm))
    story.append(bar_fill)
    story.append(Spacer(1, 0.3*cm))

    # ── Placement Readiness Score (Feature 4) ──────────────────
    placement_score    = student_data.get("placement_score")
    placement_category = student_data.get("placement_category", "—")

    if placement_score is not None:
        story.append(Paragraph("Placement Readiness Score", s_section))
        story.append(HRFlowable(width="100%", thickness=1,
                                 color=colors.HexColor("#1a3a5c"), spaceAfter=6))

        p_color = (colors.HexColor("#198754") if placement_score >= 85 else
                   colors.HexColor("#0d6efd") if placement_score >= 70 else
                   colors.HexColor("#ffc107") if placement_score >= 50 else
                   colors.HexColor("#dc3545"))

        s_p_score = st("PSCORE", fontSize=20, fontName="Helvetica-Bold",
                       textColor=p_color, alignment=TA_CENTER)
        s_p_cat   = st("PCAT", fontSize=11, fontName="Helvetica-Bold",
                       textColor=colors.white, alignment=TA_CENTER)

        # Score box + category badge side by side
        cat_badge = Table([[Paragraph(placement_category, s_p_cat)]],
                          colWidths=[cw - 3.5*cm])
        cat_badge.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), p_color),
            ("TOPPADDING",    (0,0),(-1,-1), 10),
            ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ]))

        score_cell = Table([[Paragraph(f"{placement_score}", s_p_score)],
                            [Paragraph("out of 100", s_center)]],
                           colWidths=[3.5*cm])
        score_cell.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#eaf0fb")),
            ("BOX",        (0,0),(-1,-1), 1, p_color),
            ("TOPPADDING", (0,0),(-1,-1), 8),
            ("ALIGN",      (0,0),(-1,-1), "CENTER"),
        ]))

        placement_row = Table([[score_cell, cat_badge]],
                              colWidths=[3.5*cm, cw - 3.5*cm])
        placement_row.setStyle(TableStyle([
            ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
            ("LEFTPADDING", (1,0),(1,0), 10),
        ]))
        story += [placement_row, Spacer(1, 0.3*cm)]

    # Advice Box
    advice = student_data.get("advice", "—")
    story.append(Paragraph("Counselling Advice", s_section))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a3a5c"), spaceAfter=6))
    adv_tbl = Table([[Paragraph(f"  {advice}", s_advice)]], colWidths=[cw])
    adv_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#fff3cd")),
        ("BOX",           (0,0),(-1,-1), 1.5, colors.HexColor("#ffc107")),
        ("TOPPADDING",    (0,0),(-1,-1), 12),
        ("BOTTOMPADDING", (0,0),(-1,-1), 12),
        ("LEFTPADDING",   (0,0),(-1,-1), 14),
    ]))
    story += [adv_tbl, Spacer(1, 0.3*cm)]

    # Graph
    gp = student_data.get("graph_path", "")
    if gp and os.path.exists(gp):
        story.append(Paragraph("Marks Analysis Graph", s_section))
        story.append(HRFlowable(width="100%", thickness=1,
                                 color=colors.HexColor("#1a3a5c"), spaceAfter=6))
        try:
            story.append(RLImage(gp, width=cw, height=6*cm))
        except Exception:
            pass

    # Footer
    def add_footer(cv, d):
        cv.saveState()
        cv.setFont("Helvetica", 8)
        cv.setFillColor(colors.HexColor("#888888"))
        cv.line(1.5*cm, 1.5*cm, width-1.5*cm, 1.5*cm)
        cv.drawCentredString(
            width/2, 1.0*cm,
            f"AI Student Prediction System  |  "
            f"Generated on {time.strftime('%d %B %Y at %I:%M %p')}  |  Confidential"
        )
        cv.drawString(1.5*cm, 1.0*cm, f"Page {d.page}")
        cv.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return full_path


# ── Email Notification System (Feature 5) ────────────────────────────────────
def send_email_notification(student_data: dict, pdf_path: str = None) -> dict:
    """
    Sends a result-notification email to the student (if email provided)
    and CCs the teacher/admin who ran the prediction.
    Attaches the PDF report if pdf_path is given.

    Returns: {"sent": bool, "error": str or None}
    """
    if not EMAIL_ENABLED:
        return {"sent": False, "error": "Email feature is disabled (EMAIL_ENABLED=False)."}

    student_email = student_data.get("email", "").strip()
    if not student_email:
        return {"sent": False, "error": "No student email provided."}

    result      = student_data.get("result", "—")
    name        = student_data.get("name", "Student")
    percentage  = student_data.get("percentage", 0)
    grade       = student_data.get("grade", "—")
    risk        = student_data.get("risk", "—")
    advice      = student_data.get("advice", "—")
    is_high_risk = (risk == "High Risk")

    # Find the recipient (admin/teacher) who ran this prediction
    predicted_by_name = student_data.get("predicted_by", "")
    cc_email = None
    for uname, info in USERS.items():
        if info["name"] == predicted_by_name:
            cc_email = info.get("email")
            break

    # ── Build the email ────────────────────────────────────────────
    subject = (
        f"⚠️ URGENT: Attendance Alert for {name}" if is_high_risk
        else f"📊 Your Academic Result — {name}"
    )

    result_color = "#198754" if result == "Pass" else "#dc3545"
    risk_banner  = (
        f"<div style='background:#f8d7da; border-left:5px solid #dc3545; "
        f"padding:14px 18px; border-radius:8px; margin-bottom:18px;'>"
        f"<strong>⚠️ High Risk Alert:</strong> Attendance and performance indicate "
        f"a high risk of academic difficulty. Immediate counselling is recommended."
        f"</div>"
    ) if is_high_risk else ""

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background:#f0f4f8; padding:24px;">
      <div style="max-width:600px; margin:auto; background:white; border-radius:14px;
                  overflow:hidden; box-shadow:0 4px 18px rgba(0,0,0,0.08);">

        <div style="background:#1a3a5c; color:white; padding:24px; text-align:center;">
          <h2 style="margin:0;">🎓 AI Student Prediction System</h2>
          <p style="margin:4px 0 0; opacity:0.8; font-size:13px;">Academic Performance Notification</p>
        </div>

        <div style="padding:24px;">
          <p>Dear <strong>{name}</strong>,</p>
          <p>Your latest academic prediction has been generated. Here is a summary:</p>

          {risk_banner}

          <table style="width:100%; border-collapse:collapse; margin:16px 0;">
            <tr>
              <td style="padding:10px; background:#eaf0fb; font-weight:bold;">Prediction</td>
              <td style="padding:10px; color:{result_color}; font-weight:bold;">{result}</td>
            </tr>
            <tr>
              <td style="padding:10px; background:#eaf0fb; font-weight:bold;">Percentage</td>
              <td style="padding:10px;">{percentage}%</td>
            </tr>
            <tr>
              <td style="padding:10px; background:#eaf0fb; font-weight:bold;">Grade</td>
              <td style="padding:10px;">{grade}</td>
            </tr>
            <tr>
              <td style="padding:10px; background:#eaf0fb; font-weight:bold;">Risk Level</td>
              <td style="padding:10px;">{risk}</td>
            </tr>
          </table>

          <div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px;
                      padding:14px 18px; margin:16px 0;">
            💡 <strong>Advice:</strong> {advice}
          </div>

          <p style="font-size:13px; color:#888;">
            The detailed PDF report is attached to this email.
          </p>

          <p style="margin-top:24px;">Regards,<br><strong>AI Student Prediction &amp; Counselling System</strong></p>
        </div>

        <div style="background:#f5f5f5; text-align:center; padding:12px; font-size:11px; color:#999;">
          This is an automated email. Please do not reply directly.
        </div>

      </div>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = student_email
        if cc_email:
            msg["Cc"] = cc_email

        msg.attach(MIMEText(html_body, "html"))

        # Attach PDF report if available
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                part = MIMEApplication(f.read(), Name="student_report.pdf")
                part["Content-Disposition"] = 'attachment; filename="student_report.pdf"'
                msg.attach(part)

        recipients = [student_email] + ([cc_email] if cc_email else [])

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

        return {"sent": True, "error": None}

    except smtplib.SMTPAuthenticationError:
        return {"sent": False, "error": "Email authentication failed. Check SENDER_EMAIL/SENDER_PASSWORD (use an App Password, not your normal Gmail password)."}
    except smtplib.SMTPException as e:
        return {"sent": False, "error": f"SMTP error: {e}"}
    except Exception as e:
        return {"sent": False, "error": f"Unexpected error: {e}"}


# ── Routes ────────────────────────────────────────────────────────────────────

# LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("home"))

    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username in USERS and USERS[username]["password"] == password:
            session["username"] = username
            session["role"]     = USERS[username]["role"]
            session["name"]     = USERS[username]["name"]
            flash(f"Welcome, {USERS[username]['name']}!", "success")
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password. Please try again."

    return render_template("login.html", error=error)


# LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


# HOME — prediction page (login required)
@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    # Feature 7: Search Student Record (via ?search=... query param)
    search_query = request.args.get("search", "").strip()
    display_history = load_history_from_db(search_query) if search_query else history

    ctx = dict(
        result="", advice="", risk="", name="", email="",
        course="", branch="", semester="", photo="",
        percentage=0, grade="", graph_path="",
        history=display_history, error="",
        logged_in_name=session.get("name",""),
        logged_in_role=session.get("role",""),
        placement=None,
        email_status=None,
        search_query=search_query,
        search_results_count=len(display_history) if search_query else None,
    )

    if request.method == "POST":
        try:
            ctx["name"]     = request.form["name"].strip()
            ctx["email"]    = request.form["email"].strip()
            ctx["course"]   = request.form["course"].strip()
            ctx["branch"]   = request.form["branch"].strip()
            ctx["semester"] = request.form["semester"].strip()

            send_email_flag = request.form.get("send_email") == "on"

            file = request.files.get("photo")
            if file and file.filename != "" and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                ctx["photo"] = filename

            def get_int(field):
                val = int(request.form[field])
                if not (0 <= val <= 100):
                    raise ValueError(f"{field} must be between 0 and 100.")
                return val

            attendance    = get_int("attendance")
            internal      = get_int("internal")
            assignment    = get_int("assignment")
            maths         = get_int("maths")
            cn            = get_int("cn")
            ml            = get_int("ml")
            communication = get_int("communication")

            result     = model.predict([[attendance, internal, assignment,
                                         maths, cn, ml, communication]])[0]
            percentage = min(round((attendance+internal+assignment+
                                    maths+cn+ml+communication)/7, 2), 100.0)
            grade      = compute_grade(percentage)
            advice     = compute_advice(attendance, maths, result)
            risk       = compute_risk(attendance)

            # Feature 4: Placement Readiness Score
            placement = compute_placement_score(
                internal, assignment, maths, cn, ml, communication, attendance
            )

            graph_abs = save_graph(maths, cn, ml, internal, assignment, attendance)
            graph_rel = graph_abs.replace("\\", "/")

            ctx.update(result=result, advice=advice, risk=risk,
                       percentage=percentage, grade=grade,
                       graph_path=graph_rel, placement=placement)

            record = {
                "name": ctx["name"], "email": ctx["email"],
                "course": ctx["course"], "branch": ctx["branch"],
                "semester": ctx["semester"], "photo": ctx["photo"],
                "result": result, "risk": risk,
                "percentage": percentage, "grade": grade,
                "attendance": attendance, "internal": internal,
                "assignment": assignment, "maths": maths,
                "cn": cn, "ml": ml, "communication": communication,
                "advice": advice, "graph_path": graph_abs,
                "predicted_by": session.get("name", "Unknown"),
                "placement_score":    placement["score"],
                "placement_category": placement["category"],
            }
            history.append(record)

            # Feature 6: Save permanently to SQLite database
            try:
                db_id = save_record_to_db(record)
                record["id"] = db_id
            except Exception as db_err:
                ctx["error"] = f"Prediction succeeded but failed to save to database: {db_err}"

            # Feature 5: Email Notification (only if checkbox ticked + email given)
            if send_email_flag:
                if ctx["email"]:
                    pdf_path = generate_pdf(record)
                    email_result = send_email_notification(record, pdf_path)
                    if email_result["sent"]:
                        ctx["email_status"] = {"ok": True,
                            "msg": f"Email sent successfully to {ctx['email']}."}
                    else:
                        ctx["email_status"] = {"ok": False,
                            "msg": f"Could not send email — {email_result['error']}"}
                else:
                    ctx["email_status"] = {"ok": False,
                        "msg": "Email checkbox was ticked but no email address was provided."}

        except (ValueError, KeyError) as e:
            ctx["error"] = str(e)

    return render_template("index.html", **ctx)


# DOWNLOAD PDF
@app.route("/download_report")
@login_required
def download_report():
    if not history:
        return "No student data found. Please predict first.", 400
    pdf_path = generate_pdf(history[-1])
    return send_file(pdf_path, as_attachment=True,
                     download_name="student_report.pdf",
                     mimetype="application/pdf")


# ADMIN DASHBOARD — Statistics & Charts (Admin/HOD only)
@app.route("/dashboard")
@admin_required
def dashboard():
    stats = calculate_dashboard_stats()
    return render_template(
        "dashboard.html",
        stats=stats,
        logged_in_name=session.get("name", ""),
        logged_in_role=session.get("role", ""),
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)