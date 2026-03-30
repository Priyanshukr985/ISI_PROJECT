from src.vectorstore.index_pipeline import IndexBuilder
from src.vectorstore.vector import PDFLoader , TextChunker , HFEmbedding , FAISSStore
from src.llm_model import LLM_Loader
from src.retrieval_grader import RetrievalGrader
from src.rag_generator import Rag_Generator
from src.question_rewriter import QuestionRewriter
from src.graph_node import RetrieverNode, GraderNode, GeneratorNode, QueryTransformNode, DecisionNode
from src.graph_builder import WorkflowBuilder
from src.video_search import YouTubeVideoSearch
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, redirect, url_for, session, flash
from functools import wraps
import base64
import io
import json
import math
import os
import re
import statistics
import textwrap
from urllib.parse import urlencode
from werkzeug.security import generate_password_hash, check_password_hash

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")
USERS_FILE = os.path.join(BASE_DIR, "data", "users.json")


def _ensure_users_file():
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", encoding="utf-8") as file:
            json.dump({"users": []}, file, indent=2)


def load_users():
    _ensure_users_file()
    with open(USERS_FILE, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("users", [])


def save_users(users):
    _ensure_users_file()
    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump({"users": users}, file, indent=2)


def find_user_by_email(email):
    normalized = (email or "").strip().lower()
    return next((user for user in load_users() if user.get("email", "").lower() == normalized), None)


def normalize_next_url(next_url):
    value = (next_url or "").strip()
    if value.startswith("/") and not value.startswith("//"):
        return value
    return url_for("app_home")


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if session.get("user_email"):
            return view_func(*args, **kwargs)

        if request.path.startswith("/chat") or request.path.startswith("/concept-") or request.path.startswith("/visualize") or request.path.startswith("/practice-") or request.path.startswith("/notes-"):
            return jsonify({"error": "Authentication required."}), 401

        return redirect(url_for("signin", next=request.path))

    return wrapper

# IndexBuilder("Stat_Book.pdf").build_index()

llm_loader = LLM_Loader()
llm = llm_loader.load()

embedding_loader = HFEmbedding()
embedding_model = embedding_loader.load()

faiss_store = FAISSStore(embedding_model, index_path="faiss_index")
vectordb = faiss_store.load()

retriever = faiss_store.get_retriever(vectordb)

# Instantiate functional modules
document_grader = RetrievalGrader(llm)
question_rewriter = QuestionRewriter(llm=llm)
rag_generator = Rag_Generator(llm)
video_search = YouTubeVideoSearch()

# Create Node Objects
retrieve_node = RetrieverNode(retriever)
grade_node = GraderNode(document_grader)
generate_node = GeneratorNode(rag_generator.chain)
transform_node = QueryTransformNode(question_rewriter)
decision_node = DecisionNode()

# Build LangGraph Workflow
Builder = WorkflowBuilder(retrieve_node=retrieve_node,
                          grade_node=grade_node,
                          generate_node=generate_node,
                          transform_node=transform_node,
                          decision_node=decision_node)

app_graph = Builder.build()


CONCEPT_ACTION_PROMPTS = {
    ("Probability Distributions", "Quick Summary"): (
        "Explain probability distributions from basics, including discrete vs continuous, PMF, PDF, and CDF, "
        "with simple examples. Keep it structured and easy to study."
    ),
    ("Probability Distributions", "Core Distributions"): (
        "Teach the major probability distributions: normal, binomial, Poisson, and exponential. "
        "For each one, include intuition, formula idea, when to use it, and one simple example."
    ),
    ("Probability Distributions", "Formulas and Meaning"): (
        "Explain expectation, variance, PMF, PDF, and CDF in probability distributions with formulas, "
        "plain-language meaning, and one worked example."
    ),
    ("Probability Distributions", "Practice Questions"): (
        "Generate 3 practice questions on probability distributions with answers and step-by-step explanations."
    ),
    ("Probability Distributions", "Start Analysis"): (
        "Analyze probability distributions in depth. Cover discrete vs continuous distributions, PMF, PDF, CDF, "
        "expectation, variance, and major models like normal, binomial, Poisson, and exponential. "
        "Format the answer with short sections and examples."
    ),
    ("Probability Distributions", "Compare Models"): (
        "Compare binomial, Poisson, normal, and exponential distributions in a simple study-friendly table. "
        "Explain use-cases, assumptions, and how to decide between them."
    ),
}


YOUTUBE_QUERIES = {
    ("Probability Distributions", "Quick Summary"): "probability distributions basics statistics",
    ("Probability Distributions", "Core Distributions"): "normal binomial poisson exponential probability distributions statistics",
    ("Probability Distributions", "Formulas and Meaning"): "expectation variance pmf pdf cdf probability distributions",
    ("Probability Distributions", "Practice Questions"): "probability distributions practice problems statistics",
    ("Probability Distributions", "Start Analysis"): "probability distributions full lecture statistics",
    ("Probability Distributions", "Compare Models"): "binomial poisson normal exponential comparison statistics",
}

PRACTICE_EXAMS = {
    "Descriptive Statistics and Probability": {
        "subtitle": "Practice questions on descriptive statistics, measures of dispersion, probability basics, and Bayes theorem.",
        "video_query": "descriptive statistics probability practice questions statistics",
        "starter": "Give me one practice question on descriptive statistics or probability with a short hint.",
    },
    "Univariate Distributions": {
        "subtitle": "Practice questions on random variables, cdf, pmf, pdf, moments, and standard univariate distributions.",
        "video_query": "univariate distributions practice questions statistics",
        "starter": "Give me one univariate distributions question and wait for my attempt.",
    },
    "Multivariate Distributions": {
        "subtitle": "Practice questions on joint, marginal, and conditional distributions of random vectors.",
        "video_query": "multivariate distributions practice questions statistics",
        "starter": "Give me one multivariate distributions practice question with a hint.",
    },
    "Limit Theorems": {
        "subtitle": "Practice questions on convergence, laws of large numbers, and the central limit theorem.",
        "video_query": "limit theorems practice questions statistics",
        "starter": "Give me one question on limit theorems and then provide a hint.",
    },
    "Sampling Distributions": {
        "subtitle": "Practice questions on order statistics, sampling distributions, and chi-square, t, and F distributions.",
        "video_query": "sampling distributions practice questions statistics",
        "starter": "Give me one sampling distributions question with full formulas after I try.",
    },
    "Estimation": {
        "subtitle": "Practice questions on sufficiency, UMVUE, Cramer-Rao, method of moments, MLE, and confidence intervals.",
        "video_query": "estimation practice questions mathematical statistics",
        "starter": "Give me one estimation question and wait for my attempt.",
    },
    "Testing of Hypotheses": {
        "subtitle": "Practice questions on critical regions, size, power, Neyman-Pearson, and likelihood ratio tests.",
        "video_query": "testing of hypotheses practice questions statistics",
        "starter": "Ask me one hypothesis testing practice question and then give a hint.",
    },
    "Nonparametric Methods": {
        "subtitle": "Practice questions on runs test, Kolmogorov-Smirnov, sign tests, and Mann-Whitney test.",
        "video_query": "nonparametric methods practice questions statistics",
        "starter": "Give me one nonparametric methods question with a short hint.",
    },
    "Stochastic Processes": {
        "subtitle": "Practice questions on Markov chains, transition probabilities, Chapman-Kolmogorov, and Poisson process.",
        "video_query": "stochastic processes practice questions statistics",
        "starter": "Give me one stochastic processes question and wait for my attempt.",
    },
}

def llm_text_response(prompt):
    result = llm.invoke(prompt)
    return getattr(result, "content", result)


def build_concept_response(concept, action):
    query = YOUTUBE_QUERIES.get((concept, action), f"{concept} statistics")
    videos = video_search.search(query)

    return {
        "concept": concept,
        "action": action,
        "videos": videos,
    }


def build_practice_response(exam_name):
    exam = PRACTICE_EXAMS.get(exam_name)
    if not exam:
        return None

    videos = video_search.search(
        exam["video_query"],
        query_prefix="Dr Harish Garg",
        channel_filter="harish garg",
    )
    return {
        "exam": exam_name,
        "subtitle": exam["subtitle"],
        "starter": exam["starter"],
        "videos": videos,
    }
def build_practice_prompt(exam_name, user_message):
    return (
        f"You are a statistics practice mentor for {exam_name}. "
        "Help the student practice concept-wise statistics questions. "
        "If the user asks for a question, give one practice question first, then wait for them to attempt unless they ask for the solution. "
        "If the user asks for a solution, solve step by step with formulas and shortcuts where useful. "
        "Render mathematical symbols and formulas using plain MathJax-ready LaTeX delimiters, such as $x$, $$P(X=k)$$, and $$\\mu, \\sigma^2$$. "
        "Do not escape the dollar signs with backslashes. "
        "Keep the explanation exam-focused, concise, and study-friendly.\n\n"
        f"Student request: {user_message}"
    )


def build_notes_prompt(section_name, subtopic):
    return (
        f"Create high-quality IIT JAM Statistics study notes for the section '{section_name}' and subtopic '{subtopic}'. "
        "Write in polished markdown study-note style with these exact headings: ## Overview, ## Why It Matters, ## Core Ideas, ## Key Formulas, ## Worked Insight, ## Common Mistakes, ## Exam Tips, ## One-Minute Revision. "
        "Use bullet points where helpful. Keep it concise but complete, around 700 to 1000 words. "
        "Use plain MathJax-ready LaTeX delimiters for formulas such as $x$, $$P(X=k)$$ and $$\\mu, \\sigma^2$$. "
        "Do not escape dollar signs with backslashes. Avoid markdown tables."
    )


def build_notes_chat_prompt(section_name, subtopic, user_message):
    return (
        f"You are a statistics notes assistant for IIT JAM. Section: {section_name}. Subtopic: {subtopic}. "
        "Answer the student's follow-up using step-by-step explanation when needed. "
        "Render mathematical symbols and formulas using plain MathJax-ready LaTeX delimiters, such as $x$, $$P(X=k)$$, and $$\\mu, \\sigma^2$$. "
        "Do not escape the dollar signs with backslashes.\n\n"
        f"Student request: {user_message}"
    )


def build_notes_video_query(section_name, subtopic):
    return f"IIT JAM statistics {section_name} {subtopic} lecture notes"


def _clean_note_text(text):
    return (
        str(text or "")
        .replace("### ", "")
        .replace("## ", "")
        .replace("# ", "")
        .replace("**", "")
        .replace("__", "")
        .replace("\r", "")
    )


def _pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(title, subtitle, body):
    lines = []
    lines.extend(textwrap.wrap(title, width=42))
    lines.append("")
    lines.extend(textwrap.wrap(subtitle, width=70))
    lines.append("")
    for raw_line in body.splitlines():
        clean_line = raw_line.strip()
        if not clean_line:
            lines.append("")
            continue
        wrapped = textwrap.wrap(clean_line, width=92) or [""]
        lines.extend(wrapped)

    commands = []
    y = 770
    for index, line in enumerate(lines):
        if y < 50:
            break
        if index == 0:
            font_size = 20
        elif index <= 2:
            font_size = 11
        elif line.endswith(":"):
            font_size = 13
        else:
            font_size = 10
        commands.append(f"BT /F1 {font_size} Tf 50 {y} Td ({_pdf_escape(line)}) Tj ET")
        y -= 18 if font_size >= 13 else 14

    stream = "\n".join(commands).encode("latin-1", errors="replace")
    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1")
        + stream
        + b"\nendstream endobj\n"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin-1")
    )
    return bytes(pdf)


def _plot_to_base64():
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_number_list(raw_value, fallback):
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return fallback
    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values or fallback


def generate_visualization(chart_type, payload):
    if sns:
        sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor="#0f172a")
    ax.set_facecolor("#172554")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
    ax.tick_params(colors="#e2e8f0")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#ffffff")

    explanation = ""
    stats = {}
    insights = []

    if chart_type == "Normal Curve":
        mean = float(payload.get("mean", 0))
        std = max(float(payload.get("std", 1)), 0.1)
        x_min = mean - 4 * std
        x_max = mean + 4 * std
        x_vals = [x_min + i * (x_max - x_min) / 250 for i in range(251)]
        coeff = 1 / (std * math.sqrt(2 * math.pi))
        y_vals = [coeff * math.exp(-0.5 * ((x - mean) / std) ** 2) for x in x_vals]
        ax.plot(x_vals, y_vals, color="#f8fafc", linewidth=3)
        ax.fill_between(x_vals, y_vals, color="#60a5fa", alpha=0.3)
        ax.set_title("Normal Distribution Curve")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        explanation = (
            f"This bell curve is centered at mean {mean:.2f} with standard deviation {std:.2f}. "
            f"Larger standard deviation makes the curve wider and flatter, while smaller standard deviation makes it tighter around the mean."
        )
        stats = {"Mean": f"{mean:.2f}", "Standard Deviation": f"{std:.2f}", "Symmetry": "Perfectly symmetric"}
        insights = [
            "Peak occurs exactly at the mean.",
            "Curve shape stays symmetric around the center.",
            "Changing standard deviation mainly changes spread and height.",
        ]

    elif chart_type == "Binomial Plot":
        n = max(int(float(payload.get("n", 10))), 1)
        p = min(max(float(payload.get("p", 0.5)), 0.01), 0.99)
        xs = list(range(n + 1))
        ys = [math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x)) for x in xs]
        ax.bar(xs, ys, color="#38bdf8", edgecolor="#f8fafc", width=0.7)
        ax.set_title("Binomial Distribution PMF")
        ax.set_xlabel("Number of successes")
        ax.set_ylabel("Probability")
        explanation = (
            f"This binomial PMF uses n = {n} trials and success probability p = {p:.2f}. "
            f"The tallest bars show the most likely success counts across repeated independent trials."
        )
        stats = {"Trials": str(n), "Probability": f"{p:.2f}", "Most Likely x": str(xs[ys.index(max(ys))])}
        insights = [
            "The tallest bar marks the most probable number of successes.",
            "Distribution shape depends on both trial count and success probability.",
            "As p moves away from 0.5, the plot becomes more skewed.",
        ]

    elif chart_type == "Histogram":
        values = _parse_number_list(payload.get("values"), [2, 4, 4, 5, 5, 5, 6, 7, 8, 9, 10, 10, 11, 12])
        bins = max(int(float(payload.get("bins", 6))), 2)
        if sns:
            sns.histplot(values, bins=bins, kde=True, color="#38bdf8", ax=ax)
        else:
            ax.hist(values, bins=bins, color="#38bdf8", edgecolor="#f8fafc", alpha=0.85)
        ax.set_title("Histogram")
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")
        explanation = (
            f"This histogram summarizes {len(values)} observations using {bins} bins. "
            f"Use it to inspect shape, spread, skewness, and whether values cluster in one or more regions."
        )
        stats = {
            "Count": str(len(values)),
            "Min": f"{min(values):.2f}",
            "Max": f"{max(values):.2f}",
            "Bins": str(bins),
        }
        insights = [
            "Look for peaks to identify common value ranges.",
            "Tail length helps reveal skewness.",
            "Gaps may suggest clusters or separate subgroups.",
        ]

    elif chart_type == "Box Plot":
        values = _parse_number_list(payload.get("values"), [4, 5, 5, 6, 6, 7, 8, 9, 10, 12, 15])
        ax.boxplot(
            values,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#38bdf8", alpha=0.55, edgecolor="#f8fafc"),
            medianprops=dict(color="#f97316", linewidth=2.5),
            whiskerprops=dict(color="#f8fafc"),
            capprops=dict(color="#f8fafc"),
            flierprops=dict(markerfacecolor="#facc15", markeredgecolor="#facc15", markersize=7),
        )
        ax.set_title("Box Plot")
        ax.set_xlabel("Values")
        ax.set_yticks([])
        explanation = (
            f"This box plot summarizes {len(values)} values using the median, quartiles, whiskers, and outliers. "
            f"Use it to compare spread, center, and unusual observations quickly."
        )
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        stats = {
            "Median": f"{statistics.median(values):.2f}",
            "Q1": f"{quartiles[0]:.2f}",
            "Q3": f"{quartiles[2]:.2f}",
            "IQR": f"{(quartiles[2] - quartiles[0]):.2f}",
        }
        insights = [
            "Median line shows the center of the dataset.",
            "Box width captures the middle 50 percent of values.",
            "Points beyond whiskers may indicate outliers.",
        ]

    elif chart_type == "Scatter Plot":
        x_vals = _parse_number_list(payload.get("x_values"), [1, 2, 3, 4, 5, 6])
        y_vals = _parse_number_list(payload.get("y_values"), [2, 3, 5, 4, 6, 7])
        size = min(len(x_vals), len(y_vals))
        x_vals = x_vals[:size]
        y_vals = y_vals[:size]
        ax.scatter(x_vals, y_vals, s=80, color="#38bdf8", edgecolors="#f8fafc", linewidth=0.8)
        if size >= 2:
            slope, intercept = statistics.linear_regression(x_vals, y_vals)
            fit_y = [slope * x + intercept for x in x_vals]
            ax.plot(x_vals, fit_y, color="#f97316", linewidth=2.5)
            explanation = (
                f"This scatter plot contains {size} paired observations. "
                f"The fitted line suggests an average trend with slope {slope:.2f}, which helps interpret the direction of association."
            )
            stats = {"Pairs": str(size), "Slope": f"{slope:.2f}", "Intercept": f"{intercept:.2f}"}
            insights = [
                "Point clustering around the line suggests stronger association.",
                "The slope sign shows whether the trend is positive or negative.",
                "Outliers can noticeably affect the fitted line.",
            ]
        else:
            explanation = "A scatter plot needs at least two paired observations to reveal a relationship trend clearly."
            stats = {"Pairs": str(size)}
            insights = ["At least two paired points are needed to see a relationship trend."]
        ax.set_title("Scatter Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    elif chart_type == "Bar Chart":
        labels = [item.strip() for item in (payload.get("labels", "A,B,C,D")).split(",") if item.strip()]
        values = _parse_number_list(payload.get("values"), [12, 19, 7, 14])
        size = min(len(labels), len(values))
        labels = labels[:size]
        values = values[:size]
        ax.bar(labels, values, color="#38bdf8", edgecolor="#f8fafc", alpha=0.9)
        ax.set_title("Bar Chart")
        ax.set_xlabel("Category")
        ax.set_ylabel("Value")
        explanation = (
            "This bar chart compares values across categories. Use it when categories are discrete and you want fast visual comparison of their magnitudes."
        )
        max_index = values.index(max(values))
        stats = {"Categories": str(size), "Highest": labels[max_index], "Max Value": f"{values[max_index]:.2f}"}
        insights = [
            "Bar height directly encodes category magnitude.",
            "Large gaps between bars show clear differences.",
            "Category ordering can make comparisons easier to scan.",
        ]

    elif chart_type == "Pie Chart":
        labels = [item.strip() for item in (payload.get("labels", "A,B,C,D")).split(",") if item.strip()]
        values = _parse_number_list(payload.get("values"), [30, 20, 25, 25])
        size = min(len(labels), len(values))
        labels = labels[:size]
        values = values[:size]
        colors = ["#38bdf8", "#f97316", "#22c55e", "#facc15", "#a78bfa", "#fb7185"][:size]
        ax.clear()
        ax.set_facecolor("#172554")
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"color": "#f8fafc"},
        )
        ax.set_title("Pie Chart", color="#ffffff")
        explanation = (
            "This pie chart shows how the total is divided across categories. Use it for simple part-to-whole comparisons with a small number of slices."
        )
        total = sum(values)
        max_index = values.index(max(values))
        stats = {"Categories": str(size), "Largest Slice": labels[max_index], "Largest Share": f"{(values[max_index] / total) * 100:.1f}%"}
        insights = [
            "Each slice represents a proportion of the full circle.",
            "Pie charts work best with a small number of categories.",
            "For precise comparison, bar charts are usually easier to read.",
        ]

    elif chart_type == "Pareto Distribution":
        alpha = max(float(payload.get("alpha", 2.5)), 0.2)
        xm = max(float(payload.get("xm", 1.0)), 0.1)
        x_max = xm * 8
        x_vals = [xm + i * (x_max - xm) / 250 for i in range(251)]
        y_vals = [(alpha * (xm ** alpha)) / (x ** (alpha + 1)) for x in x_vals]
        ax.plot(x_vals, y_vals, color="#f8fafc", linewidth=3)
        ax.fill_between(x_vals, y_vals, color="#60a5fa", alpha=0.28)
        ax.set_title("Pareto Distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        explanation = (
            f"This Pareto curve uses shape alpha = {alpha:.2f} and minimum scale xm = {xm:.2f}. "
            f"Smaller alpha means a heavier tail, so extreme large values become more influential."
        )
        stats = {"Alpha": f"{alpha:.2f}", "xm": f"{xm:.2f}", "Tail": "Heavy right tail"}
        insights = [
            "The curve starts high near xm and falls slowly for heavy-tail settings.",
            "Lower alpha values make extreme outcomes relatively more important.",
            "This family is useful for concentration effects and 80/20 style patterns.",
        ]

    else:
        raise ValueError("Unsupported visualization type")

    image_b64 = _plot_to_base64()
    return {"image": image_b64, "explanation": explanation, "stats": stats, "insights": insights}


# Run Graph
# state = {"question": "Central limit theorem"}
# result = app_graph.invoke(state)

# print("\n=== FINAL RESULT ===")
# print(result["generation"])


@app.route("/")
def home():
    if session.get("user_email"):
        return redirect(url_for("app_home"))
    return render_template("landing.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if session.get("user_email"):
        return redirect(url_for("app_home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        next_url = request.form.get("next", "").strip()

        user = find_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("auth.html", auth_mode="signin", next_url=next_url)

        session["user_email"] = user["email"]
        session["user_name"] = user["name"]
        target = normalize_next_url(next_url)
        return redirect(target)

    return render_template("auth.html", auth_mode="signin", next_url=request.args.get("next", ""))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_email"):
        return redirect(url_for("app_home"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(name) < 2:
            flash("Name must be at least 2 characters.", "error")
            return render_template("auth.html", auth_mode="signup")
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            flash("Enter a valid email address.", "error")
            return render_template("auth.html", auth_mode="signup")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("auth.html", auth_mode="signup")
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("auth.html", auth_mode="signup")
        if find_user_by_email(email):
            flash("An account with this email already exists.", "error")
            return render_template("auth.html", auth_mode="signup")

        users = load_users()
        users.append(
            {
                "name": name,
                "email": email,
                "password_hash": generate_password_hash(password),
            }
        )
        save_users(users)

        session["user_email"] = email
        session["user_name"] = name
        return redirect(url_for("app_home"))

    return render_template("auth.html", auth_mode="signup")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/app")
@login_required
def app_home():
    return render_template("index.html", current_user=session.get("user_name"))

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    """
    Receives JSON: { "message": "text from user" }
    Returns: { "reply": "AI doctor response" }
    """
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({"reply": "Please type a message."})

        # LLM RESPONSE
        state = {"question": user_message}
        ai_reply = app_graph.invoke(state)['generation']

        return jsonify({"reply": ai_reply})

    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": "Server error occurred. Try again later."})


@app.route("/concept-analysis", methods=["POST"])
@login_required
def concept_analysis():
    try:
        data = request.get_json() or {}
        concept = data.get("concept", "").strip()
        action = data.get("action", "").strip()
        mode = data.get("mode", "").strip()

        if not concept or not action:
            return jsonify({"error": "Concept and action are required."}), 400

        response = build_concept_response(concept, action)
        if mode != "videos_only":
            prompt = CONCEPT_ACTION_PROMPTS.get((concept, action))
            if not prompt:
                prompt = f"Explain {concept} from basics with intuition, key ideas, formulas, and one example."
            response["content"] = llm_text_response(prompt)
        response["video_source"] = "youtube_api" if video_search.is_configured() else "disabled"
        return jsonify(response)

    except Exception as e:
        print("Concept analysis error:", e)
        return jsonify({"error": "Concept analysis failed. Try again later."}), 500


@app.route("/visualize", methods=["POST"])
@login_required
def visualize():
    try:
        data = request.get_json() or {}
        chart_type = data.get("chart_type", "").strip()
        if not chart_type:
            return jsonify({"error": "chart_type is required"}), 400
        result = generate_visualization(chart_type, data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print("Visualization error:", e)
        return jsonify({"error": "Visualization failed. Try again later."}), 500


@app.route("/practice-analysis", methods=["POST"])
@login_required
def practice_analysis():
    try:
        data = request.get_json() or {}
        exam_name = data.get("exam", "").strip()
        if not exam_name:
            return jsonify({"error": "exam is required"}), 400

        response = build_practice_response(exam_name)
        if not response:
            return jsonify({"error": "Unsupported exam"}), 400

        response["video_source"] = "youtube_api" if video_search.is_configured() else "disabled"
        return jsonify(response)
    except Exception as e:
        print("Practice analysis error:", e)
        return jsonify({"error": "Practice analysis failed. Try again later."}), 500


@app.route("/practice-chat", methods=["POST"])
@login_required
def practice_chat():
    try:
        data = request.get_json() or {}
        exam_name = data.get("exam", "").strip()
        user_message = data.get("message", "").strip()

        if not exam_name or not user_message:
            return jsonify({"error": "exam and message are required"}), 400
        if exam_name not in PRACTICE_EXAMS:
            return jsonify({"error": "Unsupported exam"}), 400

        reply = llm_text_response(build_practice_prompt(exam_name, user_message))
        return jsonify({"reply": reply})
    except Exception as e:
        print("Practice chat error:", e)
        return jsonify({"error": "Practice chat failed. Try again later."}), 500


@app.route("/notes-resource", methods=["POST"])
@login_required
def notes_resource():
    try:
        data = request.get_json() or {}
        section_name = data.get("section", "").strip()
        subtopic = data.get("subtopic", "").strip()
        if not section_name or not subtopic:
            return jsonify({"error": "section and subtopic are required"}), 400

        videos = video_search.search(
            build_notes_video_query(section_name, subtopic),
            query_prefix="",
            channel_filter=None,
            fallback_channel_filter=False,
        )
        params = urlencode({"section": section_name, "subtopic": subtopic})
        filename = f"{re.sub(r'[^A-Za-z0-9]+', '_', subtopic).strip('_') or 'notes'}.pdf"
        return jsonify(
            {
                "section": section_name,
                "subtopic": subtopic,
                "videos": videos,
                "download_url": f"/notes-pdf?{params}",
                "filename": filename,
            }
        )
    except Exception as e:
        print("Notes resource error:", e)
        return jsonify({"error": "Notes resource could not be loaded."}), 500


@app.route("/notes-content", methods=["POST"])
@login_required
def notes_content():
    try:
        data = request.get_json() or {}
        section_name = data.get("section", "").strip()
        subtopic = data.get("subtopic", "").strip()
        if not section_name or not subtopic:
            return jsonify({"error": "section and subtopic are required"}), 400

        content = llm_text_response(build_notes_prompt(section_name, subtopic))
        return jsonify({"content": content})
    except Exception as e:
        print("Notes content error:", e)
        return jsonify({"error": "Notes content could not be generated."}), 500


@app.route("/notes-pdf", methods=["GET"])
@login_required
def notes_pdf():
    try:
        section_name = request.args.get("section", "").strip()
        subtopic = request.args.get("subtopic", "").strip()
        if not section_name or not subtopic:
            return jsonify({"error": "section and subtopic are required"}), 400

        note_text = llm_text_response(build_notes_prompt(section_name, subtopic))
        note_text = _clean_note_text(note_text)
        pdf_bytes = _build_simple_pdf(
            subtopic,
            f"IIT JAM Statistics Notes • {section_name}",
            note_text,
        )
        filename = f"{re.sub(r'[^A-Za-z0-9]+', '_', subtopic).strip('_') or 'notes'}.pdf"
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        print("Notes PDF error:", e)
        return jsonify({"error": "Notes PDF could not be generated."}), 500


@app.route("/notes-chat", methods=["POST"])
@login_required
def notes_chat():
    try:
        data = request.get_json() or {}
        section_name = data.get("section", "").strip()
        subtopic = data.get("subtopic", "").strip()
        user_message = data.get("message", "").strip()
        if not section_name or not subtopic or not user_message:
            return jsonify({"error": "section, subtopic, and message are required"}), 400

        reply = llm_text_response(build_notes_chat_prompt(section_name, subtopic, user_message))
        return jsonify({"reply": reply})
    except Exception as e:
        print("Notes chat error:", e)
        return jsonify({"error": "Notes chat failed."}), 500


#  RUN SERVER 
if __name__ == "__main__":
    app.run(debug=True)
