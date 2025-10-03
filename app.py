# app.py

# ===============================================================
# ARIA™ — 3-tab MVP: Demographics | Questionnaire | Recommendations
# Prosody/Cadenza fixed per user • Rule-based Moda • Gemini fallback
# ===============================================================

import os, sys, re, json, ast, time, random, logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# --- Torch debug (helps ensure correct interpreter/versions) ---
# try:
#     import torch
#     st.sidebar.write("Python:", sys.executable)
#     st.sidebar.write("Torch:", torch.__version__)
#     st.sidebar.write("Torch path:", getattr(torch, "__file__", ""))
#     st.sidebar.write("Has torch.uint64:", hasattr(torch, "uint64"))
# except Exception as e:
#     st.sidebar.warning(f"Torch not available: {e}")

# --- FAISS (optional; falls back to NumPy if unavailable) -----
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.getLogger("tornado.access").setLevel(logging.ERROR)
SENT_RE = re.compile(r"[^.!?]+[.!?]")

# -----------------------------
# Config defaults
# -----------------------------
DEFAULT_RECIPES_CSV = "dataset/Concoctions _Dataset_with_Sensory_Storytelling.csv"
TITLE_COL_DESIRED   = "Name"
ING_COL_DESIRED     = "ingredients"
EMB_MODEL_ID        = "all-MiniLM-L6-v2"
TOP_N_RECS_DEFAULT  = 3

# >>> DATASET: load from project path (no uploader)
RECIPES_PATH = DEFAULT_RECIPES_CSV  # <-- set to your CSV path in the project folder

# Archetype seed text (embedding targets)
ARCHETYPE_TEXT = {
    "Alchemist":"Experimental curious process driven ferments reductions rare techniques",
    "Curator":"Tasteful discerning minimalist design conscious elegant selected iconic brands",
    "Forager":"Wild seasonal herbal raw rustic gathered nature bound earthy",
    "Mystic":"Introspective poetic lunar symbolic dreamy sacred aesthetics",
    "Provocateur":"Bold playful rule breaking vibrant colors unusual pairings maximalist",
    "Archivist":"Historic methodical ancient recipes classical menus detail oriented",
    "Oracle":"Symbolic analytical visionary knowledge systems correspondences codices",
    "Artisan":"Grounded manual craftsmanship handmade sensorial detail loving tools",
    "Host":"Hospitable warm social entertaining shared meals joyful aperitivo",
    "Wanderer":"Curious open ended culturally fluid travel regional journeys exploration",
    "Sentinel":"Loyal rooted legacy minded protects tradition heritage techniques",
}
ARCHES = list(ARCHETYPE_TEXT.keys())

# Ingredient → Moda rules
MODA_RULES = {
    "Nocturne": ["custard","burnt","honey","lavender","rose","chamomile","earl grey","smoked"],
    "Virtuoso": ["mille-feuille","layer","ganache","truffle","pistachio","macaron","soufflé"],
    "Elixir":   ["broth","stew","dashi","miso","pho","bone","infusion","ginseng","turmeric","ginger"],
    "Tableau":  ["citrus","lemon","lime","yuzu","orange","berry","pomegranate","basil","mint","matcha","salad","fritter","crudités","tartare"],
}

# Internal ARIA archetype → Persona rows for prosody/cadenza
ARCHETYPE_TO_PERSONA = {
    "Alchemist": "The Alchemist",
    "Curator":   "The Builder",
    "Forager":   "The Explorer",
    "Mystic":    "The Romantic",
    "Provocateur":"The Rebel",
    "Archivist": "The Guardian",
    "Oracle":    "The Oracle",
    "Artisan":   "The Builder",
    "Host":      "The Host",
    "Wanderer":  "The Explorer",
    "Sentinel":  "The Guardian",
}

# Prosody/Cadenza table
PROSODY_CADENZA = {
    "The Alchemist": {"primary_prosody":"Rubato","secondary_prosody":"Cantabile","primary_cadenza":"Crescendo","secondary_cadenza":"Sfumato","feel":"Improvisational, poetic build toward mystic revelation."},
    "The Strategist":{"primary_prosody":"Staccato","secondary_prosody":"Maestoso","primary_cadenza":"Moderato","secondary_cadenza":"Maestoso","feel":"Precise, commanding delivery with formal pacing and clarity."},
    "The Nurturer":  {"primary_prosody":"Adagio","secondary_prosody":"Murmure","primary_cadenza":"Lento","secondary_cadenza":"Fermata","feel":"Gentle, close-in language with spacious emotional pacing."},
    "The Explorer":  {"primary_prosody":"Scherzando","secondary_prosody":"Vivace","primary_cadenza":"Crescendo","secondary_cadenza":"Vivace","feel":"Playful and spontaneous, building to energetic discovery."},
    "The Oracle":    {"primary_prosody":"Cantabile","secondary_prosody":"Lento","primary_cadenza":"Fermata","secondary_cadenza":"Adagio","feel":"Lyrical, contemplative speech that suspends time and deepens insight."},
    "The Rebel":     {"primary_prosody":"Staccato","secondary_prosody":"Scherzando","primary_cadenza":"Presto","secondary_cadenza":"Crescendo","feel":"Punchy, subversive momentum with sharp emotional spikes."},
    "The Romantic":  {"primary_prosody":"Cantabile","secondary_prosody":"Adagio","primary_cadenza":"Adagio","secondary_cadenza":"Sfumato","feel":"Tender, emotionally nuanced prose with soft expressive tempo."},
    "The Guardian":  {"primary_prosody":"Lento","secondary_prosody":"Maestoso","primary_cadenza":"Lento","secondary_cadenza":"Maestoso","feel":"Measured, dependable language with dignified pace and calm authority."},
    "The Builder":   {"primary_prosody":"Moderato","secondary_prosody":"Staccato","primary_cadenza":"Moderato","secondary_cadenza":"Lento","feel":"Steady, clear communication with a constructive and intentional rhythm."},
    "The Visionary": {"primary_prosody":"Rubato","secondary_prosody":"Vivace","primary_cadenza":"Crescendo","secondary_cadenza":"Presto","feel":"Dynamic, future-facing expression that swells toward catalytic insight."},
    "The Host":      {"primary_prosody":"Allegretto","secondary_prosody":"Cantabile","primary_cadenza":"Vivace","secondary_cadenza":"Moderato","feel":"Light, engaging rhythm that invites intimacy and connection."},
}

PROSODY_STYLE_GUIDE = {
    "Rubato":"Flexible tempo; vary sentence length; allow subtle asymmetry and surprise turns.",
    "Cantabile":"Lyrical, legato phrasing; flowing images; melodic cadences.",
    "Staccato":"Short, punchy clauses; crisp verbs; minimal connectors.",
    "Maestoso":"Stately, formal diction; dignified pacing; elevated cadence.",
    "Adagio":"Slow, elongated phrasing; warm tone; gentle transitions.",
    "Murmure":"Whisper-soft, intimate, fragment-friendly phrasing.",
    "Scherzando":"Playful, sprightly; light wordplay; bright motion.",
    "Vivace":"Lively, energetic rhythm; quick momentum; vivid images.",
    "Lento":"Measured, deliberate; grounded imagery; calm breath.",
    "Fermata":"Linger on a resonant final image; suspended time; soft close.",
    "Moderato":"Balanced phrasing; medium length; clear structure.",
    "Presto":"Very brisk; economy of words; kinetic close.",
    "Allegretto":"Light, easy tempo; friendly warmth; inviting cadence.",
    "Sfumato":"Soft edges; gentle ambiguity; blur the boundary of images."
}
CADENZA_STYLE_GUIDE = {
    "Crescendo":"Let sentence two rise in intensity—bigger image, higher energy.",
    "Sfumato":"Let sentence two soften and blur, ending with a gentle ambiguity.",
    "Moderato":"Keep the close even and balanced; neither swell nor hush.",
    "Maestoso":"Conclude with dignified weight and clarity.",
    "Lento":"End with calm, grounded stillness.",
    "Fermata":"Hold on a resonant final image—linger.",
    "Presto":"Snap to an agile, brisk finish.",
    "Vivace":"Finish with sparkle and lift.",
    "Adagio":"Close with tender, unhurried warmth."
}

# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="ARIA™ MVP — 3-tab Demo", layout="wide")
st.title("ARIA™ — Psychometric Recommender (MVP)")
#st.caption("1) Demographics  •  2) Questionnaire  •  3) Recommendations")

# Sidebar — app controls
st.sidebar.header("App Settings")
topk = st.sidebar.slider("Top-N recommendations", 1, 10, TOP_N_RECS_DEFAULT, 1)

#prosody_mode = st.sidebar.selectbox("Prosody/Cadenza selection", ["primary","secondary","hybrid"], index=0)
use_llm = st.sidebar.checkbox("Use Gemini for narratives", value=True)
seed_val = 42 #st.sidebar.number_input("Random seed (optional)", value=0, step=1)

# =============================
# DROP-IN: Gemini key + model
# =============================
DEFAULT_GEMINI_KEY = "AIzaSyCfxYh18uteX3PfzqdFCaaARIRHY8QJlsE"  # <-- put your real key here for the MVP


def get_gemini_key() -> str | None:
    return DEFAULT_GEMINI_KEY

@st.cache_resource(show_spinner=False)
def init_gemini(api_key: str | None, model_name: str = "gemini-2.0-flash"):
    if not api_key:
        return None
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel(model_name)
    except Exception:
        return genai.GenerativeModel("gemini-1.5-flash")

def init_gemini_noarg():
    return init_gemini(get_gemini_key())

# Initialize model immediately after sidebar
model = init_gemini_noarg() if use_llm else None
if use_llm and model is None:
    st.warning("Gemini model could not be initialized. Using deterministic fallback narratives.")

# =========================
# Helpers / Cache
# =========================
@st.cache_resource(show_spinner=True)
def load_embedder(model_id: str = EMB_MODEL_ID):
    return SentenceTransformer(model_id)

def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True))
    return df

def parse_ing(cell: str) -> str:
    try:
        obj = json.loads(cell)
    except Exception:
        try:
            obj = ast.literal_eval(cell)
        except Exception:
            return cell
    if isinstance(obj, list):
        return ", ".join((d.get("ingredient") if isinstance(d, dict) else str(d)) for d in obj)
    return str(obj)

@st.cache_data(show_spinner=True)
def load_recipes_df_from_path(path: str) -> pd.DataFrame:
    # Try to read the project CSV; fallback to a tiny demo dataset
    try:
        df = pd.read_csv(path)
        #st.sidebar.success(f"Loaded recipes from: {os.path.abspath(path)}")
    except Exception as e:
        st.sidebar.warning(f"Could not read '{path}' ({e}). Using demo recipes.")
        data = [
            {"Name":"Espresso & Dark Chocolate Semifreddo with Marsala Zabaglione",
             "ingredients": "espresso, dark chocolate, cream, egg yolk, marsala, sugar"},
            {"Name":"Turkish Courgette Fritters with Dill Yogurt and Pomegranate",
             "ingredients": "courgette, dill, yogurt, flour, egg, pomegranate"},
            {"Name":"Chilled Soba with Truffle Dashi and Uni",
             "ingredients": "soba, dashi, truffle, uni, soy, mirin"},
            {"Name":"Thai Courgette Som Tam with Roasted Peanuts and Nam Pla Prik",
             "ingredients": "courgette, chili, lime, fish sauce, peanut, garlic"},
        ]
        df = pd.DataFrame(data)

    df = tidy(df)
    title = TITLE_COL_DESIRED.lower()
    if title not in df.columns:
        title = next((c for c in df.columns if re.search(r"(title|name)", c, re.I)), df.columns[0])
    ing = ING_COL_DESIRED.lower()
    if ing not in df.columns:
        ing = next((c for c in df.columns if "ingredient" in c.lower()), df.columns[-1])
    df["ing_text"] = df[ing].astype(str).apply(parse_ing)
    df["gemini_context"] = df[title].astype(str) + ". " + df["ing_text"].str[:120]
    df["_title_col"] = title
    return df

@st.cache_resource(show_spinner=True)
def build_item_index(recipes_df: pd.DataFrame, emb_model_name: str):
    emb = load_embedder(emb_model_name)
    item_vecs = emb.encode(recipes_df["gemini_context"].tolist(), show_progress_bar=False).astype("float32")
    arch_vecs = emb.encode([ARCHETYPE_TEXT[k] for k in ARCHES], show_progress_bar=False).astype("float32")
    if HAVE_FAISS:
        index = faiss.IndexHNSWFlat(item_vecs.shape[1], 32)
        index.add(item_vecs)
    else:
        index = None
    return index, item_vecs, arch_vecs, emb

def infer_moda(ingredients: str) -> str:
    ing = ingredients.lower()
    for moda, kws in MODA_RULES.items():
        if any(kw in ing for kw in kws):
            return moda
    return random.choice(list(MODA_RULES.keys()))

def prosody_for_archetype(archetype: str, confidence: float | None = None,
                          mode: str = "primary", secondary_prob: float = 0.25,
                          rng: random.Random | None = None):
    persona = ARCHETYPE_TO_PERSONA.get(archetype, "The Builder")
    row = PROSODY_CADENZA.get(persona, PROSODY_CADENZA["The Builder"])
    if mode == "primary":
        choose_secondary = False
    elif mode == "secondary":
        choose_secondary = True
    else:  # "hybrid"
        p_secondary = secondary_prob
        if confidence is not None:
            p_secondary = max(0.05, secondary_prob * (1.0 - float(confidence)))
        r = (rng.random() if rng else random.random())
        choose_secondary = (r < p_secondary)
    prosody  = row["secondary_prosody"] if choose_secondary else row["primary_prosody"]
    cadenza  = row["secondary_cadenza"] if choose_secondary else row["primary_cadenza"]
    feel     = row["feel"]
    return prosody, cadenza, feel, persona

# Gemini helpers
def gemini_text(model, prompt: str, tries: int = 3, base_sleep: float = 1.2) -> str | None:
    if model is None:
        return None
    for attempt in range(tries):
        try:
            resp = model.generate_content(prompt)
            pf = getattr(resp, "prompt_feedback", None)
            if pf and getattr(pf, "block_reason", None):
                st.warning(f"Gemini blocked: {pf.block_reason}")
                return None
            txt = getattr(resp, "text", None)
            if txt and txt.strip():
                return txt.strip()
            cands = getattr(resp, "candidates", None) or []
            if cands:
                content = getattr(cands[0], "content", None)
                parts = getattr(content, "parts", None) or []
                joined = " ".join(getattr(p, "text", "") for p in parts if getattr(p, "text", ""))
                if joined.strip():
                    return joined.strip()
            time.sleep(base_sleep * (2 ** attempt))
        except Exception:
            time.sleep(base_sleep * (2 ** attempt))
    return None

def generate_storyline(model, title, ing, tradition, moda, prosody, cadenza, feel, archetype, use_llm=True) -> str:
    prosody_guide = PROSODY_STYLE_GUIDE.get(prosody, "Balanced phrasing; clear structure.")
    cadenza_guide = CADENZA_STYLE_GUIDE.get(cadenza, "Finish with balance.")
    parts = [p.strip() for p in ing.split(",") if p.strip()]
    first_ing  = parts[0] if parts else ing.strip()
    second_ing = parts[1] if len(parts) > 1 else ""

    prompt = f"""
You are a poetic culinary narrator, trained in the Ricettario tradition.
Write a sensory story in EXACTLY TWO sentences (<= 50 words total). Avoid instructions; do not explain—evoke.

Context:
• Title: {title}
• Ingredients: {ing}
• Culinary Tradition: {tradition}
• Moda (emotional color): {moda}
• Prosody (style): {prosody} → {prosody_guide}
• Cadenza (closing gesture): {cadenza} → {cadenza_guide}
• Narrative Feel: {feel}
• Archetype Essence: {archetype}

Requirements:
• Use the prosody style in rhythm and phrasing.
• Let the cadenza shape the second sentence’s finish.
• Weave in {first_ing}{(" and " + second_ing) if second_ing else ""} as sensory anchors.
• Keep it elegant, metaphor-rich, and human.
Two sentences only.
""".strip()

    if use_llm:
        txt = gemini_text(model, prompt, tries=3, base_sleep=1.2)
        if txt:
            sents = SENT_RE.findall(txt)
            return " ".join(s.strip() for s in sents[:2]) if sents else txt

    # deterministic fallback
    if prosody in ("Staccato","Presto"):
        return (f"{title}: {first_ing} strikes quick. "
                f"It lands, bright and gone, a {moda.lower()} flicker.")
    if prosody in ("Cantabile","Adagio","Murmure","Sfumato"):
        return (f"In {title}, {first_ing} drifts softly. "
                f"The close lingers in {moda.lower()} hush.")
    if prosody in ("Rubato","Moderato","Maestoso","Allegretto"):
        return (f"{title} opens with {first_ing} in poised measure. "
                f"The finish gathers {moda.lower()} weight.")
    return (f"{title} moves with {prosody.lower()} grace. "
            f"It finishes in {cadenza.lower()} cadence.")

import re

TITLE_HEADING_TMPL = (
    r"^\s*(?:#+\s*)?"     # optional “### ” style heading
    r"[*_~]*\s*"          # optional ** / __ / ~~ emphasis marks
    r"{title}"            # the title itself (escaped later)
    r"\s*[*_~]*\s*$"      # optional emphasis marks / trailing space
)

def _strip_repeated_title(title: str, story: str, min_overlap: float = 0.6) -> str:
    """Remove any first-line heading that just repeats the recipe title."""
    heading_re = re.compile(
        TITLE_HEADING_TMPL.format(title=re.escape(title)),
        flags=re.I,
    )

    lines = [ln for ln in story.splitlines() if ln.strip()]   # keep non-blank
    if lines and heading_re.match(lines[0]):
        story = "\n".join(lines[1:]).lstrip()

    # ── optional word-overlap fallback (keeps previous behaviour) ──
    first_clause = re.split(r"[.!?]", story, maxsplit=1)[0]
    title_words  = set(re.sub(r"[^\w]", " ", title.lower()).split())
    clause_words = set(re.sub(r"[^\w]", " ", first_clause.lower()).split())
    overlap = len(title_words & clause_words) / (len(title_words | clause_words) or 1)

    if overlap >= min_overlap:
        story = story[len(first_clause):].lstrip(" .,!?:–—-*")

    return story


# =========================
# Load recipes + embeddings
# =========================
if seed_val:
    random.seed(int(seed_val)); np.random.seed(int(seed_val))
recipes_df = load_recipes_df_from_path(RECIPES_PATH)
index, item_vecs, archetype_vecs, emb_model = build_item_index(recipes_df, EMB_MODEL_ID)



import difflib

# ------------------------------------------------------------------
# helper: drop leading sentence if it mostly repeats the title
# ------------------------------------------------------------------


# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["1) User Demographics", "2) Questionnaire", "3) Recommendations"])

# ---------- TAB 1: Demographics ----------
# ---------- TAB 1: Demographics (3 partitions: Checks | Sliders | Other) ----------
# ---------- TAB 1: Demographics ----------
with tab1:
    # Get existing demographics from session state for defaults
    existing_demo = st.session_state.get("demographics", {})

    with st.form(key="demographics_form"):
        col_other, col_checks, col_sliders = st.columns([1.0, 1.0, 1.0])

        # ------------------ OTHER INPUTS PARTITION ------------------
        with col_other:
            st.markdown("### Core Profile & Context")
            o1, o2 = st.columns(2)
            name = o1.text_input("Name", value=existing_demo.get("Name", "Steven Burton"))
            age = o2.number_input("Age", min_value=16, max_value=99, value=existing_demo.get("Age", 19))

            genders = ["Male","Female","Non-binary","Prefer not to say"]
            gender_idx = genders.index(existing_demo.get("Gender", "Male")) if existing_demo.get("Gender") in genders else 0
            gender = st.selectbox("Gender", genders, index=gender_idx)
            location = st.text_input("Location", value=existing_demo.get("Location", "Dallas"))

            st.markdown("---")
            edu_opts = ["High School","Associate","Bachelor's","Master's","PhD"]
            edu_idx = edu_opts.index(existing_demo.get("Highly_Educated", "Bachelor's")) if existing_demo.get("Highly_Educated") in edu_opts else 2
            highly_educated = st.selectbox("Highest Education", edu_opts, index=edu_idx)
            
            diet_opts = ["None","Vegan","Vegetarian","Pescatarian","Keto","Paleo","Halal","Kosher"]
            diet_idx = diet_opts.index(existing_demo.get("Dietary_Preferences", "Vegan")) if existing_demo.get("Dietary_Preferences") in diet_opts else 1
            dietary_pref = st.selectbox("Dietary Preferences", diet_opts, index=diet_idx)
            
            channel_opts = ["Online","In-store","Omnichannel"]
            channel_idx = channel_opts.index(existing_demo.get("Preferred_Shopping_Channel", "Online")) if existing_demo.get("Preferred_Shopping_Channel") in channel_opts else 0
            pref_channel = st.selectbox("Preferred Shopping Channel", channel_opts, index=channel_idx)

            hobbies = st.text_input("Hobbies & Interests", value=existing_demo.get("Hobbies_Interests", "Fitness"))
            affluent = st.number_input("Affluent (annual income)", value=existing_demo.get("Affluent", 141924), step=1000)

        # ------------------ CHECKBOX PARTITION ------------------
        with col_checks:
            st.markdown("### Life Stage & Affiliation")
            cb1, cb2 = st.columns(2)

            millennial_women = cb1.checkbox("Millennial Women", value=bool(existing_demo.get("Millennial_Women", False)))
            modern_mom = cb2.checkbox("Modern Mom", value=bool(existing_demo.get("Modern_Mom", False)))
            gen_z = cb1.checkbox("Gen Z", value=bool(existing_demo.get("Gen_Z", True)))
            entrepreneur = cb2.checkbox("Entrepreneur", value=bool(existing_demo.get("Entrepreneur", True)))
            intrepreneur = cb1.checkbox("Intrepreneur", value=bool(existing_demo.get("Intrepreneur", False)))
            early_adopter = cb2.checkbox("Early Adopter", value=bool(existing_demo.get("Early_Adopter", True)))
            well_traveled = cb1.checkbox("Well Traveled", value=bool(existing_demo.get("Well_Traveled", True)))
            social_online = cb2.checkbox("Socially Active Online", value=bool(existing_demo.get("Socially_Active_Online", True)))
            foodie = cb1.checkbox("Self-Proclaimed Foodie", value=bool(existing_demo.get("Self_Proclaimed_Foodie", True)))
            urban_families = cb2.checkbox("Urban Families", value=bool(existing_demo.get("Urban_Families", True)))
            dinks = cb1.checkbox("DINKS", value=bool(existing_demo.get("DINKS", False)))
            empty_nesters = cb2.checkbox("Empty Nesters", value=bool(existing_demo.get("Empty_Nesters", False)))
            emerging_adults = cb1.checkbox("Emerging Adults", value=bool(existing_demo.get("Emerging_Adults", True)))

        # ------------------ SLIDER PARTITION ------------------
        with col_sliders:
            st.markdown("### Preference & Behavior Scales")
            globally_intuitive = st.slider("Globally Intuitive (0–10)", 0, 10, existing_demo.get("Globally_Intuitive", 10))
            moving_doing = st.slider("Moving & Doing Constantly (0–10)", 0, 10, existing_demo.get("Moving_Doing_Constantly", 8))
            social_offline = st.slider("Socially Active Offline (0–10)", 0, 10, existing_demo.get("Socially_Active_Offline", 9))
            experience_driven = st.slider("Experience Driven (0–10)", 0, 10, existing_demo.get("Experience_Driven", 1))
            quality_conscious = st.slider("Quality Conscious (0–10)", 0, 10, existing_demo.get("Quality_Conscious", 3))
            tech_savvy = st.slider("Tech Savvy (0–10)", 0, 10, existing_demo.get("Tech_Savvy", 6))
            shopping_freq = st.slider("Shopping Frequency (0–10)", 0, 10, existing_demo.get("Shopping_Frequency", 10))
            brand_loyalty = st.slider("Brand Loyalty (0.0–1.0)", 0.0, 1.0, float(existing_demo.get("Brand_Loyalty", 0.65)))
            env_conscious = st.slider("Environmental Consciousness (0–10)", 0, 10, existing_demo.get("Environmental_Consciousness", 0))
            social_resp = st.slider("Social Responsibility Awareness (0–10)", 0, 10, existing_demo.get("Social_Responsibility_Awareness", 8))
            innovation_seeking = st.slider("Innovation & Novelty Seeking (0–10)", 0, 10, existing_demo.get("Innovation_Novelty_Seeking", 8))
            personalization = st.slider("Personalization Preference (0–10)", 0, 10, existing_demo.get("Personalization_Preference", 10))

        # Form submit button
        submitted = st.form_submit_button("Save Demographics", type="primary", use_container_width=True)
    
    # Only update session state when form is submitted
    if submitted:
        st.session_state["demographics"] = {
            "Name": name, "Age": age, "Gender": gender, "Location": location,
            "Millennial_Women": int(millennial_women),
            "Modern_Mom": int(modern_mom),
            "Gen_Z": int(gen_z),
            "Entrepreneur": int(entrepreneur),
            "Intrepreneur": int(intrepreneur),
            "Early_Adopter": int(early_adopter),
            "Well_Traveled": int(well_traveled),
            "Globally_Intuitive": globally_intuitive,
            "Highly_Educated": highly_educated,
            "Moving_Doing_Constantly": moving_doing,
            "Socially_Active_Online": int(social_online),
            "Socially_Active_Offline": social_offline,
            "Self_Proclaimed_Foodie": int(foodie),
            "Dietary_Preferences": dietary_pref,
            "Experience_Driven": experience_driven,
            "Quality_Conscious": quality_conscious,
            "Tech_Savvy": tech_savvy,
            "Hobbies_Interests": hobbies,
            "Affluent": affluent,
            "Urban_Families": int(urban_families),
            "DINKS": int(dinks),
            "Empty_Nesters": int(empty_nesters),
            "Emerging_Adults": int(emerging_adults),
            "Shopping_Frequency": shopping_freq,
            "Preferred_Shopping_Channel": pref_channel,
            "Brand_Loyalty": brand_loyalty,
            "Environmental_Consciousness": env_conscious,
            "Social_Responsibility_Awareness": social_resp,
            "Innovation_Novelty_Seeking": innovation_seeking,
            "Personalization_Preference": personalization,
        }
        st.success("Demographics saved successfully!")
    
    # Show current saved data
    if "demographics" in st.session_state:
        with st.expander("View Saved Demographics"):
            st.json(st.session_state["demographics"])
# ---------- TAB 2: Questionnaire ----------
# ---------- TAB 2: Questionnaire (33 questions, full text + options) ----------
with tab2:
    #st.subheader("Questionnaire")

    QUESTIONS = [
        {
            "q": "When selecting an everyday item, what primarily guides your choice?",
            "opts": [
                "Functionality and practicality",
                "Aesthetic appeal and design",
                "Brand reputation or recommendations",
                "Ethical and environmental considerations",
            ],
        },
        {
            "q": "Which of these indulgences do you find most rewarding?",
            "opts": [
                "Discovering a new hobby or skill",
                "Treating myself to a luxury item",
                "Enjoying a gourmet meal or delicacy",
                "Spending time in nature or traveling",
            ],
        },
        {
            "q": "Your ideal weekend is most likely to include:",
            "opts": [
                "Engaging in a creative or physical activity",
                "Exploring new trends or technologies",
                "Relaxing at home with a book or film",
                "Socializing with friends or attending an event",
            ],
        },
        {
            "q": "When making choices, what factor do you prioritize?",
            "opts": [
                "Sustainability and ethical production",
                "Quality and durability",
                "Cost-effectiveness and budget",
                "Cultural or artistic significance",
            ],
        },
        {
            "q": "How do you typically respond to emerging trends?",
            "opts": [
                "I’m usually one of the first to explore them",
                "I wait to see if they fit my style and needs",
                "I rarely follow trends, preferring timeless choices",
                "I blend trendiness with classic elements",
            ],
        },
        {
            "q": "What kind of online content are you most drawn to?",
            "opts": [
                "Educational or skill-building resources",
                "Lifestyle and wellness tips",
                "News and current events",
                "Entertainment and humor",
            ],
        },
        {
            "q": "Looking ahead, what area of your life are you most focused on enhancing?",
            "opts": [
                "Personal or professional development",
                "Health and wellbeing",
                "Relationships and social connections",
                "Leisure and travel experiences",
            ],
        },
        {
            "q": "What usually influences your decisions the most?",
            "opts": [
                "Personal research and knowledge",
                "Recommendations from friends or family",
                "Professional or expert opinions",
                "Intuition or spontaneous feeling",
            ],
        },
        {
            "q": "When investing in yourself, what do you prioritize?",
            "opts": [
                "Learning new skills or education",
                "Physical health and fitness",
                "Personal grooming and presentation",
                "Mental and emotional well-being",
            ],
        },
        {
            "q": "Which of these community engagements is most appealing to you?",
            "opts": [
                "Volunteering for a cause or charity",
                "Participating in local events or clubs",
                "Joining online forums or social media groups",
                "Attending workshops or talks",
            ],
        },
        {
            "q": "Your ideal leisure activity would likely involve:",
            "opts": [
                "Arts and culture (e.g., museums, galleries, theater)",
                "Nature and the outdoors (e.g., hiking, gardening)",
                "Culinary experiences (e.g., cooking, food tasting)",
                "Technology and innovation (e.g., gadgets, gaming)",
            ],
        },
        {
            "q": "How do you typically adapt to significant changes in life or routine?",
            "opts": [
                "I embrace change enthusiastically",
                "I adapt gradually with some planning",
                "I prefer stability and minimal changes",
                "I find new ways to blend old and new routines",
            ],
        },
        {
            "q": "When thinking about the future, what excites you the most?",
            "opts": [
                "Technological advancements",
                "Social and cultural shifts",
                "Personal or family milestones",
                "Career or educational opportunities",
            ],
        },
        {
            "q": "Looking back, which type of past experiences do you value the most?",
            "opts": [
                "Adventures and travels",
                "Career or academic achievements",
                "Personal relationships and connections",
                "Moments of personal growth and realization",
            ],
        },
        {
            "q": "When considering new products or ideas, how do you balance innovation with tradition?",
            "opts": [
                "I generally favor cutting-edge innovations.",
                "I prefer tried and true methods, but I'm open to new ideas.",
                "I mostly stick with traditional options.",
                "I blend both equally in my choices.",
            ],
        },
        {
            "q": "Which of these accomplishments would bring you the most satisfaction?",
            "opts": [
                "Achieving a personal fitness or health goal",
                "Completing a creative project or hobby",
                "Reaching a significant career milestone",
                "Fostering meaningful relationships",
            ],
        },
        {
            "q": "How do you prefer to explore and cultivate new interests?",
            "opts": [
                "Through online research and virtual experiences",
                "By attending classes or workshops",
                "Through travel and real-world experiences",
                "By connecting with knowledgeable individuals",
            ],
        },
        {
            "q": "Which type of sensory experience do you find most appealing?",
            "opts": [
                "Visual (e.g., art, photography, scenic views)",
                "Auditory (e.g., music, nature sounds, spoken word)",
                "Tactile (e.g., textures, hands-on activities)",
                "Gustatory/Olfactory (e.g., tasting, cooking, fragrances)",
            ],
        },
        {
            "q": "What is your preferred way to unwind and relax?",
            "opts": [
                "Engaging in a quiet, solitary activity",
                "Spending time with friends or family",
                "Being active, such as sports or outdoor activities",
                "Consuming entertainment like movies or books",
            ],
        },
        {
            "q": "How does technology play a role in your daily life?",
            "opts": [
                "It’s central to most of what I do.",
                "It’s important, but I maintain a balance.",
                "I use it minimally and prefer traditional methods.",
                "I use it mainly for communication and basic needs.",
            ],
        },
        {
            "q": "What kind of impact or legacy are you most motivated to leave?",
            "opts": [
                "A professional or career-oriented impact",
                "A personal or familial legacy",
                "A creative or artistic contribution",
                "An impact on community or social causes",
            ],
        },
        {
            "q": "Which perspective or world view resonates most with you?",
            "opts": [
                "Optimism and constantly seeing potential",
                "Realism and pragmatism in approach",
                "Idealism and striving for a better world",
                "Traditionalism and preserving established ways",
            ],
        },
        {
            "q": "When seeking inspiration, which of these are you most likely to turn to?",
            "opts": [
                "A beautifully crafted object or scene",
                "An emotionally moving piece of storytelling",
                "A harmonious and captivating sound or melody",
                "A thought-provoking abstract concept",
            ],
        },
        {
            "q": "In your downtime, what type of experience do you find most enriching?",
            "opts": [
                "Exploring new and unfamiliar settings or environments",
                "Engaging with a compelling narrative or plot",
                "Being immersed in rhythmic and melodic creations",
                "Observing or creating visual aesthetics",
            ],
        },
        {
            "q": "What aspect of a new culture do you find most intriguing to explore?",
            "opts": [
                "Traditional and contemporary artistic expressions",
                "The storytelling and folklore",
                "The distinctive sounds and music",
                "The architectural and design heritage",
            ],
        },
        {
            "q": "Which of these experiences resonates with you on a deeper emotional level?",
            "opts": [
                "Witnessing a breathtaking visual display",
                "Getting lost in a gripping story or performance",
                "Feeling moved by a powerful melody or rhythm",
                "Discovering an ingenious design or structure",
            ],
        },
        {
            "q": "At a social gathering, which activity are you most likely to enjoy?",
            "opts": [
                "Viewing or discussing an art exhibit",
                "Watching a film or play",
                "Listening to or playing music",
                "Debating ideas in design and architecture",
            ],
        },
        {
            "q": "Which of these creative interests do you most identify with?",
            "opts": [
                "Visual arts and photography",
                "Writing or theatrical arts",
                "Music and sound design",
                "Architectural and spatial design",
            ],
        },
        {
            "q": "Which historical aspect fascinates you the most?",
            "opts": [
                "Artistic movements and styles",
                "Legendary narratives and epics",
                "Evolution of music genres",
                "Architectural eras and design philosophies",
            ],
        },
        {
            "q": "What influences your personal style or aesthetics the most?",
            "opts": [
                "Visual art trends",
                "Cinematic or theatrical genres",
                "Musical styles or icons",
                "Design and architectural innovations",
            ],
        },
        {
            "q": "If you were to express yourself creatively, which form would it take?",
            "opts": [
                "A visual art piece",
                "A written or performed piece",
                "A musical composition",
                "A design or structural concept",
            ],
        },
        {
            "q": "How do you prefer to relax and unwind?",
            "opts": [
                "Viewing art or design pieces",
                "Watching a story unfold on screen or stage",
                "Listening to or creating music",
                "Exploring architectural spaces or designs",
            ],
        },
        {
            "q": "What role do artistic elements play in your life?",
            "opts": [
                "A source of inspiration and creativity",
                "A means of emotional expression and release",
                "A way to connect with others and the world",
                "An avenue for intellectual stimulation and thought",
            ],
        },
    ]

    # Render
    # 1️ — create / repair the defaults dict so every Q has an index
    q_defaults = st.session_state.get("q_defaults", {})
    if len(q_defaults) != len(QUESTIONS):         # new run or QUESTIONS length changed
        q_defaults = {}                           # start fresh
    for i, item in enumerate(QUESTIONS, start=1):
        q_key = str(i)
        if q_key not in q_defaults:               # add any missing key
            q_defaults[q_key] = random.randrange(len(item["opts"]))
    st.session_state["q_defaults"] = q_defaults   # persist back

    # 2️ — render the survey
    answers = {}
    for i, item in enumerate(QUESTIONS, start=1):
        q_key = str(i)
        st.markdown(f"**Q{i}. {item['q']}**")

        cols = st.columns(2)
        selected = []
        for j, opt in enumerate(item["opts"]):
            col = cols[j % 2]
            default_checked = (j == q_defaults[q_key])
            checked = col.checkbox(
                opt,
                key=f"Q{i}_opt{j}",
                value=default_checked,
            )
            if checked:
                selected.append(opt)

        # 3️ — ensure at least one option stays checked
        if not selected:
            default_j = q_defaults[q_key]
            st.session_state[f"Q{i}_opt{default_j}"] = True
            selected.append(item["opts"][default_j])

        answers[f"Q{i}"] = selected
        st.divider()

    # 4️ — save all answers
    st.session_state["questionnaire"] = answers
    st.success("Questionnaire saved with current selections.")
# ---------- TAB 3: Recommendations ----------
with tab3:
    st.subheader("Build Profile → Assign Archetype → Generate Recommendations")

    demo = st.session_state.get("demographics", {})
    qans = st.session_state.get("questionnaire", {})

    def build_joined_text(demo: Dict[str, Any], qans: Dict[str, Any]) -> str:
        parts = []
        # Demographics
        for k, v in demo.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if vv: parts.append(f"{kk} {vv}")
            else:
                parts.append(f"{k} {v}")
        # Questionnaire
        for k, v in qans.items():
            parts.append(f"{k} {v}")
        return " ".join(map(str, parts))

    colX, colY = st.columns([2,1])
    with colX:
        st.markdown("**Profile text (embedded to assign archetype):**")
        joined = build_joined_text(demo, qans)
        st.code(joined[:1200] + ("..." if len(joined) > 1200 else ""), language="text")
    with colY:
        st.write("")
        st.write("")
        compute_btn = st.button("Compute Archetype & Style", type="primary")

    if compute_btn:
        # Embed profile text vs archetypes to assign primary archetype
        mem_vec = load_embedder(EMB_MODEL_ID).encode([joined], show_progress_bar=False)
        sims = cosine_similarity(mem_vec, build_item_index(recipes_df, EMB_MODEL_ID)[2])[0]  # archetype_vecs
        best_idx = int(np.argmax(sims))
        primary_arch = ARCHES[best_idx]
        confidence = float(sims[best_idx])

        # Fix Prosody/Cadenza for this user (per sidebar mode)
        prosody, cadenza, feel, persona = prosody_for_archetype(primary_arch, confidence, mode='primary')

        st.session_state["profile_result"] = {
            "primary_archetype": primary_arch,
            "confidence": confidence,
            "prosody": prosody,
            "cadenza": cadenza,
            "feel": feel,
            "persona": persona
        }

    prof = st.session_state.get("profile_result")
    if prof:
        c1, c3, c4 = st.columns(3)
        c1.metric("Archetype", prof["primary_archetype"])
        #c2.metric("Confidence", f"{prof['confidence']:.3f}")
        c3.metric("Prosody", prof["prosody"])
        c4.metric("Cadenza", prof["cadenza"])
        with st.expander("Narrative Style Details"):
            st.markdown(f"**Persona:** {prof['persona']}")
            st.markdown(f"**Narrative feel:** {prof['feel']}")
            st.markdown(f"**Prosody guide:** {PROSODY_STYLE_GUIDE.get(prof['prosody'],'—')}")
            st.markdown(f"**Cadenza guide:** {CADENZA_STYLE_GUIDE.get(prof['cadenza'],'—')}")

        st.divider()
        st.subheader("Recommendations")

        def recommend_topk(arch: str, k: int = 3, pool: int = 200) -> pd.DataFrame:
            arch_idx = ARCHES.index(arch)
            qvec = build_item_index(recipes_df, EMB_MODEL_ID)[2][arch_idx].astype("float32").reshape(1,-1)  # archetype_vecs
            if HAVE_FAISS:
                D, I = build_item_index(recipes_df, EMB_MODEL_ID)[0].search(qvec, min(pool, len(recipes_df)))
                pool_df = recipes_df.iloc[I[0]].copy()
                d = D[0]
                d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
                pool_df["score"] = 1.0 - d_norm
                return pool_df.nlargest(k, "score")
            else:
                # cosine similarity fallback
                sims = cosine_similarity(qvec, build_item_index(recipes_df, EMB_MODEL_ID)[1])[0]  # item_vecs
                top_idx = np.argsort(-sims)[:pool]
                pool_df = recipes_df.iloc[top_idx].copy()
                pool_df["score"] = sims[top_idx]
                return pool_df.nlargest(k, "score")
        

        if st.button("Generate Recommendations", type="primary"):
            recs = recommend_topk(prof["primary_archetype"], k=topk, pool=200)
            title_col = recs["_title_col"].iloc[0]
            for _, r in recs.iterrows():
                title = r[title_col]
                moda  = infer_moda(r["ing_text"])
                story = generate_storyline(
                    model     = model,
                    title     = title,
                    ing       = r["ing_text"],
                    tradition = "Italian-Fusion",
                    moda      = moda,
                    prosody   = prof["prosody"],
                    cadenza   = prof["cadenza"],
                    feel      = prof["feel"],
                    archetype = prof["primary_archetype"],
                    use_llm   = use_llm
                )

                story = _strip_repeated_title(title, story)  # <<< NEW
                with st.container(border=True):
                    st.markdown(f"### {title}")
                    print(story)
                    st.write(story)
                    st.caption(f"**Moda:** {moda}  |  **Prosody/Cadenza:** {prof['prosody']} · {prof['cadenza']}")
                    st.caption(f"**Narrative feel:** {prof['feel']}")                    
                    with st.expander("Ingredients"):
                        st.write(r["ing_text"])
    else:
        st.info("Fill Tabs 1 & 2, then click **Compute Archetype & Style** to proceed.")

st.divider()
st.caption("© ARIA™ — MVP demo. All rights reserved.")
