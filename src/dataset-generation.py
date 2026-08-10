"""
================================================================================
 SYNTHETIC LMS DATA GENERATOR
 Personalized Learning Path Recommendation System — Hackathon Dataset Builder
================================================================================

This script programmatically builds a professional, internally-consistent
synthetic dataset that mimics a real Learning Management System (LMS):

    1. student_profiles.csv     (~10,000 students)
    2. subject_catalog.csv      (course -> subject mapping)
    3. topic_catalog.csv        (3,000+ topics, mapped to subjects)
    4. learning_history.csv     (~50,000 learning activity records)
    5. resource_catalog.csv     (learning resources per topic)

Design principles
------------------
* Curriculum-driven: Education_Level -> Course -> Subjects -> Topics is a
  real hierarchy (defined in CURRICULUM below), not random noise.
* Referential integrity: every Student_ID, Subject and Topic referenced in
  a "child" table exists in its "parent" table. Students only ever generate
  learning_history rows for subjects they are actually registered in, and
  only for topics that belong to that subject.
* Reproducibility: a single global SEED drives both NumPy and Faker (and a
  local fallback name generator if Faker isn't installed), so re-running
  this script produces byte-identical output.
* Realistic distributions: ages, scores, attempts, progress and time-spent
  are drawn from bounded/skewed distributions rather than uniform noise,
  and engagement is intentionally skewed (a minority of students generate
  most of the activity), mirroring real LMS usage patterns.

Run:
    python generate_lms_data.py

Output:
    ./lms_dataset/student_profiles.csv
    ./lms_dataset/subject_catalog.csv
    ./lms_dataset/topic_catalog.csv
    ./lms_dataset/learning_history.csv
    ./lms_dataset/resource_catalog.csv
================================================================================
"""

import os
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 0. REPRODUCIBILITY
# --------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Faker is optional but recommended ("pip install faker") for richer names.
# We fall back to a deterministic local name generator if it isn't present,
# so the script always runs end-to-end.
try:
    from faker import Faker

    fake = Faker()
    Faker.seed(SEED)
    USE_FAKER = True
except ImportError:
    USE_FAKER = False

_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Rohan", "Kabir", "Aryan", "Dev", "Yash", "Kunal",
    "Ananya", "Diya", "Saanvi", "Aadhya", "Myra", "Ira", "Anika", "Navya",
    "Kiara", "Riya", "Sara", "Pari", "Meera", "Tara", "Priya", "Neha",
    "James", "Liam", "Noah", "Oliver", "Ethan", "Lucas", "Mason", "Logan",
    "Emma", "Olivia", "Ava", "Sophia", "Isabella", "Mia", "Chloe", "Grace",
    "Wei", "Jun", "Hiro", "Yuki", "Min-jun", "Sofia", "Lucia", "Mateo",
    "Fatima", "Ahmed", "Omar", "Zainab", "Layla", "Yusuf", "Hassan", "Amir",
]
_LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Iyer", "Nair", "Reddy", "Rao", "Mehta",
    "Kapoor", "Malhotra", "Chatterjee", "Bose", "Patel", "Shah", "Joshi",
    "Kulkarni", "Pillai", "Menon", "Singh", "Yadav", "Das", "Mishra",
    "Smith", "Johnson", "Brown", "Williams", "Jones", "Garcia", "Martinez",
    "Kim", "Chen", "Wang", "Tanaka", "Suzuki", "Khan", "Ali", "Hussain",
]


def generate_name():
    """Return a plausible full name, using Faker if available."""
    if USE_FAKER:
        return fake.name()
    return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"


def slugify(text):
    keep = string.ascii_letters + string.digits + " "
    cleaned = "".join(c for c in text if c in keep)
    return cleaned.strip().lower().replace(" ", "-")


OUTPUT_DIR = "lms_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================================
# 1. CURRICULUM DEFINITION
#    This is the single source of truth for Education_Level -> Course ->
#    Subjects. Everything downstream (subject_catalog, topic_catalog,
#    student registrations, learning_history) derives from this structure,
#    which is what guarantees foreign-key / logical consistency.
# ==========================================================================

# term_type: "grade" (school year, no semester concept),
#            "semester" (numbered semesters), "year" (numbered years, used
#            for programs like MBBS/LLB/MD that are conventionally tracked
#            by year rather than semester)
CURRICULUM = [
    # ---------------------------- SCHOOL (Grade 6-12) --------------------
    {"education_level": "Grade 6-12", "course": "Grade 6", "stream": None,
     "term_type": "grade", "num_terms": 1, "age_range": (11, 12),
     "subjects": ["Mathematics", "Science", "English", "Social Science", "Hindi"]},
    {"education_level": "Grade 6-12", "course": "Grade 7", "stream": None,
     "term_type": "grade", "num_terms": 1, "age_range": (12, 13),
     "subjects": ["Mathematics", "Science", "English", "Social Science", "Hindi"]},
    {"education_level": "Grade 6-12", "course": "Grade 8", "stream": None,
     "term_type": "grade", "num_terms": 1, "age_range": (13, 14),
     "subjects": ["Mathematics", "Science", "English", "Social Science", "Hindi"]},
    {"education_level": "Grade 6-12", "course": "Grade 9", "stream": None,
     "term_type": "grade", "num_terms": 1, "age_range": (14, 15),
     "subjects": ["Mathematics", "Science", "English", "Social Science", "Computer Basics"]},
    {"education_level": "Grade 6-12", "course": "Grade 10", "stream": None,
     "term_type": "grade", "num_terms": 1, "age_range": (15, 16),
     "subjects": ["Mathematics", "Science", "English", "Social Science", "Computer Basics"]},
    {"education_level": "Grade 6-12", "course": "Grade 11", "stream": "Science",
     "term_type": "grade", "num_terms": 1, "age_range": (16, 17),
     "subjects": ["Physics", "Chemistry", "Mathematics", "Biology", "English"]},
    {"education_level": "Grade 6-12", "course": "Grade 11", "stream": "Commerce",
     "term_type": "grade", "num_terms": 1, "age_range": (16, 17),
     "subjects": ["Accountancy", "Business Studies", "Economics", "Mathematics", "English"]},
    {"education_level": "Grade 6-12", "course": "Grade 11", "stream": "Arts",
     "term_type": "grade", "num_terms": 1, "age_range": (16, 17),
     "subjects": ["History", "Political Science", "Geography", "Sociology", "English"]},
    {"education_level": "Grade 6-12", "course": "Grade 12", "stream": "Science",
     "term_type": "grade", "num_terms": 1, "age_range": (17, 18),
     "subjects": ["Physics", "Chemistry", "Mathematics", "Biology", "English"]},
    {"education_level": "Grade 6-12", "course": "Grade 12", "stream": "Commerce",
     "term_type": "grade", "num_terms": 1, "age_range": (17, 18),
     "subjects": ["Accountancy", "Business Studies", "Economics", "Mathematics", "English"]},
    {"education_level": "Grade 6-12", "course": "Grade 12", "stream": "Arts",
     "term_type": "grade", "num_terms": 1, "age_range": (17, 18),
     "subjects": ["History", "Political Science", "Geography", "Sociology", "English"]},

    # ------------------------------- DIPLOMA ------------------------------
    {"education_level": "Diploma", "course": "Diploma in Computer Science", "stream": "Computer Science",
     "term_type": "semester", "num_terms": 6, "age_range": (15, 19),
     "subjects": ["Programming Fundamentals", "Data Structures", "DBMS",
                  "Web Development", "Computer Networks", "Operating Systems"]},
    {"education_level": "Diploma", "course": "Diploma in Mechanical Engineering", "stream": "Mechanical",
     "term_type": "semester", "num_terms": 6, "age_range": (15, 19),
     "subjects": ["Engineering Mechanics", "Thermodynamics", "Machine Design",
                  "Manufacturing Processes", "Strength of Materials", "Fluid Mechanics"]},
    {"education_level": "Diploma", "course": "Diploma in Civil Engineering", "stream": "Civil",
     "term_type": "semester", "num_terms": 6, "age_range": (15, 19),
     "subjects": ["Surveying", "Structural Analysis", "Building Materials",
                  "Concrete Technology", "Geotechnical Engineering", "Transportation Engineering"]},
    {"education_level": "Diploma", "course": "Diploma in Electronics Engineering", "stream": "Electronics",
     "term_type": "semester", "num_terms": 6, "age_range": (15, 19),
     "subjects": ["Circuit Theory", "Digital Electronics", "Microprocessors",
                  "Communication Systems", "Electronic Devices", "Control Systems"]},

    # ---------------------------- UNDERGRADUATE ----------------------------
    {"education_level": "Undergraduate", "course": "BTech Computer Science", "stream": "Computer Science",
     "term_type": "semester", "num_terms": 8, "age_range": (18, 23),
     "subjects": ["OOP", "DBMS", "Operating Systems", "Computer Networks",
                  "Data Structures", "Algorithms", "Software Engineering", "Web Technologies"]},
    {"education_level": "Undergraduate", "course": "BTech Artificial Intelligence", "stream": "Artificial Intelligence",
     "term_type": "semester", "num_terms": 8, "age_range": (18, 23),
     "subjects": ["Machine Learning", "Deep Learning", "Statistics", "OOP", "DBMS",
                  "Natural Language Processing", "Computer Vision", "Reinforcement Learning"]},
    {"education_level": "Undergraduate", "course": "BTech Mechanical Engineering", "stream": "Mechanical",
     "term_type": "semester", "num_terms": 8, "age_range": (18, 23),
     "subjects": ["Thermodynamics", "Fluid Mechanics", "Machine Design",
                  "Manufacturing Processes", "Strength of Materials", "Heat Transfer"]},
    {"education_level": "Undergraduate", "course": "BTech Civil Engineering", "stream": "Civil",
     "term_type": "semester", "num_terms": 8, "age_range": (18, 23),
     "subjects": ["Structural Engineering", "Geotechnical Engineering", "Transportation Engineering",
                  "Surveying", "Concrete Technology", "Environmental Engineering"]},
    {"education_level": "Undergraduate", "course": "BTech Electronics", "stream": "Electronics",
     "term_type": "semester", "num_terms": 8, "age_range": (18, 23),
     "subjects": ["Digital Signal Processing", "Microprocessors", "VLSI Design",
                  "Communication Systems", "Control Systems", "Embedded Systems"]},
    {"education_level": "Undergraduate", "course": "MBBS", "stream": "Medical",
     "term_type": "year", "num_terms": 5, "age_range": (18, 24),
     "subjects": ["Anatomy", "Physiology", "Biochemistry", "Pathology",
                  "Pharmacology", "Microbiology", "Forensic Medicine"]},
    {"education_level": "Undergraduate", "course": "LLB", "stream": "Law",
     "term_type": "year", "num_terms": 3, "age_range": (20, 25),
     "subjects": ["Constitutional Law", "Criminal Law", "Contract Law",
                  "Law of Torts", "Jurisprudence", "Family Law"]},
    {"education_level": "Undergraduate", "course": "BBA", "stream": "Business",
     "term_type": "semester", "num_terms": 6, "age_range": (18, 22),
     "subjects": ["Principles of Management", "Marketing Management", "Financial Accounting",
                  "Business Economics", "Organizational Behavior", "Business Statistics"]},
    {"education_level": "Undergraduate", "course": "BCom", "stream": "Commerce",
     "term_type": "semester", "num_terms": 6, "age_range": (18, 22),
     "subjects": ["Financial Accounting", "Business Law", "Economics",
                  "Taxation", "Auditing", "Cost Accounting"]},
    {"education_level": "Undergraduate", "course": "BSc Science", "stream": "Science",
     "term_type": "semester", "num_terms": 6, "age_range": (18, 22),
     "subjects": ["Physics", "Chemistry", "Mathematics", "Botany", "Zoology"]},

    # ----------------------------- POSTGRADUATE -----------------------------
    {"education_level": "Postgraduate", "course": "MTech Computer Science", "stream": "Computer Science",
     "term_type": "semester", "num_terms": 4, "age_range": (22, 28),
     "subjects": ["Advanced Algorithms", "Distributed Systems", "Cloud Computing",
                  "Advanced DBMS", "Advanced Operating Systems"]},
    {"education_level": "Postgraduate", "course": "MTech Artificial Intelligence", "stream": "Artificial Intelligence",
     "term_type": "semester", "num_terms": 4, "age_range": (22, 28),
     "subjects": ["Deep Learning", "Reinforcement Learning", "Advanced Natural Language Processing",
                  "Advanced Computer Vision", "Advanced Machine Learning"]},
    {"education_level": "Postgraduate", "course": "MBA", "stream": "Business",
     "term_type": "semester", "num_terms": 4, "age_range": (22, 30),
     "subjects": ["Strategic Management", "Financial Management", "Marketing Management",
                  "Human Resource Management", "Operations Management"]},
    {"education_level": "Postgraduate", "course": "MD", "stream": "Medical",
     "term_type": "year", "num_terms": 3, "age_range": (23, 30),
     "subjects": ["Advanced Pathology", "Clinical Medicine", "Surgery",
                  "Radiology", "Advanced Pharmacology"]},
    {"education_level": "Postgraduate", "course": "LLM", "stream": "Law",
     "term_type": "semester", "num_terms": 4, "age_range": (23, 30),
     "subjects": ["Advanced Constitutional Law", "International Law",
                  "Corporate Law", "Human Rights Law"]},
    {"education_level": "Postgraduate", "course": "MSc", "stream": "Science",
     "term_type": "semester", "num_terms": 4, "age_range": (22, 28),
     "subjects": ["Advanced Physics", "Advanced Chemistry", "Advanced Mathematics", "Advanced Biology"]},
]

LEARNING_GOALS = [
    "Exam Preparation", "Concept Mastery", "Skill Building", "Career Growth",
    "Competitive Exam Prep", "Grade Improvement", "Research Preparation",
    "Certification Prep", "Revision & Practice", "College Admission Prep",
]

DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
RESOURCE_TYPES = ["Video", "Quiz", "Practice Problems", "Interactive Lab",
                   "Case Study", "Mini Project", "PDF Notes"]

# Templates used to expand each curated "core concept" into a full
# beginner -> advanced learning sequence for a subject.
TOPIC_TEMPLATES = [
    ("Introduction to {c}", "Beginner"),
    ("Fundamentals of {c}", "Beginner"),
    ("{c}: Core Concepts", "Intermediate"),
    ("{c}: Practical Applications", "Intermediate"),
    ("Advanced {c} Techniques", "Advanced"),
    ("{c}: Case Study & Project Work", "Advanced"),
]

# Curated, domain-relevant core concepts per subject. Any subject not listed
# here (should be none, given CURRICULUM above) falls back to a generic
# concept list built from the subject name itself.
CORE_TOPICS = {
    "Mathematics": ["Number Systems", "Algebra", "Geometry", "Trigonometry", "Probability & Statistics", "Calculus Basics"],
    "Science": ["Matter and Materials", "Motion and Force", "Light and Sound", "Living Organisms", "Natural Resources"],
    "English": ["Grammar Essentials", "Reading Comprehension", "Creative Writing", "Vocabulary Building", "Literature Analysis"],
    "Social Science": ["History Foundations", "Civics and Governance", "Geography Basics", "Economics Basics"],
    "Hindi": ["Vyakaran (Grammar)", "Gadya (Prose)", "Padya (Poetry)", "Lekhan Kaushal (Writing Skills)"],
    "Computer Basics": ["Computer Hardware", "Operating System Basics", "Internet Basics", "Office Applications"],
    "Physics": ["Mechanics", "Thermodynamics", "Electricity and Magnetism", "Optics", "Modern Physics", "Waves"],
    "Chemistry": ["Atomic Structure", "Chemical Bonding", "Organic Chemistry", "Periodic Table", "Chemical Reactions", "Electrochemistry"],
    "Biology": ["Cell Biology", "Genetics", "Human Physiology", "Plant Biology", "Ecology", "Evolution"],
    "Accountancy": ["Journal & Ledger", "Trial Balance", "Financial Statements", "Partnership Accounts", "Company Accounts"],
    "Business Studies": ["Business Environment", "Forms of Business", "Marketing Basics", "Business Finance"],
    "Economics": ["Microeconomics Basics", "Macroeconomics Basics", "Demand and Supply", "National Income", "Money and Banking"],
    "History": ["Ancient Civilizations", "Medieval History", "Modern History", "World Wars", "Independence Movements"],
    "Political Science": ["Political Theory", "Indian Constitution", "Comparative Politics", "International Relations"],
    "Geography": ["Physical Geography", "Human Geography", "Climatology", "Map Reading", "Resource Geography"],
    "Sociology": ["Social Structure", "Social Institutions", "Social Change", "Research Methods in Sociology"],
    "Programming Fundamentals": ["Variables and Data Types", "Control Structures", "Functions", "Arrays", "Basic I/O"],
    "Data Structures": ["Arrays and Lists", "Stacks and Queues", "Linked Lists", "Trees", "Graphs", "Hashing"],
    "DBMS": ["ER Modeling", "Relational Model", "SQL Queries", "Normalization", "Transactions", "Indexing"],
    "Web Development": ["HTML & CSS", "JavaScript Basics", "Responsive Design", "Backend Basics", "REST APIs"],
    "Computer Networks": ["Network Topologies", "OSI Model", "TCP/IP", "Routing", "Network Security"],
    "Operating Systems": ["Process Management", "Memory Management", "File Systems", "Scheduling Algorithms", "Deadlocks"],
    "Engineering Mechanics": ["Statics", "Dynamics", "Friction", "Trusses", "Center of Gravity"],
    "Thermodynamics": ["Laws of Thermodynamics", "Entropy", "Heat Engines", "Refrigeration Cycles", "Thermodynamic Properties"],
    "Machine Design": ["Design Principles", "Stress Analysis", "Fasteners", "Gears", "Bearings"],
    "Manufacturing Processes": ["Casting", "Welding", "Machining", "Forming Processes", "Quality Control"],
    "Strength of Materials": ["Stress and Strain", "Bending Moments", "Shear Force", "Torsion", "Deflection of Beams"],
    "Fluid Mechanics": ["Fluid Properties", "Fluid Statics", "Fluid Dynamics", "Flow Measurement", "Pumps and Turbines"],
    "Surveying": ["Chain Surveying", "Leveling", "Theodolite Surveying", "Total Station", "GPS Surveying"],
    "Structural Analysis": ["Determinate Structures", "Indeterminate Structures", "Influence Lines", "Slope Deflection Method"],
    "Building Materials": ["Cement and Concrete", "Bricks and Blocks", "Steel", "Timber", "Sustainable Materials"],
    "Concrete Technology": ["Concrete Mix Design", "Concrete Properties", "Curing Methods", "Reinforced Concrete"],
    "Geotechnical Engineering": ["Soil Classification", "Soil Mechanics", "Foundation Engineering", "Slope Stability"],
    "Transportation Engineering": ["Highway Design", "Traffic Engineering", "Pavement Design", "Railway Engineering"],
    "Circuit Theory": ["Circuit Laws", "AC Circuits", "DC Circuits", "Network Theorems", "Resonance"],
    "Digital Electronics": ["Logic Gates", "Boolean Algebra", "Combinational Circuits", "Sequential Circuits", "Counters"],
    "Microprocessors": ["8085 Architecture", "Instruction Sets", "Interfacing", "Interrupts", "Memory Organization"],
    "Communication Systems": ["Modulation Techniques", "Analog Communication", "Digital Communication", "Antennas"],
    "Electronic Devices": ["Diodes", "Transistors", "Amplifiers", "Oscillators", "Power Electronics Basics"],
    "Control Systems": ["Transfer Functions", "Block Diagrams", "Stability Analysis", "PID Controllers", "State Space Analysis"],
    "OOP": ["Classes and Objects", "Inheritance", "Polymorphism", "Encapsulation", "Abstraction", "Design Patterns"],
    "Algorithms": ["Sorting Algorithms", "Searching Algorithms", "Dynamic Programming", "Greedy Algorithms", "Graph Algorithms"],
    "Software Engineering": ["SDLC Models", "Requirements Engineering", "Software Testing", "Agile Methodology", "Software Design"],
    "Web Technologies": ["Frontend Frameworks", "Backend Frameworks", "APIs and Microservices", "Web Security"],
    "Machine Learning": ["Supervised Learning", "Unsupervised Learning", "Model Evaluation", "Feature Engineering", "Ensemble Methods"],
    "Deep Learning": ["Neural Networks", "Convolutional Networks", "Recurrent Networks", "Transformers", "Optimization Techniques"],
    "Statistics": ["Descriptive Statistics", "Probability Distributions", "Hypothesis Testing", "Regression Analysis", "Sampling Methods"],
    "Natural Language Processing": ["Text Preprocessing", "Word Embeddings", "Sequence Models", "Language Models", "Sentiment Analysis"],
    "Computer Vision": ["Image Processing", "Feature Extraction", "Object Detection", "Image Segmentation", "CNN Architectures"],
    "Reinforcement Learning": ["Markov Decision Processes", "Q-Learning", "Policy Gradients", "Value Functions", "Exploration Strategies"],
    "Heat Transfer": ["Conduction", "Convection", "Radiation", "Heat Exchangers", "Boiling and Condensation"],
    "Structural Engineering": ["Load Analysis", "Beam Design", "Column Design", "Earthquake Engineering", "Steel Structures"],
    "Environmental Engineering": ["Water Treatment", "Air Pollution Control", "Solid Waste Management", "Environmental Impact Assessment"],
    "Digital Signal Processing": ["Signal Sampling", "Fourier Transform", "Filters", "Z-Transform", "Spectral Analysis"],
    "VLSI Design": ["CMOS Design", "Layout Design", "Fabrication Process", "Timing Analysis", "Verification"],
    "Embedded Systems": ["Microcontrollers", "RTOS Basics", "Sensor Interfacing", "Embedded C Programming"],
    "Anatomy": ["Skeletal System", "Muscular System", "Nervous System", "Cardiovascular System", "Respiratory System"],
    "Physiology": ["Cell Physiology", "Cardiovascular Physiology", "Respiratory Physiology", "Renal Physiology", "Endocrine Physiology"],
    "Biochemistry": ["Carbohydrates", "Proteins and Enzymes", "Lipids", "Metabolism", "Molecular Biology Basics"],
    "Pathology": ["Cell Injury", "Inflammation", "Neoplasia", "Systemic Pathology", "Hematopathology"],
    "Pharmacology": ["Pharmacokinetics", "Pharmacodynamics", "Autonomic Drugs", "Antimicrobial Agents", "CNS Drugs"],
    "Microbiology": ["Bacteriology", "Virology", "Mycology", "Immunology Basics", "Parasitology"],
    "Forensic Medicine": ["Medico-legal Aspects", "Injuries and Wounds", "Toxicology", "Autopsy Procedures"],
    "Constitutional Law": ["Fundamental Rights", "Directive Principles", "Separation of Powers", "Constitutional Amendments"],
    "Criminal Law": ["Elements of Crime", "Offences Against Person", "Offences Against Property", "Criminal Procedure"],
    "Contract Law": ["Formation of Contracts", "Breach of Contract", "Remedies", "Special Contracts"],
    "Law of Torts": ["Negligence", "Defamation", "Strict Liability", "Vicarious Liability"],
    "Jurisprudence": ["Schools of Jurisprudence", "Legal Rights and Duties", "Sources of Law", "Theories of Justice"],
    "Family Law": ["Marriage Laws", "Divorce and Maintenance", "Succession Laws", "Adoption and Guardianship"],
    "Principles of Management": ["Planning", "Organizing", "Staffing", "Directing", "Controlling"],
    "Marketing Management": ["Marketing Mix", "Consumer Behavior", "Market Segmentation", "Branding", "Digital Marketing"],
    "Financial Accounting": ["Accounting Principles", "Financial Statements", "Depreciation", "Cash Flow Statements"],
    "Business Economics": ["Demand Analysis", "Cost Analysis", "Market Structures", "Pricing Strategies"],
    "Organizational Behavior": ["Individual Behavior", "Group Dynamics", "Leadership", "Organizational Culture"],
    "Business Statistics": ["Data Collection", "Measures of Central Tendency", "Correlation and Regression", "Index Numbers"],
    "Business Law": ["Contract Act Basics", "Sale of Goods Act", "Company Law Basics", "Negotiable Instruments"],
    "Taxation": ["Income Tax Basics", "GST Fundamentals", "Tax Planning", "Corporate Taxation"],
    "Auditing": ["Audit Principles", "Internal Control", "Vouching", "Audit Report Writing"],
    "Cost Accounting": ["Cost Concepts", "Costing Methods", "Budgetary Control", "Standard Costing"],
    "Botany": ["Plant Morphology", "Plant Physiology", "Plant Taxonomy", "Plant Ecology"],
    "Zoology": ["Animal Diversity", "Animal Physiology", "Developmental Biology", "Animal Behavior"],
    "Advanced Algorithms": ["Amortized Analysis", "NP-Completeness", "Approximation Algorithms", "Randomized Algorithms"],
    "Distributed Systems": ["Distributed Consensus", "Replication", "Fault Tolerance", "Distributed File Systems"],
    "Cloud Computing": ["Cloud Service Models", "Virtualization", "Container Orchestration", "Cloud Security"],
    "Advanced DBMS": ["Query Optimization", "Concurrency Control", "Distributed Databases", "NoSQL Systems"],
    "Advanced Operating Systems": ["Virtual Memory Management", "Distributed OS", "Real-Time Systems", "Kernel Design"],
    "Advanced Natural Language Processing": ["Large Language Models", "Attention Mechanisms", "Text Generation", "Machine Translation"],
    "Advanced Computer Vision": ["Object Tracking", "3D Vision", "Generative Vision Models", "Video Understanding"],
    "Advanced Machine Learning": ["Bayesian Learning", "Kernel Methods", "Model Interpretability", "AutoML"],
    "Strategic Management": ["Strategy Formulation", "Competitive Analysis", "Strategy Implementation", "Corporate Governance"],
    "Financial Management": ["Capital Budgeting", "Working Capital Management", "Risk and Return", "Dividend Policy"],
    "Human Resource Management": ["Recruitment and Selection", "Training and Development", "Performance Management", "Compensation"],
    "Operations Management": ["Process Design", "Inventory Management", "Quality Management", "Supply Chain Management"],
    "Advanced Pathology": ["Molecular Pathology", "Tumor Biology", "Autoimmune Pathology", "Forensic Pathology"],
    "Clinical Medicine": ["Cardiology Basics", "Respiratory Medicine", "Gastroenterology", "Endocrinology"],
    "Surgery": ["Pre-operative Care", "Surgical Techniques", "Post-operative Care", "Trauma Surgery"],
    "Radiology": ["X-Ray Interpretation", "CT Imaging", "MRI Basics", "Ultrasound Techniques"],
    "Advanced Pharmacology": ["Drug Interactions", "Clinical Pharmacology", "Pharmacogenomics", "Adverse Drug Reactions"],
    "Advanced Constitutional Law": ["Judicial Review", "Federalism", "Emergency Provisions", "Comparative Constitutional Law"],
    "International Law": ["Treaties and Conventions", "International Organizations", "Law of the Sea", "Human Rights Law Basics"],
    "Corporate Law": ["Company Formation", "Corporate Governance", "Mergers and Acquisitions", "Insolvency Law"],
    "Human Rights Law": ["Civil and Political Rights", "Economic and Social Rights", "International Human Rights Bodies"],
    "Advanced Physics": ["Quantum Mechanics", "Statistical Mechanics", "Relativity", "Nuclear Physics"],
    "Advanced Chemistry": ["Spectroscopy", "Reaction Kinetics", "Coordination Chemistry", "Polymer Chemistry"],
    "Advanced Mathematics": ["Real Analysis", "Abstract Algebra", "Complex Analysis", "Numerical Methods"],
    "Advanced Biology": ["Molecular Genetics", "Genomics", "Bioinformatics Basics", "Systems Biology"],
}


def core_topics_for(subject):
    """Return curated core concepts for a subject, generic fallback otherwise."""
    if subject in CORE_TOPICS:
        return CORE_TOPICS[subject]
    return [f"{subject} Concept {i}" for i in range(1, 6)]


# ==========================================================================
# 2. SUBJECT CATALOG
# ==========================================================================

def build_subject_catalog():
    """One row per unique (Education_Level, Course, Subject) combination."""
    rows = []
    for entry in CURRICULUM:
        for subject in entry["subjects"]:
            rows.append({
                "Education_Level": entry["education_level"],
                "Course": entry["course"],
                "Stream": entry["stream"] if entry["stream"] else "General",
                "Subject": subject,
            })
    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return df


# ==========================================================================
# 3. TOPIC CATALOG (3,000+ rows)
# ==========================================================================

def build_topic_catalog(subject_catalog):
    """
    For every unique subject, expand its curated core concepts through
    TOPIC_TEMPLATES to build a beginner->advanced topic sequence with a
    logically chained Prerequisite field.
    """
    unique_subjects = sorted(subject_catalog["Subject"].unique())
    rows = []

    resource_by_difficulty = {
        "Beginner": ["Video", "PDF Notes", "Quiz"],
        "Intermediate": ["Quiz", "Practice Problems", "Video", "Case Study"],
        "Advanced": ["Interactive Lab", "Mini Project", "Case Study", "Practice Problems"],
    }
    time_by_difficulty = {"Beginner": (15, 30), "Intermediate": (30, 60), "Advanced": (60, 120)}
    skill_by_difficulty = {"Beginner": (1, 2), "Intermediate": (2, 4), "Advanced": (4, 5)}

    for subject in unique_subjects:
        concepts = core_topics_for(subject)
        previous_topic = None  # chains prerequisites across the whole subject
        for concept in concepts:
            for template, difficulty in TOPIC_TEMPLATES:
                topic_name = template.format(c=concept)
                lo, hi = time_by_difficulty[difficulty]
                est_time = int(np.random.randint(lo, hi + 1))
                lo_s, hi_s = skill_by_difficulty[difficulty]
                skill_level = int(np.random.randint(lo_s, hi_s + 1))
                resource = np.random.choice(resource_by_difficulty[difficulty])

                rows.append({
                    "Subject": subject,
                    "Topic": topic_name,
                    "Difficulty": difficulty,
                    "Estimated_Time": est_time,        # minutes
                    "Prerequisite": previous_topic if previous_topic else "None",
                    "Skill_Level": skill_level,          # 1 (novice) - 5 (expert)
                    "Recommended_Resource": resource,
                })
                previous_topic = topic_name

    df = pd.DataFrame(rows)
    # De-duplicate in the (rare) case two subjects share an identical topic name
    df = df.drop_duplicates(subset=["Subject", "Topic"]).reset_index(drop=True)
    return df


# ==========================================================================
# 4. STUDENT PROFILES (~10,000 rows)
# ==========================================================================

def build_student_profiles(n_students=10000):
    rows = []
    # Weight courses so school-level education dominates enrollment, similar
    # to a real-world student population pyramid.
    level_weights = {"Grade 6-12": 0.45, "Diploma": 0.10, "Undergraduate": 0.35, "Postgraduate": 0.10}
    entries_by_level = {lvl: [e for e in CURRICULUM if e["education_level"] == lvl] for lvl in level_weights}

    levels = list(level_weights.keys())
    weights = list(level_weights.values())

    for i in range(1, n_students + 1):
        student_id = f"S{i:05d}"
        level = np.random.choice(levels, p=weights)
        entry = random.choice(entries_by_level[level])

        age_lo, age_hi = entry["age_range"]
        age = int(np.random.randint(age_lo, age_hi + 1))

        course_label = entry["course"]
        if entry["stream"] and "Grade" in entry["course"]:
            course_label = f'{entry["course"]} - {entry["stream"]}'

        if entry["term_type"] == "grade":
            semester_label = "Annual"
        elif entry["term_type"] == "year":
            semester_label = f"Year {np.random.randint(1, entry['num_terms'] + 1)}"
        else:
            semester_label = f"Semester {np.random.randint(1, entry['num_terms'] + 1)}"

        # A student registers for all subjects tied to their course/term
        # (mirrors a fixed-curriculum LMS enrollment), which anchors the
        # foreign-key relationship used by learning_history.
        registered_subjects = entry["subjects"]

        rows.append({
            "Student_ID": student_id,
            "Student_Name": generate_name(),
            "Age": age,
            "Education_Level": entry["education_level"],
            "Course": course_label,
            "Semester": semester_label,
            "Registered_Subjects": "; ".join(registered_subjects),
            "Learning_Goal": np.random.choice(LEARNING_GOALS),
        })

    df = pd.DataFrame(rows)
    assert df["Student_ID"].is_unique, "Duplicate Student_ID generated!"
    return df, entries_by_level


# ==========================================================================
# 5. LEARNING HISTORY (~50,000 rows)
# ==========================================================================

def build_learning_history(student_df, topic_catalog, n_records=50000):
    """
    Simulates realistic study sessions:
      * A student only studies subjects they are registered in.
      * Topics within a subject are studied in curriculum sequence
        (mirrors prerequisite-respecting recommendations).
      * Engagement is skewed: a minority of (student, subject) pairs
        generate a disproportionate share of activity, like real LMS logs.
    """
    topics_by_subject = (
        topic_catalog.groupby("Subject")["Topic"].apply(list).to_dict()
    )
    est_time_lookup = dict(zip(
        zip(topic_catalog["Subject"], topic_catalog["Topic"]),
        topic_catalog["Estimated_Time"],
    ))

    # Build the full pool of valid (student, subject) pairs.
    pairs = []
    for _, srow in student_df.iterrows():
        student_id = srow["Student_ID"]
        subjects = srow["Registered_Subjects"].split("; ")
        for subj in subjects:
            if subj in topics_by_subject:  # guard against any catalog mismatch
                pairs.append((student_id, subj))

    n_pairs = len(pairs)
    # Skewed engagement weights (power-law-like) so some pairs are studied
    # much more intensively than others.
    weights = np.random.exponential(scale=1.0, size=n_pairs)
    weights = weights / weights.sum()

    sampled_idx = np.random.choice(n_pairs, size=n_records, p=weights, replace=True)

    # Count how many sessions each pair received, to assign sequence numbers.
    from collections import defaultdict
    session_counter = defaultdict(int)
    start_dates = {}
    rows = []

    today = datetime(2025, 6, 1)

    for idx in sampled_idx:
        student_id, subject = pairs[idx]
        session_counter[idx] += 1
        seq = session_counter[idx]

        topics = topics_by_subject[subject]
        n_topics = len(topics)

        current_topic = topics[(seq - 1) % n_topics]
        previous_topic = topics[(seq - 2) % n_topics] if seq > 1 else "None (Initial Assessment)"

        attempts = int(np.clip(np.round(np.random.exponential(1.4)) + 1, 1, 6))
        quiz_score = float(np.clip(np.random.normal(78 - 3 * (attempts - 1), 14), 0, 100))
        quiz_score = round(quiz_score, 1)

        base_time = est_time_lookup.get((subject, current_topic), 30)
        time_spent = int(np.clip(base_time * np.random.uniform(0.6, 1.9), 5, 240))

        topics_completed = min(seq, n_topics)
        learning_progress = round(topics_completed / n_topics, 2)

        if quiz_score >= 75:
            status = "Completed"
        elif quiz_score >= 40:
            status = "In Progress"
        else:
            status = "Needs Revision"

        if idx not in start_dates:
            start_dates[idx] = today - timedelta(days=int(np.random.randint(30, 540)))
        study_date = start_dates[idx] + timedelta(days=int(seq * np.random.randint(2, 12)))
        if study_date > today:
            study_date = today - timedelta(days=int(np.random.randint(0, 5)))

        rows.append({
            "Student_ID": student_id,
            "Subject": subject,
            "Topic": current_topic,
            "Quiz_Score": quiz_score,
            "Topics_Completed": topics_completed,
            "Time_Spent_Per_Topic": time_spent,          # minutes
            "Attempts": attempts,
            "Learning_Progress": learning_progress,       # 0-1
            "Previous_Recommendation": previous_topic,
            "Completion_Status": status,
            "Study_Date": study_date.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(rows)
    return df


# ==========================================================================
# 6. RESOURCE CATALOG
# ==========================================================================

def build_resource_catalog(topic_catalog):
    rows = []
    resource_name_bank = {
        "Video": ["Explainer Video", "Concept Walkthrough", "Lecture Recording"],
        "Quiz": ["Quick Check Quiz", "Chapter-End Quiz", "Adaptive Quiz"],
        "Practice Problems": ["Practice Problem Set", "Drill Exercises", "Worksheet"],
        "Interactive Lab": ["Virtual Lab", "Interactive Simulation", "Hands-on Lab"],
        "Case Study": ["Real-World Case Study", "Applied Case Analysis"],
        "Mini Project": ["Guided Mini Project", "Capstone Mini Project"],
        "PDF Notes": ["Concise Notes PDF", "Detailed Study Notes"],
    }

    for _, row in topic_catalog.iterrows():
        subject, topic, difficulty = row["Subject"], row["Topic"], row["Difficulty"]
        n_resources = np.random.randint(1, 3)  # 1-2 resources per topic
        chosen_types = np.random.choice(RESOURCE_TYPES, size=n_resources, replace=False)

        for r_type in chosen_types:
            base_name = np.random.choice(resource_name_bank[r_type])
            resource_name = f"{base_name}: {topic}"
            duration = int(np.random.randint(5, 90))
            rating = round(float(np.clip(np.random.normal(4.2, 0.5), 2.5, 5.0)), 1)
            url = f"https://lms.example.com/resources/{slugify(subject)}/{slugify(topic)}/{slugify(r_type)}"

            rows.append({
                "Topic": topic,
                "Resource_Name": resource_name,
                "Resource_Type": r_type,
                "Difficulty": difficulty,
                "Duration": duration,          # minutes
                "Rating": rating,               # 1-5 scale
                "URL_Placeholder": url,
            })

    return pd.DataFrame(rows)


# ==========================================================================
# 7. VALIDATION (lightweight sanity checks before writing to disk)
# ==========================================================================

def validate(student_df, subject_df, topic_df, history_df, resource_df):
    errors = []

    if not student_df["Student_ID"].is_unique:
        errors.append("Duplicate Student_ID found in student_profiles.")

    valid_subjects = set(subject_df["Subject"])
    hist_subjects = set(history_df["Subject"])
    if not hist_subjects.issubset(valid_subjects):
        errors.append("learning_history references subjects not in subject_catalog.")

    valid_topics = set(zip(topic_df["Subject"], topic_df["Topic"]))
    hist_topics = set(zip(history_df["Subject"], history_df["Topic"]))
    if not hist_topics.issubset(valid_topics):
        errors.append("learning_history references (Subject, Topic) pairs not in topic_catalog.")

    valid_resource_topics = set(topic_df["Topic"])
    if not set(resource_df["Topic"]).issubset(valid_resource_topics):
        errors.append("resource_catalog references topics not in topic_catalog.")

    # Spot-check: every learning_history row's subject must be in that
    # student's Registered_Subjects.
    student_subjects = student_df.set_index("Student_ID")["Registered_Subjects"].to_dict()
    sample = history_df.sample(min(2000, len(history_df)), random_state=SEED)
    bad = sample[~sample.apply(
        lambda r: r["Subject"] in student_subjects.get(r["Student_ID"], "").split("; "), axis=1
    )]
    if len(bad) > 0:
        errors.append(f"{len(bad)} learning_history rows study a subject outside the student's registration.")

    if len(topic_df) < 3000:
        errors.append(f"topic_catalog has only {len(topic_df)} rows (< 3000 requirement).")

    if errors:
        raise AssertionError("Data validation failed:\n- " + "\n- ".join(errors))

    print("✔ All consistency checks passed.")


# ==========================================================================
# 8. MAIN ORCHESTRATION
# ==========================================================================

def main(n_students=10000, n_history_records=50000):
    print("Building subject_catalog ...")
    subject_catalog = build_subject_catalog()

    print("Building topic_catalog ...")
    topic_catalog = build_topic_catalog(subject_catalog)

    print("Building student_profiles ...")
    student_profiles, _ = build_student_profiles(n_students)

    print("Building learning_history ...")
    learning_history = build_learning_history(student_profiles, topic_catalog, n_history_records)

    print("Building resource_catalog ...")
    resource_catalog = build_resource_catalog(topic_catalog)

    print("Validating referential integrity ...")
    validate(student_profiles, subject_catalog, topic_catalog, learning_history, resource_catalog)

    print("Saving CSV files ...")
    student_profiles.to_csv(os.path.join(OUTPUT_DIR, "student_profiles.csv"), index=False)
    subject_catalog.to_csv(os.path.join(OUTPUT_DIR, "subject_catalog.csv"), index=False)
    topic_catalog.to_csv(os.path.join(OUTPUT_DIR, "topic_catalog.csv"), index=False)
    learning_history.to_csv(os.path.join(OUTPUT_DIR, "learning_history.csv"), index=False)
    resource_catalog.to_csv(os.path.join(OUTPUT_DIR, "resource_catalog.csv"), index=False)

    print("\n===== SUMMARY =====")
    print(f"student_profiles.csv : {len(student_profiles):,} rows")
    print(f"subject_catalog.csv  : {len(subject_catalog):,} rows")
    print(f"topic_catalog.csv    : {len(topic_catalog):,} rows")
    print(f"learning_history.csv : {len(learning_history):,} rows")
    print(f"resource_catalog.csv : {len(resource_catalog):,} rows")
    print(f"\nAll files saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main(n_students=10000, n_history_records=50000)