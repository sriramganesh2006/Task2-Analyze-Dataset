import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Dataset Analysis",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Student Performance Dataset Analysis")

st.write(
    "An interactive application for analyzing the "
    "UCI Student Performance Dataset using Python."
)

st.divider()


# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = Path("data/student-mat.csv")


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    # Read the file as text first.
    # This makes the program robust to the format of the
    # student-mat.csv file.

    with open(DATA_PATH, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    # Remove empty lines
    lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    # --------------------------------------------------------
    # Read column names
    # --------------------------------------------------------

    columns = lines[0].split(";")

    columns = [
        column.strip().strip('"')
        for column in columns
    ]

    # --------------------------------------------------------
    # Read student records
    # --------------------------------------------------------

    data = []

    for line in lines[1:]:

        values = line.split(";")

        values = [
            value.strip().strip('"')
            for value in values
        ]

        # Make sure every row has the same number
        # of values as the header.
        if len(values) < len(columns):

            values += [
                None
            ] * (
                len(columns) - len(values)
            )

        elif len(values) > len(columns):

            values = values[:len(columns)]

        data.append(values)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        data,
        columns=columns
    )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    for column in df.columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Convert the column to numeric if most
        # of its non-empty values are numeric.
        non_empty = df[column].notna().sum()

        if non_empty > 0:

            numeric_count = converted.notna().sum()

            if numeric_count / non_empty >= 0.80:

                df[column] = converted

    return df


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_PATH.exists():

    st.error(
        "❌ Dataset not found!"
    )

    st.info(
        "Please place student-mat.csv inside:\n\n"
        "C:\\Task2_Analyze_Dataset\\data"
    )

    st.stop()


# Load dataset
df = load_data()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("1. Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


# Total students
col1.metric(
    "Total Students",
    f"{len(df):,}"
)


# Total features
col2.metric(
    "Total Features",
    f"{len(df.columns):,}"
)


# Missing values
missing_total = int(
    df.isnull().sum().sum()
)

col3.metric(
    "Missing Values",
    f"{missing_total:,}"
)


# Duplicate rows
duplicate_total = int(
    df.duplicated().sum()
)

col4.metric(
    "Duplicate Rows",
    f"{duplicate_total:,}"
)


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# DATASET SHAPE
# ============================================================

st.subheader("Dataset Shape")

st.write(
    f"The dataset contains **{df.shape[0]} rows** "
    f"and **{df.shape[1]} columns**."
)


# ============================================================
# DATASET FEATURES
# ============================================================

st.header("2. Dataset Features")


feature_info = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": [
        str(dtype)
        for dtype in df.dtypes
    ],
    "Missing Values": [
        int(value)
        for value in df.isnull().sum()
    ],
    "Unique Values": [
        int(value)
        for value in df.nunique()
    ]
})


st.dataframe(
    feature_info,
    use_container_width=True
)


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

st.header("3. Statistical Analysis")


numeric_columns = list(
    df.select_dtypes(
        include=np.number
    ).columns
)


if len(numeric_columns) > 0:

    statistics = df[
        numeric_columns
    ].describe().T

    statistics = statistics.round(2)

    st.dataframe(
        statistics,
        use_container_width=True
    )

else:

    st.warning(
        "No numeric columns were found."
    )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

st.header("4. Missing Value Analysis")


missing_values = df.isnull().sum()

missing_values = missing_values[
    missing_values > 0
]


if len(missing_values) == 0:

    st.success(
        "✅ No missing values were found."
    )

else:

    missing_table = pd.DataFrame({
        "Feature": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(
        missing_table,
        use_container_width=True
    )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

st.header("5. Duplicate Row Analysis")


if duplicate_total == 0:

    st.success(
        "✅ No duplicate rows were found."
    )

else:

    st.warning(
        f"⚠️ {duplicate_total} duplicate rows were found."
    )


# ============================================================
# STUDENT PERFORMANCE ANALYSIS
# ============================================================

st.header("6. Student Performance Analysis")


if "G3" in df.columns:

    # Convert G3 to numeric
    df["G3"] = pd.to_numeric(
        df["G3"],
        errors="coerce"
    )

    average_grade = df["G3"].mean()
    highest_grade = df["G3"].max()
    lowest_grade = df["G3"].min()


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Average Final Grade",
        f"{average_grade:.2f}"
    )


    col2.metric(
        "Highest Final Grade",
        f"{highest_grade:.0f}"
    )


    col3.metric(
        "Lowest Final Grade",
        f"{lowest_grade:.0f}"
    )


else:

    st.error(
        "G3 column was not found."
    )


# ============================================================
# FINAL GRADE DISTRIBUTION
# ============================================================

if "G3" in df.columns:

    st.subheader(
        "Final Grade Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.hist(
        df["G3"].dropna(),
        bins=range(
            int(df["G3"].min()),
            int(df["G3"].max()) + 2
        ),
        edgecolor="black"
    )


    ax.set_title(
        "Distribution of Final Grades"
    )

    ax.set_xlabel(
        "Final Grade (G3)"
    )

    ax.set_ylabel(
        "Number of Students"
    )


    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# STUDY TIME ANALYSIS
# ============================================================

if (
    "studytime" in df.columns
    and "G3" in df.columns
):

    st.subheader(
        "Study Time vs Final Grade"
    )


    study_analysis = (
        df.groupby("studytime")["G3"]
        .mean()
        .reset_index()
    )


    study_analysis.columns = [
        "Study Time",
        "Average Final Grade"
    ]


    study_analysis[
        "Average Final Grade"
    ] = study_analysis[
        "Average Final Grade"
    ].round(2)


    st.dataframe(
        study_analysis,
        use_container_width=True
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.bar(
        study_analysis["Study Time"],
        study_analysis["Average Final Grade"]
    )


    ax.set_title(
        "Average Final Grade by Study Time"
    )

    ax.set_xlabel(
        "Study Time Category"
    )

    ax.set_ylabel(
        "Average Final Grade"
    )


    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# ABSENCES VS FINAL GRADE
# ============================================================

if (
    "absences" in df.columns
    and "G3" in df.columns
):

    st.subheader(
        "Absences vs Final Grade"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.scatter(
        df["absences"],
        df["G3"],
        alpha=0.6
    )


    ax.set_title(
        "Absences vs Final Grade"
    )

    ax.set_xlabel(
        "Number of Absences"
    )

    ax.set_ylabel(
        "Final Grade (G3)"
    )


    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

st.header("7. Correlation Analysis")


if len(numeric_columns) >= 2:

    correlation = df[
        numeric_columns
    ].corr().round(2)


    st.dataframe(
        correlation,
        use_container_width=True
    )


    # Correlation heatmap
    st.subheader(
        "Correlation Heatmap"
    )


    fig, ax = plt.subplots(
        figsize=(12, 8)
    )


    image = ax.imshow(
        correlation,
        aspect="auto"
    )


    ax.set_xticks(
        range(len(correlation.columns))
    )

    ax.set_yticks(
        range(len(correlation.columns))
    )


    ax.set_xticklabels(
        correlation.columns,
        rotation=90
    )

    ax.set_yticklabels(
        correlation.columns
    )


    fig.colorbar(
        image,
        ax=ax
    )


    ax.set_title(
        "Correlation Between Numeric Features"
    )


    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# KEY FINDINGS
# ============================================================

st.header("8. Key Findings")


if "G3" in df.columns:

    st.write(
        f"""
        - The dataset contains **{len(df)} students**.
        - There are **{len(df.columns)} features**.
        - The average final grade is **{average_grade:.2f}**.
        - The highest final grade is **{highest_grade:.0f}**.
        - The lowest final grade is **{lowest_grade:.0f}**.
        - The dataset contains **{missing_total} missing values**.
        - The dataset contains **{duplicate_total} duplicate rows**.
        """
    )

else:

    st.write(
        f"""
        - The dataset contains **{len(df)} students**.
        - There are **{len(df.columns)} features**.
        - The dataset contains **{missing_total} missing values**.
        - The dataset contains **{duplicate_total} duplicate rows**.
        """
    )


# ============================================================
# CONCLUSION
# ============================================================

st.header("9. Conclusion")


st.write(
    """
    The Student Performance Dataset was successfully analyzed.

    The application provides information about the dataset structure,
    number of students, features, missing values, duplicate records,
    statistical characteristics and student performance.

    Visualizations were created to understand final-grade distribution,
    study time, absences and relationships between numerical features.

    This analysis provides a foundation for further Machine Learning
    tasks such as regression, classification and clustering.
    """
)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    "✅ Task 2 - Dataset Analysis completed successfully!"
)