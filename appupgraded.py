import streamlit as st

# Icons
email_icon = "📧"
linkedin_icon = "🔗"

st.title("Govind Immidichetty Portfolio")

# Tabs for each section
tabs = st.tabs([
    "Introduction",
    "Professional Summary",
    "Work Experience",
    "Technical Skills",
    "Projects",
    "Testing Skills",
    "Academic Information",
    "Personal Details"
])

with tabs[0]:
    st.header("Introduction")
    st.write("📱 +91-7680809427")
    st.write(f"{email_icon} saigovindimmidichetty2105@gmail.com")
    st.write(f"{linkedin_icon} [LinkedIn](https://www.linkedin.com/in/govind-immidichetty-63ba5b127?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")

    st.markdown("""
    <a href="https://github.com/Govind-Star18" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" style="vertical-align:middle; margin-right:8px;">
        GitHub
    </a>
    """, unsafe_allow_html=True)


with tabs[1]:
    st.header("Professional Summary")
    st.markdown("""
➤ Engineered and curated high-quality datasets using Python libraries like **Pandas** and **NumPy**, ensuring clean, well-structured data pipelines for training and evaluating machine learning models.  
➤ Designed, trained, and validated machine learning models (classification, regression) using **Scikit-learn**, achieving measurable improvements in accuracy and robustness.  
➤ Performed model evaluation and fine-tuning, leveraging metrics like **F1 score**, **precision-recall**, and **confusion matrix** to optimize performance.  
➤ Developed deep learning models using **TensorFlow** for tasks in **Natural Language Processing (NLP)** and **Neural Networks**, including text classification and sentiment analysis.  
➤ Created insightful visualizations using **Matplotlib** and **Seaborn** for model explainability and stakeholder reporting.  
➤ Automated repetitive AI workflows by scripting in Python and integrating pipelines with **Docker**, enhancing reproducibility and development efficiency.  
➤ Collaborated with DevOps teams to containerize and deploy models in scalable environments using **Docker** and versioned the entire codebase using **Git**.  
➤ Researched emerging AI methodologies and experimented with state-of-the-art techniques to integrate into ongoing projects for continuous innovation.  
   
Software Testing:
                
➤ Analyzed system requirements and functional specifications to design high-impact test strategies.  
➤ Created and executed detailed **test cases** based on system requirements and business logic.  
➤ Conducted **GUI**, **functional**, **smoke**, **sanity**, **regression**, and **retesting** phases across web-based applications.  
➤ Automated test scenarios using **Selenium WebDriver**, leveraging **POM (Page Object Model)** and a **Hybrid Framework** for reusable and scalable scripts.  
➤ Authored dynamic **XPath expressions** to interact with UI elements across different browsers and environments.  
➤ Participated in **Sprint Planning**, **Daily Standups**, **Sprint Reviews**, and **Retrospectives** as part of Agile development.  
➤ Collaborated with developers to identify, report, and track bugs, assigning them to test cases using **Rally (Agile Test Management Tool)**.  
                """)

with tabs[2]:
    st.header("Work Experience")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("AI/ML Engineer – Lyros Technologies Pvt Ltd")
        st.write("📅 January 2025 – Present")
        st.markdown("""
- Designed and implemented scalable ML solutions using **Python**, **Scikit-learn**, and **TensorFlow** for classification and NLP tasks.  
- Built data pipelines and conducted preprocessing using **Pandas** and **NumPy** to support robust model training and inference.  
- Developed **interactive dashboards with Streamlit** and exposed ML predictions through **Flask REST APIs**.  
- Integrated containerized ML workflows using **Docker** and deployed them in collaborative agile environments.  
- Version-controlled ML experiments and production codebases using **Git** in a CI/CD pipeline.  
- Conducted model evaluation using metrics like **F1 Score**, **Precision**, and **Recall**, with iterative optimization via GridSearchCV.  
- Created rich visualizations using **Matplotlib** and **Seaborn** for internal demos and stakeholder reporting.  
- Collaborated with cross-functional teams to identify AI opportunities and continuously improve ML model performance and deployment strategies.  
        """)

    with col2:
        st.image(r"C:\Users\pandu\Downloads\LyrosLogo.jpg", width=1000)

with tabs[3]:
    st.header("Technical Skills")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Languages & Libraries")
        st.markdown("""
        - **Python**  
        - **NumPy**, **Pandas**  
        - **Scikit-Learn**  
        - **TensorFlow**
        - **Matplotlib**, **Seaborn**
        """)

        st.subheader("ML Techniques")
        st.markdown("""
        - Classification & Regression  
        - Clustering & Dimensionality Reduction  
        - Model Evaluation (F1, Precision, Recall)  
        - Feature Engineering  
        - Hyperparameter Tuning
        """)

    with col2:
        st.subheader("Deployment & Tools")
        st.markdown("""
        - **Flask** (REST APIs)  
        - **Streamlit** (Interactive Dashboards)  
        - **Docker** (Containerization)  
        - **Git** (Version Control)  
        - **VS Code**, **Jupyter**, **PyCharm**
        """)

        st.subheader("ML Lifecycle")
        st.markdown("""
        - Data Collection & Preprocessing  
        - Model Training & Evaluation  
        - Visualization & Tuning  
        - Deployment to Production  
        - Monitoring & Iteration
        """)

import pandas as pd 
with tabs[4]:
    st.header("Projects")
    with st.expander("1. Fertility – Fertility Prediction using Machine Learning"):
        st.write("""
        Objective: Predict fertility based on health and lifestyle indicators.  
        Tech Stack: Python, Pandas, Scikit-Learn, Matplotlib, Seaborn  
        Models Used: Logistic Regression, Decision Tree, Random Forest, SVM  
        Outcome: Achieved over 85% accuracy with Random Forest; deployed model via Streamlit
        """)

        st.subheader("Model Evaluation Metrics")
        
        # Data for visualization
        data = {
            "Model": ["Random Forest", "Logistic Regression", "Gradient Boosting"],
            "Precision": [0.884211, 0.872340, 0.872093],
            "Recall": [0.954545, 0.931818, 0.852273],
            "F1 Score": [0.918033, 0.901099, 0.862069]
        }
        df = pd.DataFrame(data)

        # Show as table
        st.dataframe(df)

        # Show bar chart for F1 Score
        st.subheader("F1 Score Comparison")
        chart_data = df[["Model", "F1 Score"]].set_index("Model")
        st.bar_chart(chart_data)
    import pandas as pd

with tabs[4]:
    

    # Airbnb Project
    import pandas as pd

with tabs[4]:
    

    with st.expander("2. Airbnb – Price Prediction & Host Analysis"):
        st.write("""
        Objective: Predict listing prices and analyze host behaviors and property trends.  
        Tech Stack: Python, Pandas, Scikit-Learn, Streamlit  
        Models Used: Linear Regression, Decision Tree Regressor, Random Forest Regressor  
        Outcome: Delivered insights for host optimization and price forecasting
        """)

        # Linear Regression Metrics
        st.subheader("Linear Regression - Evaluation Metrics")
        st.markdown("""
        - **Mean Absolute Error (MAE):** 0.37  
        - **Root Mean Squared Error (RMSE):** 0.49  
        - **R² Score:** 0.52  
        - **Intercept:** 3.9679  
        """)

        # Coefficients
        st.subheader("Linear Regression - Feature Coefficients")

        features = [
            "Feature1", "Feature2", "Feature3", "Feature4", "Feature5", 
            "Feature6", "Feature7", "Feature8", "Feature9", "Feature10", "Feature11"
        ]
        coefficients = [
            0.08061973, 0.14354731, 0.00694278, 0.15801914, -0.05616102,
            -0.63905181, -1.11562181, -0.05282358, -0.00591112, -0.06591049, -0.05747708
        ]

        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": coefficients
        }).set_index("Feature")

        st.dataframe(coef_df)
        st.bar_chart(coef_df)

        # Decision Tree Metrics
        st.subheader("Decision Tree Regressor - Evaluation Metrics")
        st.markdown("""
        - **Mean Absolute Error (MAE):** 0.37  
        - **R² Score:** 0.54  
        """)

        # Random Forest Metrics
        st.subheader("Random Forest Regressor - Evaluation Metrics")
        st.markdown("""
        - **Mean Absolute Error (MAE):** 0.36  
        - **R² Score:** 0.55  
        """)
    st.subheader("Case Study")

    with st.expander("E news Express – Article Categorization"):
        st.write("""
        Objective: Minimize manual effort in tagging articles by using unsupervised semantic clustering and sentence embeddings.  
        Approach: Built an unsupervised learning pipeline to group articles by theme (e.g., Business, Sports, Technology).  
        """)

        st.markdown("""
**🔧 Tools & Libraries Used**  
- **Transformers** (HuggingFace)  
- **Torch**, **Scikit-learn**, **SentenceTransformer**  
- **KMeans** for clustering  
- **Cosine similarity** for semantic search  

**🧠 Model Pipeline**  
- Used pre-trained model: `all-MiniLM-L6-v2` from `sentence-transformers`  
- Generated sentence embeddings using **SentenceTransformer**  
- Applied **KMeans clustering** to group articles into semantic categories  
- Performed semantic search to evaluate similarity and reduce duplication  

**📊 Classification Performance (Post-label Evaluation)**  
- Evaluated final clusters using precision, recall, and F1-score:
        """)

        report_data = {
            "Class": ["Business", "Entertainment", "Politics", "Sports", "Technology", "accuracy", "macro avg", "weighted avg"],
            "Precision": [0.96, 0.96, 0.96, 0.98, 0.91, 0.96, 0.95, 0.96],
            "Recall":    [0.93, 0.95, 0.95, 0.99, 0.97, 0.96, 0.96, 0.96],
            "F1-Score":  [0.95, 0.95, 0.95, 0.99, 0.93, 0.96, 0.96, 0.96],
            "Support":   [503, 369, 403, 505, 347, 2127, 2127, 2127]
        }

        report_df = pd.DataFrame(report_data).set_index("Class")
        st.dataframe(report_df)

        st.markdown("""
**✅ Outcome**  
- Achieved **96% accuracy** in categorizing unlabeled articles.  
- Significantly reduced manual tagging time through semantic clustering.  
- Successfully built a modular NLP pipeline ready for real-time deployment or expansion.
        """)  

with tabs[5]:
    st.header("Testing Skills")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        - Manual and Automation Testing using Selenium (Java)  
        - Test Planning, Test Design, Functional, Regression, Compatibility Testing  
        - Frameworks: TestNG, Maven, POM  
        - Bug Tracking: JIRA, RALLY
        """)
    with col2:
        st.image(r"C:\Users\pandu\Downloads\selelogo.png", width=700)

with tabs[6]:
    st.header("Academic Information")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("B.Tech in Electronics and Communication Engineering (ECE)")
        st.write("Lovely Professional University (LPU), 2019")
    with col2:
        st.image(r"C:\Users\pandu\Downloads\lpulogo.jpg", width=1500)

    st.subheader("Academic Project")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        - **Title**: Micro Strip Patch Antenna for 5G Applications  
        - **Role**: Team Leader  
        - **Software**: HFSS  
        - Designed antennas for RF to EM signal conversion for mobile & satellite communication
        """)
    with col2:
        st.image(r"C:\Users\pandu\Downloads\5gloog.jpg", width=700)

with tabs[7]:
    st.header("Personal Details")
    st.markdown("""
    - **DOB**: 21/09/1997  
    - **Gender**: Male  
    - **Languages**: English, Hindi, Telugu  
    - **Nationality/Religion**: Indian / Hindu
    """)
