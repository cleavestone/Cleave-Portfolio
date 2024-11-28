import streamlit as st

st.title("Projects")

st.markdown("""## :blue[**Medical Chatbot**]
This **AI-powered Medical Chatbot** leverages **LangChain** for orchestrating advanced natural language processing tasks, **GPT** (Generative Pre-trained Transformer) for generating accurate and contextually relevant responses, and **Pinecone** for efficient retrieval of information. The chatbot is powered by the **Gale Encyclopedia of Medicine** as its knowledge base, providing authoritative and comprehensive medical information. By using **Pinecone** as the vector database, the chatbot can quickly retrieve relevant medical data and offer accurate, real-time responses to a wide range of health-related inquiries. 

The combination of **LangChain**, **GPT**, and **Pinecone** enables the chatbot to deliver insightful medical advice, direct users to appropriate resources, and improve over time through interaction.

### **Technologies Used**:
- **LangChain**: Framework for building sophisticated language model applications with external data sources.
- **GPT**: For natural language understanding and generating human-like responses.
- **Pinecone**: A vector database for efficient retrieval of relevant information.
- **Gale Encyclopedia of Medicine**: Comprehensive medical knowledge base for accurate, authoritative information.

**GITHUB LINK** (https://github.com/cleavestone/Medical-chatbot)""")



st.markdown("""## :blue[**Housing Prices Prediction Project**]
The objective of this project is to build a machine learning model that predicts the **median housing prices** in California based on various socio-economic metrics such as population, median income, and housing data for each district. The model will learn from the dataset to predict the median housing price for any given district in California.

#### **Technologies Used:**
- **Python**: For data manipulation and model building.
- **Pandas**: For data reading, processing, and exploratory data analysis (EDA).
- **NumPy**: For numerical operations and array handling.
- **Scikit-learn**: For machine learning algorithms, preprocessing, and model evaluation.
- **Matplotlib / Seaborn**: For data visualization and exploring relationships in the data.
- **Cross-validation**: For model validation to ensure robustness.
            
  **GITHUB LINK** (https://github.com/cleavestone/CARLIFORNIA-HOUSE-PREDICTION)         
""")
st.image(r"img5.png")

st.markdown("""### :blue[**Customer Segmentation Using RFM Analysis in E-commerce**]

#### **Project Overview:**
This project uses **RFM (Recency, Frequency, Monetary)** analysis to segment customers of an e-commerce platform based on their transaction history. The goal is to categorize customers into different segments with distinct behaviors, enabling targeted marketing strategies to improve customer engagement and satisfaction. The project involves analyzing customer data to derive insights and make recommendations for each segment.

#### **Key Metrics for Segmentation:**
- **Recency**: How recently a customer made a purchase.
- **Frequency**: How often a customer makes purchases.
- **Monetary**: How much a customer spends.

#### **Technologies Used:**
- **Python**: For data manipulation and analysis.
- **Pandas**: For data reading and cleaning.
- **NumPy**: For numerical operations.
- **Seaborn / Matplotlib**: For data visualization.
- **Scikit-learn**: For clustering (KMeans) and scaling.
- **PCA (Principal Component Analysis)**: For dimensionality reduction and visualization of customer segments.

#### **Objective:**
The project aims to segment customers into distinct groups using RFM analysis, identify customer behaviors, and provide actionable insights and recommendations for targeted marketing strategies.

GITHUB: (https://github.com/cleavestone/Customer-Segmentation/blob/main/Ecommerce_customer_segmentation.ipynb)           
""")
st.image(r"data\img3.png")

st.markdown("""### :blue[**Movie Recommendation System**]

#### **Project Overview:**
The objective of this project is to build a **content-based movie recommendation system** that suggests movies similar to a given movie. By leveraging movie metadata such as genres, cast, crew, and keywords, the system provides personalized movie recommendations. The project uses **Word2Vec embeddings** and **cosine similarity** to compute the similarity between movies based on their descriptions and features.

#### **Technologies Used:**
- **Python**: For data processing and implementation.
- **Pandas**: For handling and preprocessing datasets.
- **NumPy**: For numerical operations.
- **Matplotlib / Seaborn**: For visualizing data insights.
- **Gensim**: For training the Word2Vec model.
- **Sklearn**: For computing cosine similarity between embeddings.
- **Word2Vec**: For generating word embeddings based on movie descriptions.
- **Cosine Similarity**: For calculating the similarity between movie vectors to provide recommendations.
            
GITHUB: (https://github.com/cleavestone/Movie_recommender)
""")

st.image(r"data\pic2 (1).jpg",use_container_width=True)

st.markdown("""### :blue[**PM2.5 Air Quality Forecasting Project**]

#### **Project Overview:**
This project focuses on forecasting **PM2.5 concentration levels** to address air pollution challenges. Using a **Long Short-Term Memory (LSTM)** neural network, the project predicts PM2.5 concentrations based on various environmental factors such as temperature, humidity, atmospheric pressure, wind speed, and wind direction. The predictive model enables proactive decision-making for public health management and policy development.

#### **Key Objectives:**
- Develop a deep learning model to forecast PM2.5 levels.
- Leverage environmental data to train and evaluate the model.
- Visualize predicted vs. actual PM2.5 concentrations and provide future forecasts.

#### **Technologies Used:**
- **Python**: For data preprocessing, modeling, and visualization.
- **Pandas / NumPy**: For data manipulation and feature engineering.
- **Matplotlib / Seaborn**: For data visualization and exploratory analysis.
- **Statsmodels**: For seasonal decomposition and trend analysis.
- **TensorFlow / Keras**: For building and training the LSTM neural network.
- **Scikit-learn**: For normalization and model evaluation metrics (e.g., RMSE).
- **Jupyter Notebook**: For development, experimentation, and presentation of results.

#### **Key Achievements:**
- Successfully trained an LSTM model with a **Mean Absolute Error (MAE)** of 0.0283 and an **RMSE** of 40.47.
- Predicted PM2.5 levels for future hours, showcasing the model's practical utility.
- Provided visualizations of PM2.5 trends and forecasted values for better interpretation and decision-making.
""")

st.image(r"data\imggg.jpg")