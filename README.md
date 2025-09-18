**Solar Power Analysis ->**

---

This repository contains the code for an interactive Streamlit dashboard designed to analyze solar power generation and weather data from Plant 1. The application offers a comprehensive and visual platform for exploring key performance metrics and the relationship between environmental factors and energy output.


**Features ->**

---

* Data Overview: Provides a summary of the generation and weather datasets, including data types, shapes, and missing values, for an initial understanding of the data's quality and structure.
* Generation Data Analysis: This section focuses on the performance of the solar plant. It includes visualizations for daily production distribution, AC/DC power trends over time, and a breakdown of power density per source key.
* Weather Data Analysis: This part of the dashboard is dedicated to analyzing the environmental data. It presents trends for ambient and module temperatures, irradiation levels, and their correlation with each other and with power generation.
* Interactive Visualizations: The dashboard leverages Plotly Express and Matplotlib to create a variety of informative charts, including time-series plots, heatmaps, and correlation matrices, allowing for dynamic data exploration.


**How to Run the Project ->**

---

1. Clone the Repository: Use the following command to clone the project to your local machine:
     git clone https://github.com/TusharBorase995/solar-generation-analysis.git
     cd solar-generation-analysis
2. Install Dependencies: The project relies on several Python libraries. You can install all of them at once using pip:
   pip install pandas numpy matplotlib seaborn streamlit plotly streamlit-option-menu
3. Run the Streamlit Application: With the dependencies installed, execute the following command in your terminal to start the application:

&nbsp;     streamlit run app.py

&nbsp;   


**Project Structure ->**

---

The project is structured as a single Python script (app.py), which handles all aspects of the application:

* Data Loading: The script uses the @st.cache\_data decorator to efficiently load and process the Plant\_1\_Generation\_Data.csv and Plant\_1\_Weather\_Sensor\_Data.csv files.
* User Interface: The Streamlit framework is used to create the interactive dashboard, including the sidebar menu and the main content sections.
* Data Analysis and Visualization: The code performs various data manipulations and generates plots using popular libraries like Pandas, Matplotlib, and Plotly.


**Technologies Used ->**

---

* Python: The core programming language.
* Streamlit: For creating the web application and interactive user interface.
* Pandas: For data manipulation and analysis.
* Matplotlib, Seaborn, Plotly: For data visualization.
* streamlit-option-menu: A custom component for enhanced sidebar navigation.
  

###### **License ->** This project is open-sourced under the MIT License.

