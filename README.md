# Own-Polynomial-Regression-
Polynomial Regression Learning App
Overview
This is an interactive Streamlit application designed to help users explore and understand polynomial regression. It allows users to adjust dataset parameters (number of data points, noise level, slope, and intercept) and regression parameters (polynomial degree) to visualize how these changes affect the model's fit and performance metrics like Mean Squared Error (MSE) and R² score.
Features

Generate synthetic datasets with customizable parameters.
Visualize polynomial regression fits for different degrees.
Compare the current model with the optimal polynomial degree (based on minimum MSE).
Display model performance metrics and coefficients.
Educational content explaining MSE, R², and polynomial degrees.

Requirements
To run this application, you need Python 3.8+ and the following dependencies listed in requirements.txt:

streamlit
numpy
matplotlib
scikit-learn

Installation

Clone this repository or download the project files.
Create a virtual environment (recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required packages:pip install -r requirements.txt



Usage

Navigate to the project directory:cd path/to/project


Run the Streamlit app:streamlit run app.py


Open your web browser and go to http://localhost:8501 to interact with the app.

How to Use the App

Dataset Parameters (Sidebar):
Adjust the number of data points (10–500).
Set the noise level (0.0–10.0) to add randomness to the data.
Modify the true slope (-10.0 to 10.0) and intercept (-10.0 to 10.0) for the underlying linear relationship.


Regression Parameters (Sidebar):
Choose the polynomial degree (1–10) for the regression model.


Generate Data:
Click the "Generate New Dataset" button to create a new dataset based on your parameters.


Visualize and Learn:
View the scatter plot of data points, the true relationship (if linear), the current polynomial fit, and the optimal fit.
Check performance metrics (MSE, R²) and model coefficients.
Read the educational content to understand the results and experiment with different settings.



Project Structure

app.py: The main Streamlit application script.
requirements.txt: Lists the required Python packages.

Notes

The application uses a fixed random seed for reproducibility during development. Consider removing np.random.seed(42) for varied results in production.
Ensure the polynomial degree is less than the number of data points to avoid errors.
Higher polynomial degrees may lead to overfitting, especially with noisy data.

Contributing
Contributions are welcome! Please submit a pull request or open an issue to suggest improvements or report bugs.
License
This project is licensed under the MIT License.
