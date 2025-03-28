import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

# Generate dataset
def generate_data():
    X, y = make_moons(n_samples=100, noise=0.4, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
def train_model(X_train, y_train, n_estimators, max_depth):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

# Plot decision boundary
def plot_frontier(model, X, y, title):
    plt.figure(figsize=(6, 4))
    plot_decision_regions(X, y, clf=model, legend=2)
    plt.title(title)
    st.pyplot(plt)

# Streamlit app
st.title("Comprendiendo el Sobreajuste con Random Forest")

# Load data
X_train, X_test, y_train, y_test = generate_data()

# Sidebar controls for model complexity
n_estimators = st.sidebar.slider("Número de Árboles", min_value=10, max_value=200, value=50, step=10)
max_depth = st.sidebar.slider("Profundidad Máxima de los Árboles", min_value=1, max_value=10, value=2, step=1)

# Train model with selected complexity
model = train_model(X_train, y_train, n_estimators, max_depth)

# Display plots
st.subheader("Frontera de Decisión en Datos de Entrenamiento")
plot_frontier(model, X_train, y_train, "Clasificación de Datos de Entrenamiento")

st.subheader("Frontera de Decisión en Datos de Prueba")
plot_frontier(model, X_test, y_test, "Clasificación de Datos de Prueba")

# # Explanation
# st.markdown(
#     """### ¿Qué sucede?
#     - Si la frontera es **demasiado simple**, podría no capturar el patrón real (subajuste).
#     - Si la frontera es **demasiado compleja**, podría memorizar los datos de entrenamiento pero fallar en los datos de prueba (sobreajuste).
#     - Ajusta el **Número de Árboles** y la **Profundidad Máxima** para ver cómo cambia la frontera de decisión.
#     """
# )