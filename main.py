import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.title("Curve Fitting App")


# Example fitting model (modify as needed)
def arrhenius(x, a, e, c):
    R = 8.314
    T = T_0 + x * c
    return a * np.exp(-e / (R * T))


def model(x, a, e, c):
    return np.log(arrhenius(x, a, e, c))


# Option 1: File Upload
st.subheader("Upload CSV File (X, Y)")
uploaded_file = st.file_uploader("Choose a table file", type=["csv", "xsl", "xlsx"])

if uploaded_file:
    for reader in [pd.read_csv, pd.read_excel]:
        try:
            df = reader(uploaded_file)
            break
        except:
            pass
    else:
        raise ValueError("Invalid file format")
    st.write("Uploaded Data:")
else:
    # Option 2: Manual Data Entry (Spreadsheet)
    st.subheader("Or Enter Data Manually")
    df = pd.DataFrame({"X": np.arange(5), "Y": np.exp(-np.arange(5))})
    df = st.data_editor(df, num_rows="dynamic")

x_data = df.values[:, 0]
y_data = df.values[:, 1]

# Inputs for initial guess of model parameters
st.subheader("Initial Guess for Model Parameters")
a_guess = st.number_input("A", value=y_data.mean())
e_guess = st.number_input("E_a", value=1000.0)
c_guess = st.number_input("C", value=0.0001)
T_0 = st.number_input("Ambient temperature assumed", value=298.0)

# Ensure data is not empty
if not df.empty:
    st.write("Final Data Used for Fitting:")
    st.write(df)

    # Convert DataFrame to numpy for fitting

    if st.button("Run Fitting"):
        try:
            params, _ = curve_fit(
                model, x_data, np.log(y_data), p0=[a_guess, e_guess, c_guess], method="lm", maxfev=10000
            )  # Initial guesses
            a_fit, b_fit, c_fit = params
            st.success(f"Fitted parameters: A={a_fit:.3e}, E_a={b_fit:.3e}, C={c_fit:.3e}")

            # Plot data and fit
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, label="Data", color="blue")
            ax.plot(x_data, arrhenius(x_data, *params), label="Fit", color="red")
            ax.legend()
            ax.set_title(f"Fitted parameters: A={a_fit:.3f}, E_a={b_fit:.3f}, C={c_fit:.3f}")
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Fitting error: {e}")
