import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_price():

    try:

        year = int(year_entry.get())
        present_price = float(price_entry.get())
        kms_driven = int(kms_entry.get())

        fuel_type = fuel_combo.get()
        seller_type = seller_combo.get()
        transmission = transmission_combo.get()
        owner = int(owner_combo.get())


        # -------------------------------
        # VALIDATION
        # -------------------------------

        if year < 1990 or year > 2026:
            messagebox.showerror(
                "Invalid Input",
                "Please enter a valid manufacturing year."
            )
            return

        if present_price <= 0:
            messagebox.showerror(
                "Invalid Input",
                "Present price must be greater than 0."
            )
            return

        if kms_driven < 0:
            messagebox.showerror(
                "Invalid Input",
                "Kilometers cannot be negative."
            )
            return


        # -------------------------------
        # CREATE INPUT DATA
        # -------------------------------

        input_data = {
            "Year": year,
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Transmission_Manual":
                1 if transmission == "Manual" else 0,
            "Fuel_Type_Petrol":
                1 if fuel_type == "Petrol" else 0,
            "Fuel_Type_Diesel":
                1 if fuel_type == "Diesel" else 0,
            "Seller_Type_Individual":
                1 if seller_type == "Individual" else 0
        }


        input_df = pd.DataFrame([input_data])


        # Match training columns
        input_df = input_df.reindex(
            columns=model_columns,
            fill_value=0
        )


        # -------------------------------
        # PREDICTION
        # -------------------------------

        prediction = model.predict(input_df)[0]


        # -------------------------------
        # DISPLAY RESULT
        # -------------------------------

        result_label.config(
            text=f"₹ {prediction:.2f} Lakh"
        )

        result2_label.config(
            text=f"Approximately ₹ {prediction * 100000:,.0f}"
        )


    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid values in all fields."
        )


# ==========================================
# CLEAR FUNCTION
# ==========================================

def clear_fields():

    year_entry.delete(0, tk.END)
    price_entry.delete(0, tk.END)
    kms_entry.delete(0, tk.END)

    fuel_combo.set("Petrol")
    seller_combo.set("Dealer")
    transmission_combo.set("Manual")
    owner_combo.set("0")

    result_label.config(
        text="₹ 0.00 Lakh"
    )

    result2_label.config(
        text="Approximately ₹ 0"
    )


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title("Used Car Price Predictor")

root.geometry("720x760")

root.resizable(False, False)


# ==========================================
# HEADER
# ==========================================

header = tk.Frame(root)

header.pack(fill="x", pady=20)


title = tk.Label(
    header,
    text="🚗 USED CAR PRICE PREDICTOR",
    font=("Arial", 24, "bold")
)

title.pack()


subtitle = tk.Label(
    header,
    text="Machine Learning Based Price Estimation",
    font=("Arial", 11)
)

subtitle.pack(pady=5)


# ==========================================
# INPUT SECTION
# ==========================================

input_frame = tk.LabelFrame(
    root,
    text="  Enter Car Details  ",
    font=("Arial", 13, "bold"),
    padx=20,
    pady=15
)

input_frame.pack(
    padx=40,
    pady=10,
    fill="x"
)


# Year

tk.Label(
    input_frame,
    text="Manufacturing Year",
    font=("Arial", 11)
).grid(
    row=0,
    column=0,
    sticky="w",
    pady=9
)

year_entry = tk.Entry(
    input_frame,
    width=30,
    font=("Arial", 11)
)

year_entry.grid(
    row=0,
    column=1,
    padx=25
)


# Present Price

tk.Label(
    input_frame,
    text="Present Price (Lakh)",
    font=("Arial", 11)
).grid(
    row=1,
    column=0,
    sticky="w",
    pady=9
)

price_entry = tk.Entry(
    input_frame,
    width=30,
    font=("Arial", 11)
)

price_entry.grid(
    row=1,
    column=1,
    padx=25
)


# Kilometers

tk.Label(
    input_frame,
    text="Kilometers Driven",
    font=("Arial", 11)
).grid(
    row=2,
    column=0,
    sticky="w",
    pady=9
)

kms_entry = tk.Entry(
    input_frame,
    width=30,
    font=("Arial", 11)
)

kms_entry.grid(
    row=2,
    column=1,
    padx=25
)


# Fuel

tk.Label(
    input_frame,
    text="Fuel Type",
    font=("Arial", 11)
).grid(
    row=3,
    column=0,
    sticky="w",
    pady=9
)

fuel_combo = ttk.Combobox(
    input_frame,
    values=["Petrol", "Diesel", "CNG"],
    width=27,
    state="readonly"
)

fuel_combo.grid(
    row=3,
    column=1,
    padx=25
)

fuel_combo.set("Petrol")


# Seller

tk.Label(
    input_frame,
    text="Seller Type",
    font=("Arial", 11)
).grid(
    row=4,
    column=0,
    sticky="w",
    pady=9
)

seller_combo = ttk.Combobox(
    input_frame,
    values=["Dealer", "Individual"],
    width=27,
    state="readonly"
)

seller_combo.grid(
    row=4,
    column=1,
    padx=25
)

seller_combo.set("Dealer")


# Transmission

tk.Label(
    input_frame,
    text="Transmission",
    font=("Arial", 11)
).grid(
    row=5,
    column=0,
    sticky="w",
    pady=9
)

transmission_combo = ttk.Combobox(
    input_frame,
    values=["Manual", "Automatic"],
    width=27,
    state="readonly"
)

transmission_combo.grid(
    row=5,
    column=1,
    padx=25
)

transmission_combo.set("Manual")


# Owner

tk.Label(
    input_frame,
    text="Previous Owners",
    font=("Arial", 11)
).grid(
    row=6,
    column=0,
    sticky="w",
    pady=9
)

owner_combo = ttk.Combobox(
    input_frame,
    values=["0", "1", "2", "3"],
    width=27,
    state="readonly"
)

owner_combo.grid(
    row=6,
    column=1,
    padx=25
)

owner_combo.set("0")


# ==========================================
# BUTTONS
# ==========================================

button_frame = tk.Frame(root)

button_frame.pack(pady=18)


predict_button = tk.Button(
    button_frame,
    text="🔮  PREDICT PRICE",
    font=("Arial", 13, "bold"),
    command=predict_price,
    padx=25,
    pady=10
)

predict_button.grid(
    row=0,
    column=0,
    padx=10
)


clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    font=("Arial", 11),
    command=clear_fields,
    padx=25,
    pady=10
)

clear_button.grid(
    row=0,
    column=1,
    padx=10
)


# ==========================================
# RESULT SECTION
# ==========================================

result_frame = tk.LabelFrame(
    root,
    text="  Prediction Result  ",
    font=("Arial", 13, "bold"),
    padx=20,
    pady=15
)

result_frame.pack(
    padx=40,
    pady=5,
    fill="x"
)


result_label = tk.Label(
    result_frame,
    text="₹ 0.00 Lakh",
    font=("Arial", 25, "bold")
)

result_label.pack(pady=5)


result2_label = tk.Label(
    result_frame,
    text="Approximately ₹ 0",
    font=("Arial", 12)
)

result2_label.pack()


# ==========================================
# MODEL INFORMATION
# ==========================================

info_label = tk.Label(
    root,
    text="Model: Random Forest Regression   |   R² Score: 0.9652",
    font=("Arial", 10)
)

info_label.pack(pady=15)


# ==========================================
# FOOTER
# ==========================================

footer = tk.Label(
    root,
    text="Used Car Price Prediction Project",
    font=("Arial", 9)
)

footer.pack(
    side="bottom",
    pady=12
)


# ==========================================
# START APPLICATION
# ==========================================

root.mainloop()