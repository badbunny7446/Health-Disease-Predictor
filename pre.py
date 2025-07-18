import pandas as pd
import pickle

ab = pd.read_csv("final_health_data.csv")
model = pickle.load(open("disease_model.pkl", "rb"))

symptom_columns = [col for col in ab.columns if col not in ['disease', 'symtoms_score']]
symptom_map = {i+1: symptom for i, symptom in enumerate(symptom_columns)}
predict_option = len(symptom_map) + 1
symptom_map[predict_option] = 'Predict'

while True:
    print("\nSelect Symptoms (Enter numbers separated by comma):\n")
    for num, name in symptom_map.items():
        print(f"{num}. {name}")
        
    user_input = input("\nYour Choice: ").strip()

    try:
        choices = list(map(int, user_input.split(",")))

        if predict_option not in choices or choices[-1] != predict_option:
            print("\nPlease include 'Predict' as the LAST number to proceed.")
            continue

        selected_symptoms = [symptom_map[c] for c in choices if c != predict_option and c in symptom_map]

        if not selected_symptoms:
            print("\nNo symptoms selected. Please select at least one symptom.")
            continue

        input_dict = {col: 1 if col in selected_symptoms else 0 for col in symptom_columns}
        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df).max()

        print(f"\nPredicted Disease: {prediction}")
        print(f"Risk Score: {round(probability * 100, 2)}%")

    except Exception:
        print("\nInvalid input. Please try again.")
