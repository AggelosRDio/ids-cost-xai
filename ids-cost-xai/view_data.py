import pandas as pd
import os

# Ορίζουμε τη διαδρομή του αρχείου CSV
# Χρησιμοποιούμε r'' για να αποφύγουμε προβλήματα με τα backslashes των Windows
file_path = r'data\nslkdd\processed\nslkdd_train.csv'

if os.path.exists(file_path):
    print("✅ Το αρχείο βρέθηκε! Φορτώνω δεδομένα...")
    df = pd.read_csv(file_path)
    
    print("\n--- Πρώτες 5 γραμμές των δεδομένων ---")
    print(df.head())
    
    print("\n--- Πληροφορίες συνόλου δεδομένων ---")
    print(f"Συνολικές γραμμές: {df.shape[0]}")
    print(f"Συνολικές στήλες: {df.shape[1]}")
else:
    print(f"❌ Σφάλμα: Το αρχείο δεν βρέθηκε στη διαδρομή: {file_path}")
    print("Σιγουρέψου ότι έτρεξες πρώτα το nslkdd.py για να προετοιμαστούν τα δεδομένα.")