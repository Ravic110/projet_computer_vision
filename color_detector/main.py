from Color_Identifier import ColorIdentifier

if __name__ == "__main__":
    try:
        color_identifier = ColorIdentifier(r"C:\Users\utilisateur\OneDrive\Bureau\victorien\projet ND\projet_computer_vision\color_detector\data\colors.csv")   
        color_identifier.run()
    except Exception as e:
        print(f"Erreur: {e}")