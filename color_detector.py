import webcolors


def closest_color(rgb):
    differences = {}
    for color_hex, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
        r, g, b = webcolors.hex_to_rgb(color_hex)
        difference = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2
        differences[difference] = color_name
    return differences[min(differences.keys())]


# Exemple d'utilisation
rgb_color = (23, 45, 78)  # Rouge en RVB
closest = closest_color(rgb_color)
print(f"La couleur la plus proche est : {closest}")
