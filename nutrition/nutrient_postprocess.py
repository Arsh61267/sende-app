# =====================================================
# AKG REFERENCE — MPASI
# =====================================================

AKG_TABLE = {

    "6-9_bulan": {
        "Energi (kkal)": 800,
        "Karbo (g)": 105,
        "Lemak (g)": 35,
        "Protein (g)": 15,

        "Kalsium (mg)": 270,

        "Vit A (RE)": 400,
        "Vit C (mg)": 50,
        "Vit E (mcg)": 5,

        "Zinc (mg)": 3,
        "Zat Besi (mg)": 11
    },

    "9-11_bulan": {
        "Energi (kkal)": 850,
        "Karbo (g)": 110,
        "Lemak (g)": 36,
        "Protein (g)": 18,

        "Kalsium (mg)": 300,

        "Vit A (RE)": 400,
        "Vit C (mg)": 50,
        "Vit E (mcg)": 5,

        "Zinc (mg)": 3,
        "Zat Besi (mg)": 11
    },

    "12-23_bulan": {
        "Energi (kkal)": 1125,
        "Karbo (g)": 155,
        "Lemak (g)": 44,
        "Protein (g)": 26,

        "Kalsium (mg)": 650,

        "Vit A (RE)": 400,
        "Vit C (mg)": 40,
        "Vit E (mcg)": 6,

        "Zinc (mg)": 4,
        "Zat Besi (mg)": 7
    }
}


# =====================================================
# MAIN FUNCTION — SIDANG MODE (PER MENU)
# =====================================================

def compute_akg_percent(nutrisi_dict, age_group):

    if age_group not in AKG_TABLE:
        raise ValueError("Age group tidak dikenali")

    akg_daily = AKG_TABLE[age_group]

    percent = {}

    for k, val in nutrisi_dict.items():

        if k not in akg_daily:
            continue

        ref = akg_daily[k]

        if ref <= 0:
            continue

        percent[k] = round((val / ref) * 100, 1)

    return percent
