# =====================================================
# FINAL — PERSONAL RECOMMENDATION ENGINE (INDONESIA)
# =====================================================

LOW_THRESHOLD = 70
HIGH_THRESHOLD = 130


FRIENDLY_NAME = {
    "Energy (kcal)": "Energi",
    "Carbohydrate (g)": "Karbohidrat",
    "Fat (g)": "Lemak",
    "Protein (g)": "Protein",
    "Calcium (mg)": "Kalsium",
    "Vitamin A (RE)": "Vitamin A",
    "Vitamin C (mg)": "Vitamin C",
    "Vitamin E (mg)": "Vitamin E",
    "Zinc (mg)": "Seng",
    "Iron (mg)": "Zat Besi"
}


# ===============================
# RENDAH — DI BAWAH TARGET
# ===============================

SUGGESTION_LOW = {

    "Energy (kcal)": "tambahkan sumber karbohidrat padat energi seperti bubur nasi, kentang, atau oat.",
    "Carbohydrate (g)": "tingkatkan porsi karbohidrat utama.",
    "Fat (g)": "tambahkan lemak sehat seperti minyak nabati, santan, atau alpukat.",
    "Protein (g)": "tambahkan sumber protein seperti telur, ayam, ikan, atau tahu.",
    "Calcium (mg)": "tambahkan susu, keju, atau ikan bertulang lunak.",
    "Vitamin A (RE)": "tambahkan sayuran berwarna oranye atau hijau tua seperti wortel atau bayam.",
    "Vitamin C (mg)": "tambahkan buah segar atau sayuran hijau.",
    "Vitamin E (mg)": "tambahkan kacang-kacangan, biji-bijian, atau minyak nabati.",
    "Zinc (mg)": "tambahkan daging, hati, atau telur.",
    "Iron (mg)": "tambahkan hati ayam atau daging merah."
}


# ===============================
# TINGGI — DI ATAS TARGET
# ===============================

SUGGESTION_HIGH = {

    "Energy (kcal)": "tingkat energi tinggi untuk satu porsi — perhatikan total asupan harian.",
    "Carbohydrate (g)": "kadar karbohidrat relatif tinggi — seimbangkan dengan sayur dan protein.",
    "Fat (g)": "kadar lemak tinggi — pertimbangkan penyesuaian porsi.",
    "Protein (g)": "kadar protein tinggi — baik untuk mendukung pertumbuhan.",
    "Calcium (mg)": "kadar kalsium tinggi — cukup untuk mendukung kesehatan tulang.",
    "Vitamin A (RE)": "kadar vitamin A tinggi — mendukung imun dan penglihatan.",
    "Vitamin C (mg)": "kadar vitamin C tinggi — mendukung fungsi imun.",
    "Vitamin E (mg)": "kadar vitamin E tinggi — mendukung perlindungan sel.",
    "Zinc (mg)": "kadar seng tinggi — mendukung pertumbuhan.",
    "Iron (mg)": "kadar zat besi tinggi — mendukung pembentukan darah."
}


# =====================================================
# FUNGSI UTAMA
# =====================================================

def build_recommendation(akg_percent_dict):

    recs = []

    for nut, val in akg_percent_dict.items():

        name = FRIENDLY_NAME.get(nut, nut)

        if val < LOW_THRESHOLD:

            recs.append(
                f"⚠️ {name} di bawah target — {SUGGESTION_LOW.get(nut,'disarankan penyesuaian menu.')}"
            )

        elif val > HIGH_THRESHOLD:

            recs.append(
                f"ℹ️ {name} di atas target — {SUGGESTION_HIGH.get(nut,'tingkat sudah tinggi.')}"
            )

        else:

            recs.append(
                f"✅ {name} cukup — memenuhi kebutuhan per porsi."
            )

    return recs