"""
Adversarial / Evasion Testing

Tests whether your model can still detect hate speech/offensive content
when people try to evade detection using common tricks:
- Adding spaces between letters (h a t e)
- Adding dashes between words
- Padding with unrelated neutral text (dilution attack)
- Repeating letters (informal evasion / typing style)
- Breaking words up with emoji

Uses your own confirmed-toxic examples as the base, then generates
evasion variants automatically and tests each through your live backend.
"""

import requests

API_URL = "http://127.0.0.1:8000/predict/ensemble"

# Base confirmed-toxic comments from your own dataset
BASE_COMMENTS = [
    "Yenu madidiri e lofer na",
    "Lofar beluga nana maga",
    "Sulle maga veerendra jain",
    "Loooaferr sullle maga ivanyuuuu",
    "Sule maga virendra bvc ninu",
    "ಧರ್ಮಸ್ಥಳದಲ್ಲಿ ಧರ್ಮವಿದೆ, ಈ ರಂಡಿ ಮಕ್ಳಳಲ್ಲಿ ಧರ್ಮವಿಲ್ಲ ಅಷ್ಟೇ thoo ಇವರ ಜನ್ಮಕ್ಕೆ",
    "Thu kantri ಕಜ್ಜಿ ನಾಯಿ sulemaga",
    "ninge ennu chaappli yetu biltade road ge bandre",
    "Lofer mindri sulle maga nakli devamanava",
    "Not intreasted... But must tell some thing... He is a fool... ಕಮಂಧರ ಜನ್ಮ ಜಾಲಡ ಬೇಕು.. ಭೂಕಳ್ಳ.",
    "Maga Huch hadsi maga",
    "Randi maga veerendra button",
    "ಸೂಳೆ ಮಗ ನೀನು",
    "bro adu hindugalu kattirodu adake alli hindugala kuruvugalive",
    "Taj Mahal story henge agidu hindu palace ge bandu akrama madkondu mele gommata madi adhu muslim",
]


def make_evasion_variants(text):
    """Generate common evasion tricks for a given piece of text."""
    variants = {}

    variants["spaced_out"] = " ".join(list(text.replace(" ", "")))
    variants["dashed"] = text.replace(" ", "-")
    variants["padded"] = f"Nice video bro really enjoyed it {text} thanks for sharing great content"
    variants["letter_repeat"] = text.replace("a", "aa").replace("e", "ee")

    words = text.split(" ")
    variants["emoji_broken"] = " \U0001F642 ".join(words)

    return variants


def classify(text):
    try:
        r = requests.post(API_URL, json={"text": text})
        result = r.json()
        return result.get("tier", "ERROR"), result.get("final_label", "-")
    except Exception as e:
        return "ERROR", str(e)


print("=" * 90)
print("ADVERSARIAL / EVASION TESTING")
print("=" * 90)

results_log = []

for base in BASE_COMMENTS:
    print(f"\n--- Base comment: {base}")
    tier, label = classify(base)
    print(f"    Original -> {tier} ({label})")
    results_log.append({"comment": base, "variant": "original", "tier": tier, "label": label})

    variants = make_evasion_variants(base)
    for variant_name, variant_text in variants.items():
        tier, label = classify(variant_text)
        print(f"    [{variant_name}] -> {tier} ({label})")
        print(f"        text: {variant_text[:80]}...")
        results_log.append({"comment": base, "variant": variant_name, "tier": tier, "label": label})

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

for base in BASE_COMMENTS:
    base_results = [r for r in results_log if r["comment"] == base]
    original_tier = base_results[0]["tier"]
    evaded_count = sum(1 for r in base_results[1:] if r["tier"] == "No Action")
    print(f"\n'{base[:50]}...'")
    print(f"  Original detection: {original_tier}")
    print(f"  Evasion variants that slipped through undetected: {evaded_count}/{len(base_results)-1}")