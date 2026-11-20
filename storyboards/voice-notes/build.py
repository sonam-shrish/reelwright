"""Writes the "Voice Notes" storyboards (run: python storyboards/voice-notes/build.py, then
scripts/retime.py on them once the voices exist, because this overwrites the timing).

Ten micro-films about people reaching each other through a voice: voice notes, a phone call,
a radio call-in, a translator app. Shot like small real films: 35mm look, ordinary people,
real places, sound design, designed voices with emotion, subtitles. About 18-22 s each.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent
STYLE = ("35mm motion picture film still, tungsten film stock, natural available light, no logos, no brand names, candid and unposed, real skin "
         "texture with pores and small imperfections, ordinary everyday person, lived-in real location, "
         "slightly off-center framing, shallow depth of field, soft film grain, muted natural colors")
HANDHELD = "handheld camera with slight natural sway, documentary feel, "


# For shots with no people, or where Wan invents extra people: no "camera" wording (Wan reads
# "handheld camera" literally and adds someone holding a camera), and a locked-off frame.
STEADY = "steady locked-off frame, "
EMPTY = ", nobody else appears and nobody enters the frame"


def shot(still, motion, with_=None, trim=(0.3, 3.3), seconds=3.5, moves=("Handheld",), seed=1, sfx=None, vol=0.6,
         handheld=True):
    s = {"still": still, "motion": (HANDHELD if handheld else STEADY) + motion, "moves": list(moves),
         "seconds": seconds,
         "trim": list(trim), "seed": seed}
    if with_:
        s["with"] = with_
    if sfx:
        s["sfx"], s["sfx_volume"] = sfx, vol
    return s


def line(who, text, at, emotion=None, fx="", subtitle=None, strength=0.8, volume=1.4):
    d = {"who": who, "text": text, "at": at, "emotion": emotion or {"calm": 0.6}, "fx": fx,
         "strength": strength, "volume": volume}
    if subtitle:
        d["subtitle_text"] = subtitle
    return d


def film(n, slug, title, people, cast, shots, lines, ambience, music, caption):
    # 30 s versions: extra shots and lines inserted at their final positions (see EXTRA below)
    for idx, sh in EXTRA.get(slug, {}).get("shots", []):
        shots.insert(idx, sh)
    for idx, ln in EXTRA.get(slug, {}).get("lines", []):
        lines.insert(idx, ln)
    for name, c in cast.items():
        c.setdefault("label", LABELS.get(name, name.title()))
    sb = {"episode": n, "story_seconds": STORY_SECONDS, "end_card_seconds": 2.5, "series": "VOICE NOTES",
          "tagline": TAGLINES.get(slug, ""), "handle": "@_babelground", "title": title, "slug": slug, "look": "None", "style": STYLE, "aspect": "9:16", "reel": True,
          "film_look": True, "subtitles": True, "people": people, "cast": cast, "shots": shots, "lines": lines,
          "ambience": ambience, "music": music, "post_caption": caption}
    (OUT / f"{n:02d}-{slug}.json").write_text(json.dumps(sb, indent=1, ensure_ascii=False))


TAGS = "#VoiceNotes #HumanConnection #ShortFilm"
STORY_SECONDS = 27.5  # + a 2.5 s end card in Remotion = 30 s

# End-card line per film, and the names shown on the voice-message / call / radio pill.
TAGLINES = {
    "is-it-recording": "Keep their voices.", "wrong-number": "Say the kind thing.",
    "listen-to-the-pan": "Call home.", "last-cookie": "Leave them a voice note.",
    "ten-years": "Send the message.", "translator": "Love needs no translation.",
    "on-the-air": "Show up.", "fridge-notes": "Say sorry first.",
    "stuck": "Ask a stranger how they are.", "future-me": "Be kind to your younger self.",
}
LABELS = {
    "grandpa": "Grandpa", "maya": "Maya", "bev": "Auntie Bev", "daniel": "Daniel", "receptionist": "Reception",
    "amma": "Amma", "priya": "Priya", "leo": "Leo", "ana": "Ana", "sam": "Sam", "jonah": "Jonah",
    "halmeoni": "Halmeoni", "app": "Translate", "claire": "Claire", "host": "Night Lines 98.4",
    "dad": "Caller · Rick", "jess": "Jess", "nadia": "Nadia", "tom": "Tom", "ines": "Ines", "ray": "Ray",
    "teen": "Me, 16", "adult": "Me, 30",
}

# Extra beats for the 30-second cut: (position, shot) and (position, line), applied in order.
EXTRA = {
    "is-it-recording": {
        "shots": [(0, shot("Wide shot of an almost empty city night bus driving through a quiet rain-wet street at 3am, "
                           "warm interior lights glowing, reflections on the asphalt",
                           "the bus drives slowly through the empty wet street", None, seed=7121, moves=["Static"],
                           sfx="a city bus passing at night on a wet road", vol=0.5)),
                  (4, shot("Close-up of an old framed black-and-white 1960s wedding photograph: a smiling young bride "
                           "in a white lace dress and veil beside a young groom in a dark suit, both white, standing "
                           "outside a small church; the frame sits on a kitchen shelf next to a pair of reading "
                           "glasses, warm lamp light, dust in the air", "dust drifts slowly in the lamp light" + EMPTY,
                           None, seed=7124, moves=["Push-in Close-up"], handheld=False)),
                  (8, shot("standing on the pavement at dawn as a city bus pulls away behind her, holding her phone to "
                           "her ear, smiling with tired happy eyes", "the bus pulls away behind her, she smiles and "
                           "walks off", ["maya"], seed=7123, sfx="bus pulling away, early morning birds", vol=0.45))],
        "lines": [(3, line("grandpa", "She made me practice this four times.", 12.2, {"happy": 0.45, "calm": 0.3},
                           "voicenote"))]},
    "wrong-number": {
        "shots": [(0, shot("walking alone into a tall glass office lobby in the early morning, too-big suit, holding a "
                           "folder, looking up at the ceiling", "he walks in slowly and looks up, nervous", ["daniel"],
                           seed=7221, moves=["Tracking Shot"], sfx="busy office lobby, footsteps echo, revolving door",
                           vol=0.45)),
                  (5, shot("standing at her sunny kitchen window with the phone to her ear, looking out, a soft proud "
                           "smile", "she gazes out of the window, smiling softly", ["bev"], seed=7222)),
                  (8, shot("An office door with a frosted glass panel closing slowly in a plain corridor, an empty "
                           "plastic chair beside it with a paper cup left on it", "the closed door stays closed, only a faint shadow moves "
                           "behind the frosted glass" + EMPTY, None, seed=7224, moves=["Static"],
                           sfx="office door closing softly, latch click", vol=0.6, handheld=False))],
        "lines": [(3, line("bev", "Your mama would be so proud of you. I'm proud of you.", 10.0,
                           {"happy": 0.3, "melancholic": 0.3}, "phone"))]},
    "listen-to-the-pan": {
        "shots": [(0, shot("Wide shot of a tiny city apartment kitchen at night, smoke drifting under a hanging bulb, "
                           "a phone propped against a jar on the counter, nobody in frame", "smoke drifts slowly across the "
                           "empty small kitchen" + EMPTY, None, seed=7324, moves=["Static"], sfx="extractor fan, "
                           "sizzling pan, city outside", vol=0.45, handheld=False)),
                  (5, shot("in her kitchen stirring a big steel pot with one hand, laughing at her own story, steam "
                           "rising", "she stirs the pot and laughs", ["amma"], seed=7322)),
                  (8, shot("sitting on her small kitchen floor with a bowl of dal, phone to her ear, smiling and "
                           "rolling her eyes affectionately", "she eats and smiles while listening on the phone",
                           ["priya"], seed=7323))],
        "lines": [(3, line("amma", "Your nani never measured. I never measured. You will not measure.", 9.4,
                           {"happy": 0.4, "calm": 0.2}, "voicenote")),
                  (6, line("priya", "Yes, I'm eating. Amma says hi.", 24.4, {"happy": 0.45}, "room"))]},
    "last-cookie": {
        "shots": [(0, shot("Wide shot of a long empty hospital corridor at 3am under fluorescent lights, a nurse in blue "
                           "scrubs sitting alone on a chair at the far end", "the lights hum, the nurse sits still far "
                           "away", None, seed=7421, moves=["Static"], sfx="empty hospital corridor at night, "
                           "fluorescent hum, distant monitor beep", vol=0.45)),
                  (4, shot("Close-up of a small child's fingers pressing a glowing phone screen in the dark, a night "
                           "light shaped like a moon behind", "a small finger presses the glowing screen", None,
                           seed=7422, moves=["Static"])),
                  (8, shot("sitting on the edge of her bed at dawn still in scrubs, eating the cookie with her eyes "
                           "closed, smiling", "she eats the cookie slowly, eyes closed, smiling", ["ana"], seed=7423,
                           sfx="quiet bedroom at dawn, birds outside", vol=0.4))],
        "lines": [(3, line("leo", "Also, Grandma let me stay up till nine. Don't tell her I told you.", 10.8,
                           {"happy": 0.45, "surprised": 0.1}, "voicenote", strength=0.7)),
                  (6, line("ana", "Best cookie ever.", 25.0, {"happy": 0.5, "melancholic": 0.15}, "room",
                           volume=1.2))]},
    "ten-years": {
        "shots": [(0, shot("small in the frame on an apartment balcony at dusk, a big city skyline behind him, holding "
                           "a phone to his ear", "he stands still on the balcony as city lights come on", ["jonah"],
                           seed=7521, moves=["Static"], sfx="city skyline at dusk, distant sirens, wind", vol=0.45)),
                  (4, shot("Close-up of a man's thumb hovering over a phone screen in dusk light, screen glare hiding "
                           "the screen, then tapping", "a thumb hesitates over the phone, then taps", None, seed=7522,
                           moves=["Static"])),
                  (8, shot("Wide shot through a rain-streaked diner window at night, two men laughing in a booth "
                           "inside, neon reflections on the glass", "rain runs down the glass, they laugh inside",
                           ["jonah", "sam"], seed=7523, moves=["Static"], sfx="rain on a window, muffled diner "
                           "laughter", vol=0.5))],
        "lines": [(2, line("sam", "Remember the milkshake thing? We still owe that place a mop.", 8.0,
                           {"happy": 0.4, "melancholic": 0.2}, "voicenote")),
                  (6, line("jonah", "Milkshakes are on you.", 24.8, {"happy": 0.5}, "room"))]},
    "translator": {
        "shots": [(0, shot("Wide shot of a small warm apartment dining room in the evening, a low table crowded with "
                           "Korean dishes, an old woman and a young woman sitting across from each other in silence",
                           "they sit in awkward silence, steam rises from the dishes", ["halmeoni", "claire"],
                           seed=7621, moves=["Static"])),
                  (4, shot("nodding slowly while chewing, studying the young woman across the table with narrowed eyes",
                           "she chews slowly and studies her guest", ["halmeoni"], seed=7622)),
                  (8, shot("the grandmother leaning across the low table to put a piece of grilled fish into the young "
                           "woman's rice bowl with her chopsticks, both laughing warmly, warm lamp light, dishes everywhere",
                           "the grandmother puts the fish in the bowl with her chopsticks, both laugh and look at each "
                           "other; slow natural movement, faces stay the same" + EMPTY, ["halmeoni", "claire"], seed=7624,
                           moves=["Static"], sfx="chopsticks on ceramic, two women laughing softly", vol=0.55,
                           handheld=False))],
        "lines": [(4, line("claire", "Oh. How do I say thank you?", 13.6, {"happy": 0.4, "calm": 0.2}, "room")),
                  (5, line("halmeoni", "고마워요.", 15.6, {"calm": 0.6}, "room", "(Korean) Thank you."))]},
    "on-the-air": {
        "shots": [(0, shot("sitting alone in a parked car in a school parking lot at night in a graduation gown, "
                           "families in dresses walking past toward glowing doors", "families walk past the car, she "
                           "stays in her seat", ["jess"], seed=7721, moves=["Static"], sfx="parking lot at night, "
                           "families chatting, car doors", vol=0.4)),
                  (4, shot("glancing at a faded photo of a little girl tucked into the truck's sun visor while driving "
                           "at night, a small smile", "he glances up at the photo and smiles", ["dad"], seed=7722,
                           sfx="truck cab engine rumble", vol=0.45)),
                  (8, shot("climbing down from a big truck cab in a school parking lot at night, holding a small bunch "
                           "of supermarket flowers, nervous smile", "he climbs down from the cab with the flowers",
                           ["dad"], seed=7723, sfx="truck door slam, footsteps on asphalt", vol=0.6))],
        "lines": [(3, line("host", "Jess, if you're out there... that's one proud dad.", 11.0, {"calm": 0.7}, "radio")),
                  (5, line("dad", "Look out your window, kid.", 18.0, {"happy": 0.5}, "radio"))]},
    "fridge-notes": {
        "shots": [(0, shot("Wide shot of a small sunny kitchen in the morning, a fridge covered in old magnets and "
                           "photos, two coffee mugs on the counter, nobody there", "morning light moves slowly across the "
                           "empty quiet kitchen" + EMPTY, None, seed=7824, moves=["Static"], sfx="quiet kitchen "
                           "morning, clock ticking, birds", vol=0.4, handheld=False)),
                  (4, shot("writing on a yellow sticky note against the kitchen counter, tongue between his teeth, "
                           "concentrating hard", "he writes carefully, crosses something out, writes again", ["tom"],
                           seed=7822, sfx="pen scratching on paper", vol=0.6)),
                  (8, shot("Close-up of a fridge door covered in yellow sticky notes overlapping each other, warm "
                           "morning sun, text not readable", "sunlight slowly moves over the notes" + EMPTY, None,
                           seed=7825, moves=["Push-in Close-up"], handheld=False))],
        "lines": [(2, line("nadia", "He bought the good coffee. Coward.", 6.4, {"happy": 0.3, "angry": 0.15},
                           "", "(note) Bought the good coffee.")),
                  (4, line("tom", "Okay. Last note. Don't mess this up.", 12.0, {"afraid": 0.3, "calm": 0.3}, "",
                           "(thinking) Last note. Don't mess this up."))]},
    "stuck": {
        "shots": [(0, shot("rushing through a busy office building lobby at 8am, squeezing into an elevator just as "
                           "the doors close", "she rushes through the crowd and slips into the elevator", ["ines"],
                           seed=7921, moves=["Tracking Shot"], sfx="busy lobby, footsteps, elevator ding", vol=0.5)),
                  (5, shot("sitting on the elevator floor unwrapping a sandwich and offering half across, kind grin",
                           "he unwraps the sandwich and offers half", ["ray"], seed=7922,
                           sfx="paper wrapper crinkling", vol=0.6)),
                  (8, shot("walking out onto a sunny city street, pulling a small sketchbook out of her tote bag and "
                           "smiling at it", "she walks out into the sun and opens a sketchbook", ["ines"], seed=7923,
                           sfx="busy sunny street, traffic, birds", vol=0.45))],
        "lines": [(4, line("ray", "Sandwich? Half. Cheese and pickle. Best in London.", 12.8,
                           {"happy": 0.45, "calm": 0.2}, "room")),
                  (7, line("ray", "Ray. Now go buy some paint.", 24.8, {"happy": 0.45}, "room"))]},
    "future-me": {
        "shots": [(0, shot("Wide shot from outside of a suburban bedroom window at night glowing with fairy lights, a "
                           "teenage girl visible inside sitting on the floor with her phone", "the curtains move "
                           "slightly, the girl sits inside", ["teen"], seed=8021, moves=["Static"],
                           sfx="suburban street at night, crickets", vol=0.4)),
                  (4, shot("Close-up of an old phone with a cracked case lying on a bedroom carpet beside a diary and "
                           "a pencil, fairy light bokeh", "fairy lights twinkle softly", None, seed=8022,
                           moves=["Static"])),
                  (8, shot("walking out of an office building into golden evening light, putting her phone in her "
                           "pocket, taking a deep breath, smiling", "she walks out, breathes in deeply and smiles",
                           ["adult"], seed=8023, sfx="city street at golden hour, footsteps", vol=0.45))],
        "lines": [(2, line("teen", "Also, did we ever learn to drive? Mum says I'm a danger.", 7.6,
                           {"happy": 0.3, "afraid": 0.2}, "voicenote")),
                  (6, line("adult", "Also... we can drive now. Mostly.", 23.8, {"happy": 0.5}, "voicenote"))]},
}

# 1 ------------------------------------------------------------------------------------------
film(1, "is-it-recording", "Is it recording?",
     {"maya": {"look": "a fictional woman in her mid twenties with dark curly hair tied up loosely, tired eyes, "
               "olive skin, a worn navy scrub top under an open grey parka", "seed": 7101},
      "grandpa": {"look": "a fictional man in his late seventies with thin white hair, big reading glasses, "
                  "a brown cardigan over a checked shirt, kind lined face", "seed": 7102}},
     {"grandpa": {"voice": "an elderly American man in his late seventies, warm gravelly voice, a little unsure "
                  "of technology, slow and gentle", "sample": "Okay. Is this thing on? I think it's on."},
      "maya": {"voice": "a young American woman in her twenties, soft warm voice, tired after a night shift",
               "sample": "Hey. I'm on the bus. I just got off work."}},
     [shot("sitting by the window of an almost empty night bus, holding her phone to her ear, eyes closed, "
           "city lights smearing across the glass", "she listens with her eyes closed, the bus sways, lights slide "
           "across her face", ["maya"], seed=7111, sfx="city bus interior at night, engine hum, soft rattle", vol=0.5),
      shot("at a small kitchen table at night holding a smartphone far from his face, squinting at it through "
           "his glasses, one finger hovering, a wall clock and a teapot behind him",
           "he squints at the phone, holds it further away, leans in to speak", ["grandpa"], seed=7112,
           sfx="quiet kitchen at night, wall clock ticking, fridge hum", vol=0.5),
      shot("on the night bus, a small smile starting, she presses her lips together, eyes wet, looking out of "
           "the window", "she smiles and looks out of the window, blinking slowly", ["maya"], seed=7113,
           moves=["Push-in Close-up"]),
      shot("Close-up of an old man's hand with a wedding ring tapping carefully at a smartphone lying on a "
           "kitchen tablecloth, warm lamp light", "an old finger taps the phone screen twice, unsure", None,
           seed=7114, trim=(0.4, 2.6), moves=["Static"], sfx="soft tap on a phone screen, a small beep", vol=0.7),
      shot("at the kitchen table looking at the phone with suspicion over his glasses, a little proud of himself",
           "he looks at the phone suspiciously, then chuckles to himself", ["grandpa"], seed=7115),
      shot("on the night bus holding the phone close to her mouth, laughing quietly while recording a reply, "
           "dawn light starting outside", "she laughs softly and talks into the phone, the bus slows down",
           ["maya"], seed=7116, trim=(0.2, 3.4), sfx="bus brakes hissing softly, doors opening", vol=0.45)],
     [line("grandpa", "Is it recording? ...Okay. I think it's recording.", 0.15, {"surprised": 0.3, "calm": 0.4}, "voicenote"),
      line("grandpa", "Your grandmother says I should use these voice things. So. Hello, Maya.", 3.6,
           {"happy": 0.35, "calm": 0.4}, "voicenote"),
      line("grandpa", "I heard about the exam. I'm proud of you, kiddo. That's all.", 8.9,
           {"happy": 0.3, "melancholic": 0.35, "calm": 0.25}, "voicenote"),
      line("grandpa", "Now how do I stop this-", 13.0, {"surprised": 0.55, "happy": 0.2}, "voicenote"),
      line("maya", "Grandpa. It's still recording.", 15.6, {"happy": 0.55, "calm": 0.2}, "room")],
     "quiet city night bus interior, low engine rumble",
     {"description": "sparse felt piano, two warm notes, intimate and tender, slow, with soft room tone",
      "start": 8.5, "volume": 0.32, "fade_in": 1.5, "seed": 7190},
     f"He finally learned voice notes. 🎙️\n\nWho's the person whose voice you'd keep forever?\n\n{TAGS} #Grandpa #Family\nMade with AI")

# 2 ------------------------------------------------------------------------------------------
film(2, "wrong-number", "Wrong number",
     {"daniel": {"look": "a fictional young Black man in his early twenties, short twists, a slightly too big "
                 "borrowed navy suit and white shirt, nervous", "seed": 7201},
      "bev": {"look": "a fictional Caribbean woman in her sixties with short grey hair and gold hoop earrings, "
              "a bright floral blouse, reading glasses on a chain", "seed": 7202}},
     {"bev": {"voice": "a Jamaican woman in her sixties, warm, loud and loving, speaks fast, laughs easily",
              "sample": "Rosa, baby, it's your Auntie Bev. Listen to me now."},
      "daniel": {"voice": "a young Black British man in his twenties, soft spoken, a little nervous",
                 "sample": "Hi. Sorry. I think you have the wrong number."},
      "receptionist": {"voice": "a polite woman in her forties, neutral office voice", "sample": "Daniel? They're ready for you."}},
     [shot("sitting alone on a plastic chair in a plain office corridor, knee bouncing, phone to his ear, "
           "frowning in confusion", "his knee bounces nervously, he frowns at the voice on the phone", ["daniel"],
           seed=7211, sfx="quiet office corridor, distant printer, air conditioning", vol=0.45),
      shot("in her small sunny kitchen holding a phone up to her mouth, talking with her free hand, a pot steaming "
           "on the stove", "she talks animatedly into the phone, waving her hand", ["bev"], seed=7212,
           sfx="pot simmering, radio playing very quietly in the background", vol=0.4),
      shot("Close-up of a young man's hands holding a printed CV, the text blurred and out of focus, slightly crumpled corner, shallow depth of field",
           "his thumb smooths the crumpled corner of the paper", None, seed=7213, trim=(0.3, 2.5), moves=["Static"],
           sfx="paper rustle", vol=0.6),
      shot("in the corridor, a slow smile, sitting up straighter, still holding the phone to his ear",
           "he slowly smiles and straightens his back", ["daniel"], seed=7214, moves=["Push-in Close-up"]),
      shot("laughing into the phone in her kitchen, wiping her eye with a tea towel",
           "she laughs warmly into the phone", ["bev"], seed=7215),
      shot("standing up from the plastic chair, buttoning his jacket, an office door opening beside him",
           "he stands, buttons his jacket and walks toward the opening door", ["daniel"], seed=7216,
           trim=(0.2, 3.4), sfx="door opening, footsteps on office carpet", vol=0.5)],
     [line("bev", "Rosa, baby, it's Auntie Bev. Today's the big interview, eh?", 0.15, {"happy": 0.55}, "phone"),
      line("daniel", "...I'm not Rosa.", 3.4, {"surprised": 0.3, "calm": 0.4}, "room",),
      line("bev", "You walk in there with your head up. They need you more than you need them, you hear?", 5.6,
           {"happy": 0.4, "calm": 0.3}, "phone"),
      line("bev", "And eat something first! I love you, baby.", 11.3, {"happy": 0.6}, "phone"),
      line("receptionist", "Daniel? They're ready for you.", 14.7, {"calm": 0.6}, "room"),
      line("daniel", "Thank you, Auntie Bev.", 16.8, {"happy": 0.45, "calm": 0.3}, "", "(whispers) Thank you, Auntie Bev.", volume=1.1)],
     "quiet office corridor, distant phones",
     {"description": "warm understated soul guitar and soft brushed drums, hopeful, builds gently",
      "start": 9.0, "volume": 0.3, "fade_in": 1.2, "seed": 7290},
     f"Wrong number. Right words. ☎️\n\nWhat's the best thing a stranger ever told you?\n\n{TAGS} #Kindness #JobInterview\nMade with AI")

# 3 ------------------------------------------------------------------------------------------
film(3, "listen-to-the-pan", "Listen to the pan",
     {"priya": {"look": "a fictional Indian woman in her late twenties with long black hair in a messy bun, "
                "an oversized university sweatshirt, a tiny apartment kitchen", "seed": 7301},
      "amma": {"look": "a fictional Indian woman in her late fifties, silver streak in her black hair, a cotton "
               "salwar kameez, bangles, reading glasses pushed up on her head", "seed": 7302}},
     {"amma": {"voice": "an Indian mother in her late fifties, Indian English accent, quick, funny, bossy and loving",
               "sample": "Priya, are you listening? Don't measure anything. Listen."},
      "priya": {"voice": "a young British Indian woman in her late twenties, dry sense of humor, warm",
                "sample": "Amma, I followed the recipe exactly."}},
     [shot("in a tiny apartment kitchen waving smoke away from a frying pan with a tea towel, coughing, phone "
           "propped against a jar", "she waves smoke away from the pan and coughs, annoyed", ["priya"], seed=7311,
           sfx="hot oil sizzling, smoke alarm chirp once", vol=0.55),
      shot("in her own kitchen at home recording a voice note, gesturing at the camera with a wooden spoon like a "
           "teacher", "she lectures into the phone, pointing with a wooden spoon", ["amma"], seed=7312),
      shot("Side-angle macro close-up of cumin seeds frying and popping in shimmering golden oil inside a small open steel pan, tiny bubbles, wisps of smoke, dark kitchen behind",
           "cumin seeds crackle and jump in the hot oil", None, seed=7313, trim=(0.3, 2.6), moves=["Static"],
           sfx="cumin seeds crackling and popping in hot oil", vol=0.8),
      shot("leaning over the pan, listening to it with her head tilted, a small surprised smile",
           "she tilts her head, listening to the pan, then smiles", ["priya"], seed=7314,
           moves=["Push-in Close-up"]),
      shot("in her kitchen laughing at her own joke, then pointing at the phone seriously",
           "she laughs, then points at the phone sternly", ["amma"], seed=7315),
      shot("sitting on the kitchen counter eating dal from the pan with a spoon, eyes closed, then picking up "
           "the phone to call someone", "she tastes, closes her eyes, then lifts the phone to call", ["priya"],
           seed=7316, trim=(0.2, 3.4), sfx="spoon on steel pan, soft dial tone", vol=0.5)],
     [line("priya", "Amma, I followed your recipe exactly.", 0.15, {"angry": 0.2, "sad": 0.2, "calm": 0.3}, "room"),
      line("amma", "Recipe? What recipe? You don't measure. You listen to the pan.", 2.6,
           {"happy": 0.35, "surprised": 0.2, "calm": 0.2}, "voicenote"),
      line("amma", "When the cumin talks back, it's ready.", 6.2, {"happy": 0.3, "calm": 0.4}, "voicenote"),
      line("amma", "And call your father. He thinks you're angry with him. Silly man.", 11.4,
           {"happy": 0.4, "melancholic": 0.2}, "voicenote"),
      line("priya", "Hi, Appa. ...No, I'm not angry.", 21.8, {"happy": 0.4, "calm": 0.4}, "room")],
     "small apartment kitchen, extractor fan hum, city outside the window",
     {"description": "playful warm acoustic guitar with light tabla, cozy and homely",
      "start": 6.0, "volume": 0.3, "fade_in": 1.0, "seed": 7390},
     f"Moms don't give recipes. They give instructions for life. 🍲\n\nWhat's the advice your mom always repeats?\n\n{TAGS} #Mom #Cooking\nMade with AI")

# 4 ------------------------------------------------------------------------------------------
film(4, "last-cookie", "The last cookie",
     {"ana": {"look": "a fictional Latina woman in her thirties, hair in a low ponytail, light blue hospital "
              "scrubs, a lanyard, exhausted kind eyes", "seed": 7401},
      "leo": {"look": "a fictional six year old boy with messy brown hair and dinosaur pajamas, gap tooth",
              "seed": 7402}},
     {"leo": {"voice": "a six year old boy, small sleepy voice, whispering seriously, slight lisp",
              "sample": "Mama. It's me. I'm whispering because it's late."},
      "ana": {"voice": "a Latina woman in her thirties, soft tired voice, gentle", "sample": "Hi, my love. Mama's at work."}},
     [shot("sitting on a hospital stairwell step on her break, holding a paper cup of coffee and her phone, "
           "fluorescent light, eyes closed", "she leans her head against the wall and plays a voice note",
           ["ana"], seed=7411, sfx="hospital stairwell echo, distant pager beeps, ventilation hum", vol=0.45),
      shot("sitting in bed under a blanket fort holding a phone with both hands very close to his mouth, "
           "whispering, a night light glowing", "he whispers seriously into the phone, glancing at the door",
           ["leo"], seed=7412),
      shot("on the stairwell, laughing silently with tears in her tired eyes, covering her mouth",
           "she laughs silently, covering her mouth, eyes wet", ["ana"], seed=7413, moves=["Push-in Close-up"]),
      shot("Close-up of a single chocolate chip cookie wrapped in a paper napkin hidden under a pillow, dim "
           "morning light", "dawn light slowly crosses the napkin-wrapped cookie" + EMPTY, None,
           seed=7417, trim=(0.3, 2.6), moves=["Static"], handheld=False),
      shot("pushing through hospital double doors back onto a busy ward, straightening her lanyard, a small smile",
           "she pushes through the doors and walks back onto the ward", ["ana"], seed=7415,
           sfx="hospital doors swinging, ward chatter, monitors beeping", vol=0.4),
      shot("at dawn in a small apartment bedroom, lifting a pillow and finding a napkin-wrapped cookie, laughing "
           "softly, still in scrubs", "she lifts the pillow, finds the cookie and laughs softly", ["ana"],
           seed=7416, trim=(0.2, 3.4), sfx="birds at dawn, quiet apartment", vol=0.45)],
     [line("leo", "Mama. It's me. I'm whispering 'cause it's late.", 0.15, {"calm": 0.5, "afraid": 0.1}, "voicenote", strength=0.7),
      line("leo", "I brushed my teeth. Both sides.", 3.6, {"happy": 0.4, "calm": 0.3}, "voicenote", strength=0.7),
      line("leo", "And I saved you the last cookie. It's under the... I'm not telling.", 7.0,
           {"happy": 0.5, "surprised": 0.1}, "voicenote", strength=0.7),
      line("leo", "Goodnight, Mama. Don't be tired.", 11.6, {"calm": 0.5, "melancholic": 0.2}, "voicenote", strength=0.7),
      line("ana", "Found it.", 17.2, {"happy": 0.6}, "room")],
     "hospital stairwell at night, faint hum",
     {"description": "soft music box melody over gentle warm strings, lullaby-like, tender",
      "start": 3.4, "volume": 0.28, "fade_in": 2.0, "seed": 7490},
     f"Night shift. One voice note. 🍪\n\nTag a nurse who needs this today.\n\n{TAGS} #Nurses #Parenting\nMade with AI")

# 5 ------------------------------------------------------------------------------------------
film(5, "ten-years", "Ten years",
     {"jonah": {"look": "a fictional man in his mid thirties with a short dark beard and tired eyes, a faded "
                "green work jacket", "seed": 7501},
      "sam": {"look": "a fictional man in his mid thirties with sandy hair going thin, freckles, a navy wool "
              "coat and scarf", "seed": 7502}},
     {"sam": {"voice": "an American man in his thirties, hesitant, nervous laugh, sincere",
              "sample": "Hey. It's Sam. Yeah. I know."},
      "jonah": {"voice": "an American man in his thirties, low dry voice, quietly moved", "sample": "Sam. Wow. Hi."}},
     [shot("standing on a small apartment balcony at dusk with a phone to his ear, frozen, a half-eaten sandwich "
           "in his other hand", "he freezes, listening, the sandwich forgotten", ["jonah"], seed=7511,
           sfx="city at dusk from a balcony, distant traffic, a dog barking far away", vol=0.45),
      shot("sitting in a parked car in the rain holding his smartphone up near his mouth to speak a voice "
           "message, no microphone, rubbing his forehead with his other hand, rain on the windshield", "he rubs his forehead and keeps talking into the phone, rain streaks down",
           ["sam"], seed=7512, sfx="rain on a car roof, wipers off", vol=0.55),
      shot("Close-up of an old instant photo of two teenage boys laughing in a diner booth, held between fingers, "
           "slightly faded", "fingers tilt the old photo toward the light", None, seed=7513, trim=(0.3, 2.6),
           moves=["Static"]),
      shot("on the balcony, lowering the phone, staring at the city, deciding, jaw tight",
           "he lowers the phone and stares at the city, deciding", ["jonah"], seed=7514, moves=["Dolly In"]),
      shot("walking fast down a wet street at night under neon diner signs, hands in his jacket pockets",
           "he walks quickly down the wet street toward the diner", ["jonah"], seed=7515,
           sfx="footsteps on wet pavement, passing car", vol=0.5),
      shot("both men sitting across from each other in a worn diner booth with two coffees, both laughing, "
           "a little emotional", "they laugh across the table, one shakes his head", ["jonah", "sam"], seed=7516,
           trim=(0.2, 3.4), sfx="diner ambience, cutlery, coffee machine", vol=0.45)],
     [line("sam", "Hey. It's Sam. ...Yeah. I know. Ten years.", 0.15, {"afraid": 0.2, "melancholic": 0.3, "calm": 0.2}, "voicenote"),
      line("sam", "I'm in town till Sunday. That diner on Fifth is still there.", 4.6,
           {"calm": 0.4, "melancholic": 0.2}, "voicenote"),
      line("sam", "No pressure. I just... I missed my friend.", 8.4, {"sad": 0.35, "calm": 0.3}, "voicenote"),
      line("jonah", "You're late.", 21.8, {"happy": 0.4, "calm": 0.3}, "room"),
      line("sam", "Ten years late.", 23.0, {"happy": 0.55}, "room")],
     "city evening, distant traffic",
     {"description": "slow melancholic Rhodes piano and soft upright bass, late night, warm",
      "start": 7.8, "volume": 0.3, "fade_in": 1.5, "seed": 7590},
     f"Ten years of silence. One voice note. 🕰️\n\nWho's the friend you still think about?\n\n{TAGS} #Friendship #OldFriends\nMade with AI")

# 6 ------------------------------------------------------------------------------------------
film(6, "translator", "Lost in translation",
     {"halmeoni": {"look": "a fictional Korean grandmother in her seventies with short permed grey hair, a "
                   "lilac cardigan, a jade bracelet, a stern face that hides a smile", "seed": 7601},
      "claire": {"look": "a fictional woman in her late twenties with a short copper-red bob haircut and freckles, "
                 "wearing a cream knit sweater, friendly polite smile", "seed": 7602}},
     {"halmeoni": {"voice": "an elderly Korean woman in her seventies, firm, dry, secretly warm",
                   "sample": "어서 와. 앉아.", "language": "Korean"},
      "app": {"voice": "a neutral, slightly synthetic female voice assistant, clear and flat",
              "sample": "Translation. She says: please sit down."},
      "claire": {"voice": "a young American woman, nervous, polite, sweet", "sample": "Oh! Thank you. It's so nice to meet you."}},
     [shot("sitting at a low table covered with many small Korean side dishes, holding up a smartphone toward "
           "someone across the table, unsmiling", "she holds the phone out and speaks into it firmly",
           ["halmeoni"], seed=7611, sfx="small apartment, kettle, chopsticks on dishes", vol=0.4),
      shot("sitting at a low table full of Korean dishes, holding chopsticks awkwardly, leaning toward a phone "
           "held out to her, nervous smile", "she leans toward the phone and listens, smiling nervously",
           ["claire"], seed=7612),
      shot("Close-up of chopsticks placing a large piece of grilled fish onto a full rice bowl, then another",
           "chopsticks keep adding food to the already full rice bowl", None, seed=7613, trim=(0.3, 2.6),
           moves=["Static"], sfx="chopsticks clicking on a ceramic bowl", vol=0.6),
      shot("laughing out loud with her hand over her mouth at a low table full of food",
           "she bursts out laughing, covering her mouth", ["claire"], seed=7614, moves=["Push-in Close-up"]),
      shot("putting the phone face down on the table, looking straight ahead with a tiny smile",
           "she puts the phone down and looks up with a small smile", ["halmeoni"], seed=7615),
      shot("the grandmother and the young woman sitting side by side at the low table, the grandmother patting "
           "the back of the young woman's hand once, both smiling", "the grandmother pats her hand once, both smile",
           ["halmeoni", "claire"], seed=7616, trim=(0.2, 3.4))],
     [line("halmeoni", "너무 말랐어. 많이 먹어.", 0.15, {"calm": 0.5, "angry": 0.15}, "room",
           "(Korean) You're too skinny. Eat."),
      line("app", "Translation. She says: you are too skinny. Eat more.", 2.4, {"calm": 0.8}, "phone"),
      line("claire", "Oh! Okay. Tell her it's delicious.", 6.4, {"happy": 0.4, "afraid": 0.2}, "room"),
      line("app", "Translation. She says: you are now my granddaughter. Also, eat more.", 9.4, {"calm": 0.8}, "phone"),
      line("halmeoni", "Welcome. ...Family.", 15.4, {"happy": 0.35, "calm": 0.4}, "room")],
     "quiet apartment dinner, faint kitchen sounds",
     {"description": "gentle plucked gayageum and soft piano, warm and a little playful",
      "start": 6.0, "volume": 0.28, "fade_in": 1.5, "seed": 7690},
     f"Grandma doesn't need the app. 🍚\n\nWhat did your grandparents say when they met your partner?\n\n{TAGS} #Grandma #Family #Korean\nMade with AI")

# 7 ------------------------------------------------------------------------------------------
film(7, "on-the-air", "On the air",
     {"jess": {"look": "a fictional young woman of eighteen with long straight brown hair, a black graduation "
               "gown and cap, mascara slightly smudged", "seed": 7701},
      "dad": {"look": "a fictional man in his fifties with a grey goatee, a trucker cap and a flannel shirt, "
              "weathered hands", "seed": 7702}},
     {"host": {"voice": "a late-night radio host, deep smooth male voice, calm and kind", "sample": "You're on the air with Night Lines."},
      "dad": {"voice": "an American man in his fifties, rough voice, not used to talking about feelings",
              "sample": "Yeah, hi. Long time listener."},
      "jess": {"voice": "an American teenage girl, choked up, laughing through tears", "sample": "Dad? Oh my God."}},
     [shot("sitting alone in a parked car outside a school hall at night in a graduation gown, the car radio "
           "glowing, staring at the dashboard", "she stares at the glowing radio, eyes glistening", ["jess"],
           seed=7711, sfx="inside a parked car at night, muffled crowd far away", vol=0.4),
      shot("driving a big truck on a dark highway at night, talking on a phone headset, dashboard lights on his "
           "face", "he drives and talks, glancing at the road", ["dad"], seed=7712,
           sfx="truck cab engine rumble, road noise", vol=0.5),
      shot("Close-up of a car radio display glowing orange in a dark car, a hand turning the volume knob up",
           "a hand turns the volume knob up", None, seed=7713, trim=(0.3, 2.4), moves=["Static"],
           sfx="radio volume knob click", vol=0.6),
      shot("in the parked car laughing and crying at the same time, wiping her mascara",
           "she laughs and cries, wiping her eyes", ["jess"], seed=7714, moves=["Push-in Close-up"]),
      shot("a big truck with bright headlights pulling into a dark school parking lot late at night, black sky, lamp posts",
           "a truck pulls into the parking lot, headlights sweeping", None, seed=7715, moves=["Static"],
           sfx="truck air brakes, two long horn blasts", vol=0.85),
      shot("stepping out of the car in her graduation gown, turning toward the headlights, hand over her mouth",
           "she gets out of the car and turns to the headlights, hand over her mouth", ["jess"], seed=7716,
           trim=(0.2, 3.4))],
     [line("host", "You're on the air with Night Lines. Who's this one for?", 0.15, {"calm": 0.7}, "radio"),
      line("dad", "My daughter, Jess. She graduates tonight and I'm... four hundred miles out.", 3.0,
           {"melancholic": 0.4, "calm": 0.3}, "radio"),
      line("dad", "Jess, I'm real proud of you, kid.", 8.4, {"sad": 0.3, "happy": 0.3}, "radio"),
      line("dad", "...Also, I lied about the four hundred miles.", 11.6, {"happy": 0.55}, "radio"),
      line("jess", "Dad?!", 16.4, {"surprised": 0.6, "happy": 0.3}, "room")],
     "quiet parking lot at night",
     {"description": "warm americana slide guitar and soft organ, rising to a hopeful swell",
      "start": 8.0, "volume": 0.3, "fade_in": 1.2, "seed": 7790},
     f"He said he couldn't make it. 🚚\n\nWhat's the best surprise anyone ever gave you?\n\n{TAGS} #Dad #Graduation\nMade with AI")

# 8 ------------------------------------------------------------------------------------------
film(8, "fridge-notes", "Fridge notes",
     {"nadia": {"look": "a fictional woman in her mid thirties with a short black bob and round glasses, an "
                "oversized grey t-shirt, arms crossed", "seed": 7801},
      "tom": {"look": "a fictional man in his mid thirties with messy light brown hair, stubble, a faded band "
              "t-shirt, sheepish", "seed": 7802}},
     {"nadia": {"voice": "a woman in her thirties, dry, quietly hurt, inner voice, close and soft",
                "sample": "Fine. I'm not talking to him."},
      "tom": {"voice": "a man in his thirties, awkward, soft, inner voice, close", "sample": "Okay. She's still mad."}},
     [shot("in a small morning kitchen sticking a yellow sticky note on the fridge door, arms then crossed, "
           "jaw set", "she sticks a note on the fridge and crosses her arms", ["nadia"], seed=7811,
           sfx="morning kitchen, kettle starting to boil", vol=0.45),
      shot("in pajamas in the same small kitchen reading a yellow sticky note on the fridge, rubbing the back of "
           "his neck", "he reads the note and rubs his neck awkwardly", ["tom"], seed=7812),
      shot("Close-up of a hand pressing a yellow sticky note onto a fridge door next to two other notes, morning "
           "light", "a hand presses a new sticky note onto the fridge", None, seed=7813, trim=(0.3, 2.4),
           moves=["Static"], sfx="paper sticky note pressed on a fridge door", vol=0.7),
      shot("reading a new sticky note on the fridge, trying not to smile, failing",
           "she reads the note and tries not to smile, then smiles", ["nadia"], seed=7814, moves=["Push-in Close-up"]),
      shot("both reaching for the same last sticky note on the fridge at the same time, hands touching, looking "
           "at each other", "both reach for the note at once and look at each other", ["nadia", "tom"], seed=7815),
      shot("the couple standing in the small kitchen laughing, sharing one mug of coffee, morning sun",
           "they laugh and pass the coffee mug between them", ["nadia", "tom"], seed=7816, trim=(0.2, 3.4),
           sfx="soft laughter, mug on counter, birds outside", vol=0.4)],
     [line("nadia", "Coffee's made. ...That's all he gets.", 0.15, {"angry": 0.25, "calm": 0.4}, "", "(note) Coffee's made."),
      line("tom", "Thanks. ...Wow. Okay. Still mad.", 3.1, {"sad": 0.25, "calm": 0.4}, "", "(note) Thanks."),
      line("nadia", "I hate that you're right about the thing.", 6.2, {"angry": 0.2, "sad": 0.25}, "", "(note) You were right about the thing."),
      line("tom", "I'm sorry I said it like that.", 9.4, {"sad": 0.35, "calm": 0.3}, "", "(note) Sorry I said it like that."),
      line("nadia", "...Idiot.", 13.6, {"happy": 0.5}, "whisper", "(note) Idiot. ♥")],
     "small apartment kitchen in the morning",
     {"description": "light pizzicato strings and soft piano, quirky and warm, romantic comedy feel",
      "start": 9.2, "volume": 0.3, "fade_in": 1.0, "seed": 7890},
     f"Too proud to talk. Not too proud to write. 📝\n\nHow do you two make up after a fight?\n\n{TAGS} #Couples #Relationships\nMade with AI")

# 9 ------------------------------------------------------------------------------------------
film(9, "stuck", "Stuck",
     {"ines": {"look": "a fictional Portuguese woman in her late twenties with long wavy dark hair, a camel "
               "coat and a large tote bag, impatient", "seed": 7901},
      "ray": {"look": "a fictional older man in his sixties, bald with a white beard, a plain dark grey polo "
              "shirt with no print, holding a small brown parcel", "seed": 7902}},
     {"ines": {"voice": "a woman in her late twenties with a slight Portuguese accent, stressed then warm",
               "sample": "No, no, no. Not today."},
      "ray": {"voice": "a Londoner in his sixties, calm, funny, kind", "sample": "Happens twice a week, love."}},
     [shot("in a small old elevator pressing the button repeatedly, looking at her watch, stressed, harsh "
           "overhead light", "she jabs the elevator button again and checks her watch", ["ines"], seed=7911,
           sfx="elevator jolting to a stop, a clunk, faint hum", vol=0.6),
      shot("leaning calmly against the elevator wall holding a parcel, amused", "he leans back calmly, amused",
           ["ray"], seed=7912),
      shot("the woman and the older delivery man sitting on the floor of the stuck elevator, a parcel between "
           "them, talking", "they sit on the elevator floor and talk, he gestures with his hands",
           ["ines", "ray"], seed=7913, moves=["Static"]),
      shot("Close-up of an elevator button panel with one button glowing, a single bulb flickering",
           "the bulb flickers and then steadies" + EMPTY, None, seed=7917, trim=(0.3, 2.4), moves=["Static"],
           handheld=False,
           sfx="fluorescent light buzz and flicker", vol=0.6),
      shot("sitting on the elevator floor laughing, her head tipped back against the wall",
           "she laughs, head tipped back", ["ines"], seed=7915, moves=["Push-in Close-up"]),
      shot("the elevator doors opening on the two of them getting up from the floor, shaking hands, both smiling",
           "the doors slide open as they stand up and shake hands", ["ines", "ray"], seed=7916, trim=(0.2, 3.4),
           sfx="elevator ding, doors sliding open", vol=0.7)],
     [line("ines", "No, no, no. Not today. I have a meeting.", 0.15, {"angry": 0.3, "afraid": 0.2}, "room"),
      line("ray", "Happens twice a week, love. Sit down.", 3.0, {"calm": 0.6, "happy": 0.2}, "room"),
      line("ray", "So what's the meeting? ...And what would you rather be doing?", 6.2, {"calm": 0.5, "happy": 0.2}, "room"),
      line("ines", "...Painting. I used to paint.", 9.6, {"melancholic": 0.4, "calm": 0.3}, "room"),
      line("ray", "Then you've got two meetings today.", 13.1, {"happy": 0.45, "calm": 0.3}, "room"),
      line("ines", "I'm Ines.", 16.3, {"happy": 0.4}, "room")],
     "inside a small elevator, faint mechanical hum",
     {"description": "minimal warm guitar arpeggio and soft synth pad, gentle, hopeful",
      "start": 9.6, "volume": 0.28, "fade_in": 1.5, "seed": 7990},
     f"Twenty minutes stuck with a stranger. 🛗\n\nHas a stranger ever changed your day?\n\n{TAGS} #Strangers #Kindness\nMade with AI")

# 10 -----------------------------------------------------------------------------------------
film(10, "future-me", "Note to future me",
     {"teen": {"look": "a fictional sixteen year old girl with frizzy auburn hair, braces, a hoodie covered in "
               "pins, sitting on a bedroom floor", "seed": 8001},
      "adult": {"look": "a fictional woman of thirty with auburn hair in a neat low bun, a crisp white shirt, the "
                "same face as the teenager grown up, confident", "seed": 8002}},
     {"teen": {"voice": "a sixteen year old girl, nervous, fast, a bit dramatic, self-conscious laugh",
               "sample": "Okay. Hi. This is weird. Hi, future me."},
      "adult": {"voice": "a woman of thirty, calm, warm, a little moved", "sample": "Hi. I got your message."}},
     [shot("sitting on a messy bedroom floor at night holding a phone up to record herself, embarrassed, posters "
           "and fairy lights behind her", "she records herself, laughs nervously and hides her face", ["teen"],
           seed=8011, sfx="teen bedroom at night, quiet music through a wall", vol=0.35),
      shot("in a bright office stairwell at lunchtime holding a phone to her ear, listening, completely still",
           "she stands still, listening, the corner of her mouth moving", ["adult"], seed=8012,
           sfx="office stairwell, distant lift ding, footsteps above", vol=0.4),
      shot("Close-up of a teenager's hands picking at a chipped pin badge on a hoodie sleeve",
           "nervous fingers pick at the pin badge", None, seed=8013, trim=(0.3, 2.4), moves=["Static"]),
      shot("on the bedroom floor, suddenly serious, looking straight into her phone camera, eyes a little wet",
           "she stops laughing and looks into the camera seriously", ["teen"], seed=8014,
           moves=["Push-in Close-up"]),
      shot("in the office stairwell laughing softly while wiping one eye with her knuckle",
           "she laughs softly and wipes one eye", ["adult"], seed=8015),
      shot("in the office stairwell holding the phone close and recording a reply, smiling",
           "she smiles and records a reply into the phone", ["adult"], seed=8016, trim=(0.2, 3.4))],
     [line("teen", "Okay. Hi. This is so weird. Hi, future me.", 0.15, {"happy": 0.3, "afraid": 0.25}, "voicenote"),
      line("teen", "Are we still scared of everything? Did we ever do the thing?", 3.4,
           {"afraid": 0.3, "calm": 0.3}, "voicenote"),
      line("teen", "Please tell me we're okay.", 8.0, {"sad": 0.35, "afraid": 0.2}, "voicenote"),
      line("adult", "Hi. We're scared of everything.", 12.4, {"happy": 0.35, "calm": 0.35}, "voicenote"),
      line("adult", "We did it anyway.", 15.4, {"happy": 0.4, "calm": 0.4}, "voicenote")],
     "quiet room tone",
     {"description": "delicate glockenspiel and warm piano, nostalgic, gently uplifting",
      "start": 7.4, "volume": 0.3, "fade_in": 1.5, "seed": 8090},
     f"A voice note from 16 to 30. ⏳\n\nWhat would you tell your 16-year-old self?\n\n{TAGS} #FutureMe #GrowingUp\nMade with AI")
print("written", sorted(p.name for p in OUT.glob("*.json")))
