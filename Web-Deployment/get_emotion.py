def get_group_name(valence, arousal, dominance, liking):
    """
    Returns the group name based on binary values for valence, arousal, dominance, and liking.
    """
    valence_label = "Hv" if valence == 1 else "Lv"
    arousal_label = "Ha" if arousal == 1 else "La"
    dominance_label = "Hd" if dominance == 1 else "Ld"
    liking_label = "Hl" if liking == 1 else "Ll"

    return f"{valence_label}{arousal_label}{dominance_label}{liking_label}"

def get_emotions_from_group(group_name):
    """
    Returns the emotional description for a given group name.

    Args:
        group_name (str): The group name in the format "HvLaHdLl".

    Returns:
        str: Description of the emotions associated with the group.
    """
    # Mapping of group names to emotions
    group_emotions = {
        "LvLaLdLl": "Apathetic, Disconnected",
        "LvLaLdHl": "Indifferent but Favorable",
        "LvLaHdLl": "Submissive, Defeated",
        "LvLaHdHl": "Sympathetic, Mildly Caring",
        "LvHaLdLl": "Agitated, Stressed",
        "LvHaLdHl": "Frustrated but Favorable",
        "LvHaHdLl": "Angry, Overwhelmed",
        "LvHaHdHl": "Determined but Critical",
        "HvLaLdLl": "Calm but Detached",
        "HvLaLdHl": "Relaxed and Affectionate",
        "HvLaHdLl": "Confident but Reserved",
        "HvLaHdHl": "Content, Grateful",
        "HvHaLdLl": "Excited but Unconnected",
        "HvHaLdHl": "Energetic and Loving",
        "HvHaHdLl": "Passionate, Assertive",
        "HvHaHdHl": "Joyful, Enthusiastic, Empowered"
    }

    # Fetch and return the emotion description
    emotion_description = group_emotions.get(group_name, "Unknown Group Name")
    return emotion_description