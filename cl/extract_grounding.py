#!/usr/bin/env python3
"""
Extract grounding objects from LongVideoBench questions using spaCy NLP.

For each question, extracts nouns/noun-phrases as target_objects and
scene-related words as cue_objects. This is a lightweight alternative
to GPT-4o grounding — sufficient for the CL course project where the
focus is on λ adaptation, not grounding quality.

Usage:
    python cl/extract_grounding.py \
        --input_json /l/users/jianxiang.he/continual-vsi/data/LongVideoBench/lvb_val.json \
        --output_json /l/users/jianxiang.he/continual-vsi/data/LongVideoBench/lvb_val_grounded.json
"""
import argparse
import json
import re
from typing import Dict, List


SCENE_WORDS = {
    "room", "street", "road", "building", "house", "kitchen", "office",
    "park", "garden", "field", "stage", "studio", "court", "gym",
    "restaurant", "store", "shop", "market", "school", "hospital",
    "beach", "mountain", "forest", "lake", "river", "ocean", "sea",
    "sky", "ground", "floor", "wall", "door", "window", "table", "desk",
    "chair", "bed", "sofa", "car", "bus", "train", "airplane", "boat",
    "city", "town", "village", "country", "bridge", "tunnel", "station",
    "airport", "classroom", "library", "bathroom", "bedroom", "hallway",
    "backyard", "driveway", "garage", "parking",
}

STOP_OBJECTS = {
    "video", "scene", "frame", "clip", "moment", "time", "part",
    "beginning", "end", "middle", "start", "way", "thing", "something",
    "anything", "everything", "nothing", "one", "ones", "kind", "type",
    "order", "number", "question", "answer", "option", "choice",
    "subtitle", "text", "word", "sentence", "caption",
}


def extract_noun_phrases(text: str) -> List[str]:
    """Extract noun phrases using simple regex patterns."""
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text_clean.split()

    # Simple noun phrase extraction: adjective? + noun sequences
    nouns = []
    for w in words:
        if len(w) > 2 and w not in STOP_OBJECTS and w not in {
            "the", "a", "an", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might",
            "can", "shall", "must", "need",
            "not", "no", "yes", "and", "or", "but", "if", "then",
            "when", "where", "how", "what", "which", "who", "whom",
            "very", "more", "most", "much", "many", "some", "any",
            "all", "each", "every", "both", "few", "several",
            "first", "second", "third", "last", "next",
            "same", "different", "other", "another",
            "with", "without", "from", "into", "onto", "upon",
            "about", "after", "before", "during", "between",
            "through", "above", "below", "under", "over",
            "here", "there", "also", "just", "only", "still",
            "already", "again", "too", "so", "than",
        }:
            nouns.append(w)

    return nouns


def extract_grounding(question: str, candidates: list = None) -> Dict:
    """Extract target and cue objects from a question."""
    all_text = question
    if candidates:
        all_text += " " + " ".join(str(c) for c in candidates)

    nouns = extract_noun_phrases(all_text)

    target_objects = []
    cue_objects = []
    seen = set()

    for noun in nouns:
        if noun in seen:
            continue
        seen.add(noun)
        if noun in SCENE_WORDS:
            cue_objects.append(noun)
        else:
            target_objects.append(noun)

    # Keep top objects (YOLO-World works best with 3-8 objects)
    target_objects = target_objects[:8]
    cue_objects = cue_objects[:4]

    # Ensure at least one target
    if not target_objects:
        target_objects = ["object"]

    return {
        "target_objects": target_objects,
        "cue_objects": cue_objects,
        "relations": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    for item in data:
        candidates = item.get("candidates", item.get("options", []))
        item["grounding_objects"] = extract_grounding(
            item["question"], candidates
        )

    with open(args.output_json, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(data)} entries → {args.output_json}")

    # Show a few examples
    for item in data[:3]:
        print(f"\n  Q: {item['question'][:80]}...")
        print(f"  targets: {item['grounding_objects']['target_objects']}")
        print(f"  cues: {item['grounding_objects']['cue_objects']}")


if __name__ == "__main__":
    main()
