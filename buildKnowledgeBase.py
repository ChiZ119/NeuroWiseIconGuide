import json
from pathlib import Path

crawlResult = {
    "perfect_matches": {
        "Wash hands": {
        "url": "https://www.wikihow.com/Wash-Hand-Wraps",
        "steps": [
      {
        "step": "Seal the hook-and-loop fastener to itself before you wash your wraps.",
        "image": "Wrap dress"
      },
      {
        "step": "Place your hand wraps inside of a mesh laundry bag.",
        "image": "Laundry Bag"
      },
      {
        "step": "Launder your wraps with your other clothes if they are a neutral color.",
        "image": "Wrap dress"
      },
      {
        "step": "Wash your wraps by themselves if they are brightly colored.",
        "image": "Wrap dress"
      },
      {
        "step": "Pour your detergent into its appropriate slot.",
        "image": "Laundry Detergent"
      },
      {
        "step": "Wash the wraps on a delicate cycle with cold water.",
        "image": "Wrap dress"
      },
      {
        "step": "Take the wraps out of the laundry bag when the cycle is complete.",
        "image": "Laundry Bag"
      },
      {
        "step": "Avoid putting your wraps in your dryer.",
        "image": "Wrap dress"
      },
      {
        "step": "Drape each wrap across the bottom of a hanger.",
        "image": "Clothes Hanger"
      },
      {
        "step": "Let your wraps air-dry for 1-3 hours.",
        "image": "Wrap dress"
      }
    ]
  },

        "Make the bed": {
            "url": "https://www.wikihow.com/Make-a-Bed-Neatly",
            "steps": [
      {
        "step": "Place a bed skirt between the box spring and mattress to hide the frame.",
        "image": "Open Shipping Box"
      },
      {
        "step": "Place a mattress topper and pad on the mattress for added comfort.",
        "image": "Mattress"
      },
      {
        "step": "Stretch a fitted sheet over the 4 corners of the mattress.",
        "image": "Mattress"
      },
      {
        "step": "Tuck the edges of a flat sheet underneath the mattress.",
        "image": "Mattress"
      },
      {
        "step": "Drape a blanket over the flat sheet, then fold back the top.",
        "image": "Blanket"
      },
      {
        "step": "Place a duvet or comforter on top of the blanket for a luxurious feel.",
        "image": "Blanket"
      },
      {
        "step": "Set throw pillows and a throw blanket on the bed to make it more inviting.",
        "image": "Pillow"
      }
    ]
  },
        "Tie shoelaces": {
            "url": "https://www.wikihow.com/Hide-Shoelaces",
            "steps": [
      {
        "step": "Put the ends of the laces into the holes closest to your toe.",
        "image": "Ankle Boots"
      },
      {
        "step": "Make your first \"bar\" by bringing the left shoelace up and across to the right.",
        "image": "Shoelace"
      },
      {
        "step": "Make your second \"bar\" by bringing the right shoelace up and across to the left.",
        "image": "Shoelace"
      },
      {
        "step": "Continue making \"bars\" up the front of the shoe.",
        "image": "Light-Up Sneakers"
      },
      {
        "step": "Tie a bow on the inside of the shoe.",
        "image": "Bow"
      },
      {
        "step": "Start threading your laces at the end closest to your toe.",
        "image": "Ankle Boots"
      },
      {
        "step": "Make criss-cross lacing by crossing the right lace over the left lace.",
        "image": "Mirrored Nail Wraps"
      },
      {
        "step": "Have the laces going into the shoe at the top.",
        "image": "Ankle Boots"
      },
      {
        "step": "Cross right over left, then left over right totie a square knotin the laces.",
        "image": "Tie"
      },
      {
        "step": "Tuck the knot all the way inside your shoe.",
        "image": "Tie"
      }
    ]
  },
  "Do laundry": {
    "url": "https://www.wikihow.com/Use-Bleach-when-Doing-Your-Laundry",
    "steps": [
      {
        "step": "Choose a liquid bleach with polymers to avoid yellow marks.",
        "image": "Bleach"
      },
      {
        "step": "Set your washing machine to the hottest setting.",
        "image": "Washing Machine"
      },
      {
        "step": "Add your clothes and washing powder to the machine.",
        "image": "Washing Machine"
      },
      {
        "step": "Place the bleach in the bleach dispenser if your machine has one.",
        "image": "Bleach"
      },
      {
        "step": "Add 1 capful of bleach into the machine once it is filled with water.",
        "image": "Bleach"
      },
      {
        "step": "Fill a bucket or sink with 1 gallon (3.8 L) of hot water.",
        "image": "Hot Water Bottle"
      },
      {
        "step": "Mix 1 tablespoon (15 mL) of oxygen or chlorine bleach into the water.",
        "image": "Bleach"
      },
      {
        "step": "Add your usual washing detergent to treat stained items.",
        "image": "Laundry Detergent"
      },
      {
        "step": "Soak your laundry for up to 1 hour.",
        "image": "Laundry Detergent"
      },
      {
        "step": "Rinse the items under cold water.",
        "image": "Cleaning Gloves"
      }
    ]
  },
  "Set the table": {
    "url": "https://www.wikihow.com/Proper-Table-Setting",
    "steps": [
      {
        "step": "Set the plate in the center, the fork on the left, and the knife and spoon on the right.",
        "image": "Fork"
      },
      {
        "step": "Stack a salad plate and a soup bowl on the dinner plate, and set out extra utensils.",
        "image": "Cooking Utensil Set"
      },
      {
        "step": "Put a charger under the plate and set out a bread plate and dessert utensils.",
        "image": "Plate"
      },
      {
        "step": "Set plates on the buffet station and utensils and napkins at place settings.",
        "image": "Plate"
      },
      {
        "step": "Spread out a fabric tablecloth and fold fabric napkins for formal meals.",
        "image": "Folded Napkin"
      },
      {
        "step": "Set out a centerpiece and position it so that guests can see one another.",
        "image": "Hotel Lobby"
      }
    ]
  },
  "Use a glue stick": {
    "url": "https://www.wikihow.com/Melt-Glue-Sticks",
    "steps": [
      {
        "step": "Remove the glue sticks from their original packaging.",
        "image": "Glue Stick"
      },
      {
        "step": "Put the stick(s) into a microwave-safe dish.",
        "image": "Preztel Stick"
      },
      {
        "step": "Microwave the glue on full power in 20-second intervals.",
        "image": "Hot Glue Gun"
      },
      {
        "step": "Remove the glue from the microwave carefully.",
        "image": "Microwave"
      },
      {
        "step": "Obtain aglue gunat least as big as the glue sticks.",
        "image": "Glue Gun"
      },
      {
        "step": "Plug in the glue gun over cardboard or newspaper.",
        "image": "Glue Gun"
      },
      {
        "step": "Squeeze the trigger of the glue gun to feed the sticks.",
        "image": "Glue Stick"
      },
      {
        "step": "Insert another stick once the first no longer feeds through.",
        "image": "Streaming Stick"
      }
    ]
  },
  "Use scissors to cut on paper": {
    "url": "https://www.wikihow.com/Play-Rock,-Paper,-Scissors",
    "steps": [
      {
        "step": "Rock beats scissors and loses to paper.",
        "image": "Papercutting Knife"
      },
      {
        "step": "Paper beats rock, but loses to scissors.",
        "image": "Origami Paper"
      },
      {
        "step": "Scissors beat paper but loses to rock.",
        "image": "Papercutting Knife"
      },
      {
        "step": "Face the other player and hold your hands up at your stomach.",
        "image": "Handball player"
      },
      {
        "step": "Count “rock, paper, scissors” by tapping your hands together.",
        "image": "Children's Scissors"
      },
      {
        "step": "Play rock, paper, or scissors on the final count and at the same time.",
        "image": "Rock climbing"
      },
      {
        "step": "Determine the winner.",
        "image": "Winner"
      },
      {
        "step": "Use a quick game to settle unimportant disputes.",
        "image": "Puzzle Solving"
      },
      {
        "step": "Play for fun or to kill the time when you’re bored.",
        "image": "Juggling"
      },
      {
        "step": "Play competitively at an organized tournament if you’re passionate.",
        "image": "Tournament"
      }
    ]
  },
  "Put on socks": {
    "url": "https://www.wikihow.com/Do-a-Sock-Bun",
    "steps": [
      {
        "step": "Select a sock.",
        "image": "Celestial Sock"
      },
      {
        "step": "Cut off the end.",
        "image": "Life-Size Cutout"
      },
      {
        "step": "Put your hair in aponytail.",
        "image": "Faux Hawk Hairstyle"
      },
      {
        "step": "Pull the sock to the ends of your hair.",
        "image": "Celestial Sock"
      },
      {
        "step": "Roll the sock down.",
        "image": "Socks"
      },
      {
        "step": "Finish the bun.",
        "image": "Squid Ink Bun"
      },
      {
        "step": "Finished.",
        "image": "Finish line"
      }
    ]
  }
    },
    "manual_review_list": [
        {
            "query": "Wash face",
            "url": "https://www.wikihow.com/Wash-Your-Face",
            "approved": False,   # set True once reviewed, for this case 我觉得Deodorant不是保湿霜
            "steps": [
                {"step": "Wet your face with lukewarm water.", "image": "Faucet"},
                {"step": "Apply a mild cleanser in circular motions.", "image": "Washing Face"},
                {"step": "Rinse thoroughly and pat dry gently.", "image": "Washing Face"},
                {"step": "Apply moisturizer to lock in hydration.", "image": "Deodorant"},
            ]
        },
        {
            "query": "Sweep the floor",
            "url": "https://www.wikihow.com/Sweep-a-Floor",
            "approved": False, # 只有簸箕没有垃圾桶，只有房间没有障碍物。。。
            "steps": [
                {"step": "Clear the area of obstructions.", "image": "Room"},
                {"step": "Use short deliberate strokes with the broom.", "image": "Broom"},
                {"step": "Work from edges toward the center.", "image": "Room"},
                {"step": "Sweep the pile into a dustpan.", "image": "Dustpan"},
                {"step": "Empty the dustpan into a trash bin.", "image": "Dustpan"},
            ]
        }
    ],
    "failed_queries": ["Brush teeth", "Pack the schoolbag", "Take a shower"],
}

manualFallback = {
    "Brush teeth": [
        {"step": "Get your toothbrush and toothpaste.", "image": "Toothbrush"},
        {"step": "Put a small amount of toothpaste on the toothbrush.", "image": "Toothpaste"},
        {"step": "Wet the toothbrush with water.", "image": "Faucet"},
        {"step": "Brush all surfaces of your teeth for 2 minutes.", "image": "Brushing Teeth"},
        {"step": "Spit out the toothpaste.", "image": "Sink"},
        {"step": "Rinse your mouth with water.", "image": "Cup of Water"},
    ],
    "Pack the schoolbag": [
        {"step": "Check your timetable to see which books you need.", "image": "Timetable"},
        {"step": "Place your textbooks and notebooks in the bag.", "image": "Backpack"},
        {"step": "Add your pencil case.", "image": "Pencil Case"},
        {"step": "Pack your water bottle.", "image": "Water Bottle"},
        {"step": "Close all zippers.", "image": "Backpack"},
    ],
    "Take a shower": [
        {"step": "Turn on the shower and adjust the water temperature.", "image": "Shower"},
        {"step": "Wet your hair and body.", "image": "Shower"},
        {"step": "Apply shampoo and wash your hair.", "image": "Shampoo"},
        {"step": "Rinse out the shampoo.", "image": "Shower"},
        {"step": "Apply body wash and scrub your body.", "image": "Soap"},
        {"step": "Rinse off the soap.", "image": "Shower"},
        {"step": "Turn off the shower and dry yourself with a towel.", "image": "Towel"},
    ],
}


def build_knowledge_base(
    crawl_results: dict,
    manual_fallback: dict,
    output_path: str = "knowledgeBase.json"
) -> dict:
    kb = {}

    # 1. Perfect matches, add directly
    for query, data in crawl_results["perfect_matches"].items():
        kb[query.lower()] = {
            "query": query,
            "source": data["url"],
            "origin": "crawler_perfect",
            "steps": data["steps"],
        }
        print(f"Added (perfect): {query} — {len(data['steps'])} steps")

    # 2. Approved manual review items
    for item in crawl_results["manual_review_list"]:
        if item.get("approved"):
            key = item["query"].lower()
            kb[key] = {
                "query": item["query"],
                "source": item["url"],
                "origin": "crawler_reviewed",
                "steps": item["steps"],
            }
            print(f"Added (reviewed): {item['query']} — {len(item['steps'])} steps")
        else:
            print(f"Skipped (not approved): {item['query']}")

    # 3. Manual fallbacks for failed queries
    for query, steps in manual_fallback.items():
        key = query.lower()
        if key not in kb:
            kb[key] = {
                "query": query,
                "source": "manual",
                "origin": "manual_fallback",
                "steps": steps,
            }
            print(f"Added (manual): {query} — {len(steps)} steps")

    # 4. Save
    Path(output_path).write_text(
        json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nKnowledge base saved in {output_path}  ({len(kb)} entries)")
    return kb


if __name__ == "__main__":
    build_knowledge_base(crawlResult, manualFallback)
