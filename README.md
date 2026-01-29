# 2025 - The 45x45 Word Grouping Puzzle

A massive word-grouping puzzle game where you must identify 45 groups of 45 related items by combining them two at a time.

## The Game

**2025** presents you with a 45x45 grid (2,025 cells) containing words from 45 different categories. Your goal is to find all 45 groups by matching pairs of words that belong to the same category.

### How to Play

1. **Click** a word to select it (it will become inverted)
2. **Click** another word:
   - If both words are from the **same category**, they merge into a single cell showing both items
   - If they're from **different categories**, both shake and you get a mistake
3. **Continue merging** until each category has all 45 of its items combined
4. **Win** when you reach a score of 1,980 (44 merges × 45 categories)

### Controls

- Click a selected word again to deselect it
- Use the **Deselect** button to clear your selection
- Your progress is automatically saved to your browser's local storage

## Files

| File | Description |
|------|-------------|
| `2025.html` | Original game by Thomas Colthurst with hand-curated categories |
| `generate_game.py` | Python tool to generate new games from Wikipedia categories |
| `game_*.html` | Generated game instances |
| `category_cache.json` | Cache of discovered Wikipedia categories (auto-generated) |

## Generating New Games

The `generate_game.py` script creates new game instances by fetching categories from Wikipedia.

### Requirements

- Python 3.10+
- `requests` library

```bash
pip install requests
```

### Usage

```bash
# Generate a new game (creates game_YYYY-MM-DD_HHMMSS.html)
python generate_game.py

# Specify output filename
python generate_game.py -o my_game.html

# Generate a reproducible game with a specific seed
python generate_game.py --seed 12345 -o my_game.html

# Show detailed progress
python generate_game.py -v

# Expand the category cache (finds 50 new categories, doesn't generate a game)
python generate_game.py --expand-cache 50

# Use a different cache file
python generate_game.py -c my_cache.json
```

### Reproducibility

Each generated game records its random seed in an HTML comment:
```html
<!-- Generated with seed: 12345 -->
```

Use `--seed` with the same value to regenerate the exact same game.

### How It Works

1. **Seed Categories**: Starts with a curated list of promising Wikipedia category types (animals, foods, arts, sciences, etc.)
2. **Exploration**: Fetches subcategories from Wikipedia's API
3. **Filtering**: Rejects unsuitable categories:
   - Administrative/maintenance categories
   - Too broad (500+ members) or too narrow (<45 members)
   - Date-based or demographic categories
   - Categories about single people/entities
4. **Member Selection**:
   - Filters out entries that are just the category name (e.g., "Board game" in "Board games")
   - Spreads selection across the alphabet (not just A, B, C...)
   - Fetches up to 300 members and samples 45 distributed across all letters
5. **Duplicate Prevention**: Each word appears in exactly one category - if a word exists in multiple Wikipedia categories, it's only included in the first one selected
6. **Diversity**: Ensures variety by rejecting categories with overlapping keywords (e.g., won't include both "Mammals of Africa" and "Mammals of Asia" in the same game)
7. **Caching**: Discovered categories are cached locally for faster subsequent runs
8. **Generation**: Creates a standalone HTML file with the selected categories

### Example Categories

Generated games might include categories like:
- Shark genera, Renaissance composers, Telescopes
- Hairstyles, Phobias, Board games
- Architectural styles, Puzzle video games, Trains
- Edible fruits, Documentary films, Herbal teas

## Original Categories (2025.html)

The original game includes 45 hand-curated categories:

Elements, African Countries, Muppets, US State Capitals, Prepositions, Types of Numbers, Memes, Mammals, Musical Instruments, Dog Breeds, Olympic Sports, D&D Monsters, Tolkien Characters, Cartoonists, US Vice Presidents, SNL Cast Members, World Wonders, College Majors, Car Brands, Large Companies, Collective Nouns, Programming Languages, Cheeses, Pastas, MCU Characters, Rock Bands, Birds, Tom Hanks Movies, Famous Novels, Trees, Fallacies, Google Products, Gemstones, Books of the Bible, Beatles Songs, Cocktails, Flowers, Fabrics, Sci-Fi TV Shows, Keyboard Characters, Weather Words, Inventions, Musicals, Poets, Legal Doctrines

## Credits

- **Original Game**: [2025](https://thomaswc.com/2025.html) by Thomas Colthurst
- **License**: [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Wikipedia Generator**: Creates games using data from [Wikipedia](https://en.wikipedia.org/) via their [API](https://www.mediawiki.org/wiki/API:Main_page)

## Tips for Playing

- Start by looking for obvious groupings (scientific terms, place names, proper nouns)
- When you find one member of a category, scan for similar-looking items
- Merged cells show the category count in red - use this to track progress
- The tooltip on merged cells shows all items in that group so far
- Categories with technical or domain-specific vocabulary are often easier to identify
