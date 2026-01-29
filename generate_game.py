#!/usr/bin/env python3
"""
Wikipedia Category Game Generator

Generates instances of the "2025" word-grouping puzzle game by fetching
45 Wikipedia categories (each with at least 45 members).

Usage:
    python generate_game.py [--output FILE] [--cache FILE] [--verbose]
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Optional

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import requests
except ImportError:
    print("Error: 'requests' library is required. Install with: pip install requests")
    exit(1)


class WikipediaAPI:
    """Handles Wikipedia API calls with rate limiting."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikiCategoryGameGenerator/1.0 (Educational puzzle game)'
        })
        self.backoff_until = 0  # Timestamp for rate limit backoff

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        # Check if we're in backoff period
        now = time.time()
        if now < self.backoff_until:
            wait_time = self.backoff_until - now
            print(f"  Rate limited, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

        elapsed = now - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, params: dict, retries: int = 3) -> Optional[dict]:
        """Make a rate-limited API request with retry logic."""
        self._rate_limit()
        params['format'] = 'json'

        for attempt in range(retries):
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=30)

                if response.status_code == 429:
                    # Rate limited - back off exponentially
                    wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                    print(f"  Rate limited (429), backing off {wait_time}s...")
                    self.backoff_until = time.time() + wait_time
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 2
                    print(f"  API error (attempt {attempt+1}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  API error: {e}")
                    return None
        return None

    def get_category_members(self, category: str, limit: int = 100) -> list[str]:
        """Get page members of a category (not subcategories)."""
        members = []
        continue_token = None

        while len(members) < limit:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmtype': 'page',
                'cmlimit': min(50, limit - len(members)),
            }
            if continue_token:
                params['cmcontinue'] = continue_token

            data = self._make_request(params)
            if not data or 'query' not in data:
                break

            for item in data['query'].get('categorymembers', []):
                members.append(item['title'])

            if 'continue' in data:
                continue_token = data['continue'].get('cmcontinue')
            else:
                break

        return members[:limit]

    def get_subcategories(self, category: str, limit: int = 50) -> list[str]:
        """Get subcategories of a category."""
        subcats = []
        continue_token = None

        while len(subcats) < limit:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmtype': 'subcat',
                'cmlimit': min(50, limit - len(subcats)),
            }
            if continue_token:
                params['cmcontinue'] = continue_token

            data = self._make_request(params)
            if not data or 'query' not in data:
                break

            for item in data['query'].get('categorymembers', []):
                # Remove "Category:" prefix
                name = item['title']
                if name.startswith('Category:'):
                    name = name[9:]
                subcats.append(name)

            if 'continue' in data:
                continue_token = data['continue'].get('cmcontinue')
            else:
                break

        return subcats[:limit]

    def get_category_info(self, category: str) -> dict:
        """Get info about a category including member count."""
        params = {
            'action': 'query',
            'titles': f'Category:{category}',
            'prop': 'categoryinfo',
        }

        data = self._make_request(params)
        if not data or 'query' not in data:
            return {}

        pages = data['query'].get('pages', {})
        for page in pages.values():
            return page.get('categoryinfo', {})
        return {}


class CategoryFilter:
    """Determines if a category is suitable for the game."""

    # Words that indicate maintenance/administrative categories
    REJECT_PATTERNS = [
        r'\bwikipedia\b', r'\barticles\b', r'\bstubs?\b', r'\bpages\b',
        r'\bcs1\b', r'\bwikidata\b', r'\bneeding\b', r'\blocking\b',
        r'\bwithout\b', r'\bdisambiguation\b', r'\btemplates?\b',
        r'\bwebarchive\b', r'\bshort description\b', r'\binfobox\b',
        r'\bredirects?\b', r'\bduplicate\b', r'\borphan', r'\bdead links?\b',
        r'\bmissing\b', r'\buncategorized\b', r'\bunreferenced\b',
        r'\ball-stub\b', r'\bgeo-stub\b', r'\bbio-stub\b',
        r'\bby year\b', r'\bby country\b', r'\bby nationality\b',
        r'\bby decade\b', r'\bby century\b', r'\bby language\b',
        r'\bbirths\b', r'\bdeaths\b', r'\bliving people\b',
        r'^\d{4}s?\s', r'\s\d{4}s?$',  # Years at start/end
        r'\bestablishments\b', r'\bdisestablishments\b',
        # Categories about specific people/entities (too narrow)
        r'^works by\b', r'^cultural depictions of\b',
        r'\bpeople$', r'^.{1,30} people$',  # "[Nationality] people" but allow longer compound names
        # Sensitive demographic categories (not fun for a game)
        r'\bblind\b', r'\bdeaf\b', r'\bdisabled\b', r'\bdisabilit', r'\bdisorders?\b',
        r'\bjewish\b', r'\bmuslim\b', r'\bchristian\b', r'\bhindu\b', r'\bbuddhist\b',
        r'\bgay\b', r'\blesbian\b', r'\blgbt', r'\btransgender\b', r'\bqueer\b',
        r'\bmale\b', r'\bfemale\b', r'\bwomen\b', r'\bmen\b',
        r'\bafrican.american\b', r'\basian.american\b', r'\bhispanic\b',
        r'\bwhite\b', r'\bblack\b',
        # Meta categories (about other categories/lists)
        r'\bfilmograph', r'\bdiscograph', r'\bbibliograph',
        r'\blists? of\b', r'\bawards? received\b',
    ]

    # Patterns that suggest a category is about a single person/entity (too narrow)
    SINGLE_ENTITY_PATTERNS = [
        r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # "First Last" - likely a person's name
        r'^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+$',  # "First M. Last"
        r'^The [A-Z]',  # "The Something" - often a band/show
        r' songs$',  # Songs by a specific artist
        r' albums$',  # Albums by a specific artist
        r' discography$',
        r' filmography$',
        r' characters$',  # Characters from a specific work
        r'^Deeds of ',  # Greek mythology specific deeds
        r'^Works of ',
        r'^Films of ',
        r'^Paintings of ',
        r'^Sculptures of ',
        r'^Cultural depictions of ',
    ]

    def __init__(self, min_members: int = 45, max_members: int = 500):
        self.min_members = min_members
        self.max_members = max_members
        self.reject_re = re.compile('|'.join(self.REJECT_PATTERNS), re.IGNORECASE)
        self.single_entity_re = re.compile('|'.join(self.SINGLE_ENTITY_PATTERNS))

    def is_name_acceptable(self, name: str) -> bool:
        """Check if category name passes basic filters."""
        if self.reject_re.search(name):
            return False

        # Reject if it's mostly numbers
        alpha_chars = sum(1 for c in name if c.isalpha())
        if alpha_chars < len(name) * 0.5:
            return False

        # Reject categories that look like they're about a single person/entity
        if self.single_entity_re.match(name):
            return False

        return True

    def are_members_acceptable(self, members: list[str]) -> bool:
        """Check if the member list is suitable."""
        if len(members) < self.min_members:
            return False

        # Check how many members contain years (indicates date-based category)
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        year_count = sum(1 for m in members if year_pattern.search(m))
        if year_count > len(members) * 0.3:
            return False

        # Check for too many disambiguation pages
        disambig_count = sum(1 for m in members if '(disambiguation)' in m.lower())
        if disambig_count > len(members) * 0.2:
            return False

        return True

    def clean_member_name(self, name: str) -> str:
        """Clean up a member name for display in the game."""
        # Remove common disambiguation suffixes
        name = re.sub(r'\s*\([^)]*\)\s*$', '', name)
        # Remove "List of " prefix if present
        name = re.sub(r'^List of\s+', '', name, flags=re.IGNORECASE)
        return name.strip()

    def is_member_too_similar_to_category(self, member: str, category: str) -> bool:
        """Check if a member name is just a variation of the category name."""
        member_lower = member.lower().strip()
        category_lower = category.lower().strip()

        # Normalize: remove common suffixes/prefixes and pluralization
        def normalize(s):
            s = re.sub(r'^(the|a|an)\s+', '', s)
            s = re.sub(r'\s+(of|and|the)$', '', s)
            # Simple depluralization
            if s.endswith('ies'):
                s = s[:-3] + 'y'
            elif s.endswith('es'):
                s = s[:-2]
            elif s.endswith('s') and not s.endswith('ss'):
                s = s[:-1]
            return s

        norm_member = normalize(member_lower)
        norm_category = normalize(category_lower)

        # Exact match after normalization
        if norm_member == norm_category:
            return True

        # Member is substring of category or vice versa (with some length)
        if len(norm_member) > 4 and len(norm_category) > 4:
            if norm_member in norm_category or norm_category in norm_member:
                return True

        # Check word overlap - if most words match, it's too similar
        member_words = set(norm_member.split())
        category_words = set(norm_category.split())
        if member_words and category_words:
            overlap = member_words & category_words
            # If more than 60% of the smaller set overlaps, too similar
            min_len = min(len(member_words), len(category_words))
            if min_len > 0 and len(overlap) / min_len > 0.6:
                return True

        return False

    def select_alphabetically_spread(self, members: list[str], count: int = 45) -> list[str]:
        """Select members spread across the alphabet rather than bunched at the start."""
        if len(members) <= count:
            return members

        # Group by first letter
        by_letter = {}
        for m in members:
            first = m[0].upper() if m else '?'
            if first not in by_letter:
                by_letter[first] = []
            by_letter[first].append(m)

        # Calculate how many to take from each letter
        letters = sorted(by_letter.keys())
        result = []

        # First pass: take proportionally from each letter
        remaining = count
        for letter in letters:
            available = by_letter[letter]
            # Take proportional amount, at least 1 if available
            proportion = len(available) / len(members)
            take = max(1, int(proportion * count))
            take = min(take, len(available), remaining)

            # Randomly sample from this letter's entries
            selected = random.sample(available, take)
            result.extend(selected)
            remaining -= take

            if remaining <= 0:
                break

        # If we still need more, randomly fill from remaining
        if len(result) < count:
            used = set(result)
            remaining_members = [m for m in members if m not in used]
            needed = count - len(result)
            if remaining_members:
                result.extend(random.sample(remaining_members, min(needed, len(remaining_members))))

        # Shuffle the final result so it's not strictly alphabetical
        random.shuffle(result)
        return result[:count]


class CacheManager:
    """Manages caching of discovered good categories."""

    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {'categories': {}, 'rejected': [], 'last_updated': None}
        return {'categories': {}, 'rejected': [], 'last_updated': None}

    def save(self):
        """Save cache to file."""
        self.cache['last_updated'] = datetime.now().isoformat()
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def get_category(self, name: str) -> Optional[list[str]]:
        """Get cached members for a category."""
        return self.cache['categories'].get(name)

    def add_category(self, name: str, members: list[str]):
        """Add a good category to cache."""
        self.cache['categories'][name] = members

    def is_rejected(self, name: str) -> bool:
        """Check if category was previously rejected."""
        return name in self.cache.get('rejected', [])

    def add_rejected(self, name: str):
        """Mark a category as rejected."""
        if 'rejected' not in self.cache:
            self.cache['rejected'] = []
        if name not in self.cache['rejected']:
            self.cache['rejected'].append(name)

    def get_all_good_categories(self) -> dict:
        """Get all cached good categories."""
        return self.cache.get('categories', {})


class HTMLGenerator:
    """Generates the game HTML file."""

    def __init__(self, template_file: str = '2025.html'):
        self.template_file = template_file

    def generate(self, categories: dict[str, list[str]], output_file: str, seed: int = None):
        """Generate game HTML with the given categories."""

        # Read template
        with open(self.template_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Build the categories JavaScript
        cats_js = self._build_categories_js(categories)

        # Find and replace the cats object in the template
        # Look for "var cats = {};" and everything until the next major section
        pattern = r'(var cats = \{\};.*?)(// Functions)'
        replacement = f'var cats = {{}};\n\n{cats_js}\n\n            \\2'

        html = re.sub(pattern, replacement, template, flags=re.DOTALL)

        # Update title to indicate this is a generated version
        html = html.replace('<title>2025</title>', '<title>2025 - Wikipedia Edition</title>')

        # Add seed comment after the opening <html> tag
        if seed is not None:
            seed_comment = f'\n    <!-- Generated with seed: {seed} -->'
            html = html.replace('<html>', f'<html>{seed_comment}', 1)

        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\nGenerated: {output_file}")

    def _build_categories_js(self, categories: dict[str, list[str]]) -> str:
        """Build JavaScript code for the categories object."""
        lines = []
        for i, (name, members) in enumerate(categories.items(), 1):
            # Escape quotes in names and members
            safe_name = name.replace('"', '\\"').replace("'", "\\'")
            safe_members = [m.replace('"', '\\"').replace("'", "\\'") for m in members[:45]]

            members_str = ', '.join(f'"{m}"' for m in safe_members)
            lines.append(f'            // #{i}')
            lines.append(f'            cats["{safe_name}"] = [{members_str}];')
            lines.append('')

        return '\n'.join(lines)


class DiversityChecker:
    """Ensures selected categories are diverse and not too similar."""

    def __init__(self):
        # Track word stems used in category names
        self.used_keywords = {}  # keyword -> count
        # Track exact category names to avoid near-duplicates
        self.used_names = set()

    def _extract_keywords(self, name: str) -> set[str]:
        """Extract significant keywords from a category name."""
        # Common words to ignore
        stopwords = {
            'the', 'a', 'an', 'of', 'in', 'by', 'for', 'and', 'or', 'with',
            'from', 'to', 'at', 'on', 'as', 'is', 'are', 'was', 'were',
            'programming', 'language', 'languages', 'software', 'free',
            'types', 'type', 'techniques', 'technique', 'materials',
            'theory', 'theories', 'objects', 'three',
        }

        words = re.findall(r'[a-z]+', name.lower())
        return {w for w in words if len(w) > 3 and w not in stopwords}

    def _get_primary_keyword(self, name: str) -> str:
        """Get the main topic word from a category name."""
        keywords = self._extract_keywords(name)
        # Return the longest keyword as the "primary" one
        if keywords:
            return max(keywords, key=len)
        return ""

    def is_diverse_enough(self, name: str, max_keyword_overlap: int = 1) -> bool:
        """Check if a category is different enough from already selected ones."""
        keywords = self._extract_keywords(name)

        if not keywords:
            return True

        # Strict check: if the primary keyword has been used already, reject
        primary = self._get_primary_keyword(name)
        if primary and self.used_keywords.get(primary, 0) >= 1:
            return False

        # Count how many times each keyword has been used
        overlap_count = sum(1 for kw in keywords if self.used_keywords.get(kw, 0) >= max_keyword_overlap)

        # If more than 30% of keywords are overused, reject
        if overlap_count > len(keywords) * 0.3:
            return False

        return True

    def add_category(self, name: str):
        """Record that a category has been selected."""
        self.used_names.add(name.lower())
        keywords = self._extract_keywords(name)
        for kw in keywords:
            self.used_keywords[kw] = self.used_keywords.get(kw, 0) + 1


class GameGenerator:
    """Main class that orchestrates game generation."""

    def __init__(self, cache_file: str, verbose: bool = False):
        self.api = WikipediaAPI()
        self.filter = CategoryFilter()
        self.cache = CacheManager(cache_file)
        self.html_gen = HTMLGenerator()
        self.diversity = DiversityChecker()
        self.verbose = verbose
        # Track all words used across all categories to prevent duplicates
        self.used_words: set[str] = set()

    # Seed categories - these tend to have good subcategories
    # Organized by domain to ensure diversity
    SEED_CATEGORIES = [
        # Animals - specific groups
        "Mammals of Africa",
        "Mammals of Asia",
        "Birds of North America",
        "Birds of Europe",
        "Sharks",
        "Whales",
        "Butterflies",
        "Beetles",
        "Freshwater fish",
        "Reptiles",
        "Amphibians",

        # Plants & Nature
        "Trees",
        "Flowers",
        "Medicinal plants",
        "Edible fruits",
        "Vegetables",
        "Herbs",
        "Ferns",
        "Cacti",

        # Food & Drink
        "Cheeses",
        "Breads",
        "Cocktails",
        "Wines",
        "Beers",
        "Desserts",
        "Soups",
        "Sauces",
        "Spices",
        "Pasta",

        # Geography
        "Rivers of Europe",
        "Rivers of Asia",
        "Mountains",
        "Volcanoes",
        "Islands of the Pacific Ocean",
        "Lakes",
        "Deserts",
        "Waterfalls",
        "National parks",
        "UNESCO World Heritage Sites",

        # History & Culture
        "Ancient Greek deities",
        "Roman deities",
        "Norse mythology",
        "Egyptian mythology",
        "Legendary creatures",
        "Heraldic charges",
        "Folk dances",
        "Traditional games",

        # Arts & Literature
        "Art movements",
        "Sculpture",
        "Painting techniques",
        "Literary genres",
        "Poetry forms",
        "Narrative techniques",
        "Rhetorical techniques",
        "Figures of speech",
        "Philosophical concepts",

        # Music
        "Musical instruments",
        "Music genres",
        "Opera singers",
        "Jazz musicians",
        "Classical composers",
        "Record labels",
        "Concert halls",

        # Film & TV
        "Film genres",
        "Animated films",
        "Documentary films",
        "Horror films",
        "Science fiction films",
        "Film directors",
        "Academy Award winners",

        # Sports
        "Olympic sports",
        "Ball games",
        "Combat sports",
        "Water sports",
        "Winter sports",
        "Martial arts",
        "Athletics (track and field)",
        "Racquet sports",
        "Equestrian sports",

        # Science & Technology
        "Chemical elements",
        "Minerals",
        "Astronomical objects",
        "Constellations",
        "Scientific instruments",
        "Medical equipment",
        "Laboratory techniques",
        "Units of measurement",

        # Games
        "Board games",
        "Card games",
        "Video game genres",
        "Puzzle video games",
        "Role-playing games",
        "Dice games",

        # Architecture & Engineering
        "Architectural styles",
        "Building types",
        "Bridges",
        "Towers",
        "Castles",
        "Cathedrals",

        # Textiles & Fashion
        "Fabrics",
        "Clothing",
        "Hats",
        "Footwear",
        "Jewelry",
        "Hairstyles",
        "Textile arts",

        # Language & Writing
        "Alphabets",
        "Writing systems",
        "Punctuation",
        "Parts of speech",
        "Grammatical tenses",

        # Transportation
        "Aircraft",
        "Ships",
        "Automobiles",
        "Trains",
        "Motorcycles",

        # Miscellaneous
        "Phobias",
        "Dances",
        "Knots",
        "Toys",
        "Tools",
        "Weapons",
        "Currency",
        "Flags",
        "Coats of arms",
    ]

    def log(self, msg: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(msg)

    def _try_add_cached_category(self, name: str, members: list[str], good_categories: dict) -> bool:
        """Try to add a cached category, filtering out globally-used words."""
        if not self.filter.is_name_acceptable(name):
            return False

        if not self.diversity.is_diverse_enough(name):
            return False

        # Filter out words already used in other categories
        available = [m for m in members if m.lower() not in self.used_words]

        # If cache doesn't have enough unique words, fetch more from Wikipedia
        if len(available) < 45:
            self.log(f"  Cache has only {len(available)} unique words for {name}, fetching more...")
            return self._try_add_category(name, good_categories, skip_cache=True)

        # Take 45 available words
        selected = available[:45]
        good_categories[name] = selected
        self.diversity.add_category(name)

        # Track these words globally
        for word in selected:
            self.used_words.add(word.lower())

        return True

    def discover_categories(self, target_count: int = 45) -> dict[str, list[str]]:
        """Discover good categories from Wikipedia."""
        good_categories = {}

        # First, use any cached good categories
        cached = self.cache.get_all_good_categories()
        available_cached = {k: v for k, v in cached.items()
                           if len(v) >= 45 and self.filter.is_name_acceptable(k)}
        self.log(f"Found {len(available_cached)} categories in cache")

        # Shuffle cached categories for variety
        cached_keys = list(available_cached.keys())
        random.shuffle(cached_keys)

        # Try to add cached categories (with global duplicate checking)
        for key in cached_keys:
            if len(good_categories) >= target_count:
                break
            self._try_add_cached_category(key, available_cached[key], good_categories)

        if len(good_categories) >= target_count:
            # We have enough from cache
            return dict(list(good_categories.items())[:target_count])

        # Explore seed categories to find more
        explored_seeds = set()
        seeds_to_explore = list(self.SEED_CATEGORIES)
        random.shuffle(seeds_to_explore)

        max_seeds_to_try = 200  # Limit total exploration
        seeds_tried = 0

        while len(good_categories) < target_count and seeds_to_explore and seeds_tried < max_seeds_to_try:
            seed = seeds_to_explore.pop(0)
            if seed in explored_seeds:
                continue
            explored_seeds.add(seed)
            seeds_tried += 1

            print(f"[{len(good_categories)}/{target_count}] Exploring: {seed}")

            added_this_seed = 0
            max_per_seed = 2  # Limit categories added per seed exploration

            # Try the seed category itself
            if self._try_add_category(seed, good_categories):
                print(f"  + Added: {seed}")
                added_this_seed += 1

            # Get subcategories
            subcats = self.api.get_subcategories(seed, limit=30)
            random.shuffle(subcats)

            for subcat in subcats:
                if len(good_categories) >= target_count:
                    break

                if added_this_seed >= max_per_seed:
                    break  # Move on to next seed for diversity

                if self.cache.is_rejected(subcat):
                    continue

                if subcat in good_categories:
                    continue

                if self._try_add_category(subcat, good_categories):
                    print(f"  + Added: {subcat} ({len(good_categories)}/{target_count})")
                    added_this_seed += 1
                else:
                    self.cache.add_rejected(subcat)

            # Add remaining subcategories to END of list for later exploration
            for subcat in subcats[:5]:
                if subcat not in explored_seeds:
                    seeds_to_explore.append(subcat)

        # Save cache
        self.cache.save()

        if len(good_categories) < target_count:
            print(f"\nWarning: Only found {len(good_categories)} categories (needed {target_count})")

        # Select exactly target_count categories
        if len(good_categories) > target_count:
            selected = random.sample(list(good_categories.keys()), target_count)
            good_categories = {k: good_categories[k] for k in selected}

        return good_categories

    def _try_add_category(self, name: str, good_categories: dict, skip_cache: bool = False) -> bool:
        """Try to add a category if it passes all filters. Requires exactly 45 unique words."""
        # Check name first (fast)
        if not self.filter.is_name_acceptable(name):
            self.log(f"  - Rejected (name): {name}")
            return False

        # Check diversity
        if not self.diversity.is_diverse_enough(name):
            self.log(f"  - Rejected (too similar to existing): {name}")
            return False

        # Check cache first (unless skip_cache is set)
        if not skip_cache:
            cached_members = self.cache.get_category(name)
            if cached_members:
                # Filter out words already used in other categories
                available = [m for m in cached_members if m.lower() not in self.used_words]
                if len(available) >= 45:
                    selected = available[:45]
                    good_categories[name] = selected
                    self.diversity.add_category(name)
                    for word in selected:
                        self.used_words.add(word.lower())
                    return True
                # Cache doesn't have enough unique words, fall through to fetch more
                self.log(f"  Cache has only {len(available)} unique words for {name}, fetching more...")

        # Fetch more members than needed to account for duplicates
        members = self.api.get_category_members(name, limit=300)

        if len(members) < 45:
            self.log(f"  - Rejected (too few: {len(members)}): {name}")
            return False

        if len(members) > 500:
            self.log(f"  - Rejected (too many: {len(members)}): {name}")
            return False

        # Check member quality
        if not self.filter.are_members_acceptable(members):
            self.log(f"  - Rejected (member quality): {name}")
            return False

        # Clean member names and filter out category name variations
        cleaned = []
        for m in members:
            clean_name = self.filter.clean_member_name(m)
            if clean_name and not self.filter.is_member_too_similar_to_category(clean_name, name):
                cleaned.append(clean_name)

        # Remove duplicates within category while preserving order
        seen = set()
        unique_cleaned = []
        for m in cleaned:
            m_lower = m.lower()
            if m_lower not in seen and m:
                seen.add(m_lower)
                unique_cleaned.append(m)

        # Filter out words already used in other categories (global duplicate check)
        available = [m for m in unique_cleaned if m.lower() not in self.used_words]

        if len(available) < 45:
            self.log(f"  - Rejected (only {len(available)} unique words, need 45): {name}")
            return False

        # Select 45 members spread across the alphabet
        selected = self.filter.select_alphabetically_spread(available, 45)

        # Success! Cache and add
        self.cache.add_category(name, selected)
        good_categories[name] = selected
        self.diversity.add_category(name)
        # Track these words globally to prevent duplicates in other categories
        for word in selected:
            self.used_words.add(word.lower())
        return True

    def expand_cache(self, target_additional: int):
        """Explore Wikipedia to add more categories to the cache."""
        print(f"Expanding cache by finding {target_additional} new categories...")
        print("(This may take several minutes)\n")

        initial_count = len(self.cache.get_all_good_categories())
        target_count = initial_count + target_additional

        # Explore seed categories
        explored_seeds = set()
        seeds_to_explore = list(self.SEED_CATEGORIES)
        random.shuffle(seeds_to_explore)

        good_categories = {}  # Temporary dict for exploration
        categories_found = 0

        while categories_found < target_additional and seeds_to_explore:
            seed = seeds_to_explore.pop(0)
            if seed in explored_seeds:
                continue
            explored_seeds.add(seed)

            print(f"[{categories_found}/{target_additional}] Exploring: {seed}")

            # Get subcategories
            subcats = self.api.get_subcategories(seed, limit=50)
            random.shuffle(subcats)

            for subcat in subcats:
                if categories_found >= target_additional:
                    break

                if self.cache.is_rejected(subcat):
                    continue

                if self.cache.get_category(subcat):
                    continue  # Already cached

                # Try to add (without diversity checking - we want variety in cache)
                if self._try_cache_category(subcat):
                    print(f"  + Cached: {subcat}")
                    categories_found += 1
                else:
                    self.cache.add_rejected(subcat)

            # Add subcategories to explore later
            for subcat in subcats[:10]:
                if subcat not in explored_seeds:
                    seeds_to_explore.append(subcat)

        self.cache.save()
        final_count = len(self.cache.get_all_good_categories())
        print(f"\nCache expanded from {initial_count} to {final_count} categories (+{final_count - initial_count})")

    def _try_cache_category(self, name: str) -> bool:
        """Try to add a category to cache (without diversity checking)."""
        if not self.filter.is_name_acceptable(name):
            self.log(f"  - Rejected (name): {name}")
            return False

        members = self.api.get_category_members(name, limit=300)

        if len(members) < 45:
            self.log(f"  - Rejected (too few: {len(members)}): {name}")
            return False

        if len(members) > 500:
            self.log(f"  - Rejected (too many: {len(members)}): {name}")
            return False

        if not self.filter.are_members_acceptable(members):
            self.log(f"  - Rejected (member quality): {name}")
            return False

        # Clean member names
        cleaned = []
        for m in members:
            clean_name = self.filter.clean_member_name(m)
            if clean_name and not self.filter.is_member_too_similar_to_category(clean_name, name):
                cleaned.append(clean_name)

        # Remove duplicates
        seen = set()
        unique_cleaned = []
        for m in cleaned:
            m_lower = m.lower()
            if m_lower not in seen and m:
                seen.add(m_lower)
                unique_cleaned.append(m)

        if len(unique_cleaned) < 45:
            self.log(f"  - Rejected (not enough unique: {len(unique_cleaned)}): {name}")
            return False

        # Select 45 members
        selected = self.filter.select_alphabetically_spread(unique_cleaned, 45)
        self.cache.add_category(name, selected)
        return True

    def generate_game(self, output_file: str, seed: int = None):
        """Generate a complete game."""
        self.seed = seed
        print("Discovering categories from Wikipedia...")
        print("(This may take a few minutes on first run)\n")

        categories = self.discover_categories(45)

        if len(categories) < 45:
            print(f"\nError: Could not find enough categories ({len(categories)}/45)")
            print("Try running again - the cache will help find more categories.")
            return False

        # Display what we found
        print(f"\n{'='*60}")
        print("Categories selected for this game:")
        print('='*60)
        for i, (name, members) in enumerate(categories.items(), 1):
            print(f"{i:2}. {name}")
            if self.verbose:
                print(f"    Examples: {', '.join(members[:3])}...")

        # Generate HTML
        self.html_gen.generate(categories, output_file, seed=self.seed)
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate a Wikipedia-based 2025 puzzle game'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output HTML file (default: game_TIMESTAMP.html)'
    )
    parser.add_argument(
        '-c', '--cache',
        default='category_cache.json',
        help='Cache file for discovered categories (default: category_cache.json)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible generation (default: random)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed progress'
    )
    parser.add_argument(
        '--expand-cache',
        type=int,
        default=0,
        metavar='N',
        help='Explore to find N additional categories for the cache (does not generate game)'
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    random.seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # Generate output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        args.output = f'game_{timestamp}.html'

    # Generate game
    generator = GameGenerator(args.cache, verbose=args.verbose)

    # Handle --expand-cache mode
    if args.expand_cache > 0:
        generator.expand_cache(args.expand_cache)
        return 0

    # Check template exists
    if not os.path.exists('2025.html'):
        print("Error: 2025.html template not found in current directory")
        return 1

    success = generator.generate_game(args.output, seed=args.seed)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
