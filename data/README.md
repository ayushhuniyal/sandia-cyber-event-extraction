# Data

## Source
Cybersecurity news articles provided by Sandia National 
Laboratories, scraped from open-access sources via their 
internal web scraping tool, Ghost Trap.

## Structure
Each record contains four attributes:
- `date`: Publication date
- `title`: Article headline
- `source`: Domain or publisher
- `text`: Full article body

## Scale
~13,700 unique articles following preprocessing.
Timeframe: Early 2024 – Mid 2025.

## Preprocessing Applied
- Null entries removed
- Exact and metadata duplicates dropped
- Near-duplicates retained (may reflect same event 
  across multiple outlets — essential for clustering)
- Unicode/encoding artifacts cleaned
- Processed in Alteryx Designer 2025.1.2.120

## Note
Raw data is not included in this repository.
