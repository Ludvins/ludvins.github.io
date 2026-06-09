# ludvins.github.io

Source for the personal academic website at https://ludvins.github.io.

## Structure

- `_data/publications.yml`: publication metadata, abstracts, tags, and links.
- `_data/experience.yml`: research, teaching, and visiting positions.
- `_data/education.yml`: education timeline and academic honors.
- `_data/code.yml`: research-code entries.
- `_includes/publication-card.html`: shared publication rendering.
- `_layouts/base.liquid`: site shell and navigation.
- `css/site.css`: site styling.
- `js/site.js`: publication search and filters.

Run locally with:

```sh
bundle install
bundle exec jekyll serve
```
