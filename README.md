# ludvins.github.io

Source for the personal academic website at https://ludvins.github.io.

## Structure

- `_data/profile.yml`: shared contact details, current position, research areas, and skills.
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

## Updating the website and CV

The website and PDF resume use the same files under `_data/`. Update those files first;
do not edit `data/cv.pdf` manually.

Install the resume builder once:

```sh
python -m pip install -r requirements-resume.txt
```

Then rebuild and validate the public PDF:

```sh
python scripts/build_resume.py
python scripts/build_resume.py --check
```

The builder writes `data/cv.pdf` only after checking its page count, expected content,
source-data fingerprint, and clickable links. Publications and ongoing projects are
included by default; add `include_in_resume: false` to an entry in
`_data/publications.yml` to omit it from the PDF while keeping it on the website.
