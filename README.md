# ludvins.github.io

Source for [ludvins.github.io](https://ludvins.github.io), Luis A. Ortega's
academic website and generated PDF resume.

## Repository layout

```text
.
|-- _data/                 Shared profile, publication, experience, and education data
|-- _includes/             Reusable Jekyll fragments
|-- _layouts/              Shared page shell and navigation
|-- _pages/                Site pages with stable, explicit permalinks
|-- assets/
|   |-- css/               Site styles
|   |-- icons/             Browser icons
|   |-- images/            Site imagery
|   `-- js/                Browser-side behavior
|-- data/                  Generated public documents
|-- tools/resume/          PDF generator and its Python dependencies
|-- index.html             Homepage
|-- _config.yml            Jekyll configuration
`-- Gemfile                Local Jekyll dependencies
```

The files under `_data/` are the source of truth for both the website and the
resume:

- `profile.yml` contains contact details, the current position, research areas,
  skills, and the homepage update date.
- `publications.yml` contains publication metadata, abstracts, tags, and links.
- `experience.yml` contains research, teaching, and visiting positions.
- `education.yml` contains degrees, training, and academic honors.
- `code.yml` contains research-code and open-source entries.

## Run the website locally

Install the Ruby dependencies and start Jekyll:

```sh
bundle install
bundle exec jekyll serve
```

The content pages live under `_pages/`, but their `permalink` front matter keeps
the public URLs at `/publications/`, `/experience/`, and so on.

## Update the website and resume

Edit the relevant YAML file under `_data/`; do not edit the generated PDF
manually. To omit a publication from the resume while retaining it on the site,
set `include_in_resume: false` on that publication.

Install the resume dependencies once:

```sh
python -m pip install -r tools/resume/requirements.txt
```

Rebuild and validate the resume:

```sh
python tools/resume/build.py
python tools/resume/build.py --check
```

The generator writes `data/cv.pdf` atomically after validating its page
count, expected content, source-data fingerprint, and clickable links. GitHub
Actions runs the same validation on every push and pull request.
