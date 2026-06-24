# Overleaf DMRGAT Paper

This folder contains a LaTeX version of the DMRGAT sector-rotation paper.

## Files

- `main.tex`: main manuscript.
- `references.bib`: bibliography database.

## Overleaf Usage

Upload the entire `overleaf_dmrgat` folder to Overleaf and compile `main.tex`.

Recommended compiler:

```text
pdfLaTeX
```

Compile sequence:

```text
pdfLaTeX -> BibTeX -> pdfLaTeX -> pdfLaTeX
```

## Style Notes

The structure imitates the provided finance-engineering SSRN paper in broad layout:

- centered title and authors;
- abstract and keywords;
- introduction with background, contribution, literature review, and organization;
- formal model/product design section;
- algorithm and training procedure;
- theoretical propositions;
- numerical examples and tables;
- application and conclusion.

The content is original to the DMRGAT project and does not copy the SSRN paper.
