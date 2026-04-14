# Morphic Resonance Paper — Build with pdflatex
#
# Usage:
#   make          — build PDF (two pdflatex passes for references)
#   make once     — single pdflatex pass (faster, no cross-refs)
#   make clean    — remove build artefacts (keep PDF)
#   make distclean — remove everything including PDF
#   make view     — open PDF in system viewer (macOS)

PAPER   = morphic-sim-paper
LATEX   = pdflatex
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all once clean distclean view

all: $(PAPER).pdf

# Two-pass build: first pass generates aux, second resolves references
$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(LATEXFLAGS) $(PAPER).tex
	$(LATEX) $(LATEXFLAGS) $(PAPER).tex
	@echo ""
	@echo "=== Build complete: $(PAPER).pdf ==="

once: $(PAPER).tex
	$(LATEX) $(LATEXFLAGS) $(PAPER).tex
	@echo ""
	@echo "=== Single pass complete: $(PAPER).pdf ==="

clean:
	rm -f $(PAPER).aux \
	       $(PAPER).log \
	       $(PAPER).out \
	       $(PAPER).toc \
	       $(PAPER).bbl \
	       $(PAPER).blg \
	       $(PAPER).lof \
	       $(PAPER).lot \
	       $(PAPER).lol \
	       $(PAPER).fls \
	       $(PAPER).fdb_latexmk \
	       $(PAPER).synctex.gz \
	       texput.log
	@echo "Cleaned build artefacts."

distclean: clean
	rm -f $(PAPER).pdf
	@echo "Removed PDF."

view: $(PAPER).pdf
	open $(PAPER).pdf
