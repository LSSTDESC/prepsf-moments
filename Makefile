
texfile := prepsfmom
default: pdf

# just run it
bib:
	bibtex ${texfile}

pdf:
	pdflatex ${texfile}
	pdflatex ${texfile}

clean:
	rm -f \
	${texfile}.dvi \
	${texfile}.out \
	${texfile}.ps \
	${texfile}.pdf \
	${texfile}.aux \
	${texfile}.bbl \
	${texfile}.blg \
	${texfile}.toc \
	${texfile}.log \
	${texfile}.fdb_latexmk \
	${texfile}.fls \
	${texfile}.brf
