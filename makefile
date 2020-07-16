
setup:
	python -m spacy download en
	depccg_en download elmo

ccg2lambda:
	cat data/glue/${file} | depccg_en --elmo --format ccg2lambda --annotator spacy > data/formal/${file}
