ccg2lambda:
	cat ${file} | depccg_en --format ccg2lambda --annotator spacy > data/formal/$(shell basename ${file})
