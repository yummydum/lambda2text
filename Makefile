ccg2lambda:
	cat ${src} | depccg_en --format ccg2lambda --annotator spacy --model elmo  > ${trg}

ccg2lambda_xml:
	cat ${src} | depccg_en --format jigg_xml_ccg2lambda --annotator spacy --model elmo  > ${trg}
	# cat sentences.txt | depccg_en --format jigg_xml_ccg2lambda --annotator spacy --model elmo  > sentences.sem.xml                  
	# python scripts/prove.py sentences.sem.xml --graph_out graphdebug.html

prove:
	cd ccg2lambda
	cp en/coqlib_sick.v coqlib.v
	coqc coqlib.v
	cp en/tactics_coq_sick.txt tactics_coq.txt
	python scripts/prove.py ../data/SICK/pair_1.sem.xml --proof ../data/SICK_proof/pair_1.sem.xml --subgoals --subgoals_out ../data/SICK_subgoal/pair_1.txt 
