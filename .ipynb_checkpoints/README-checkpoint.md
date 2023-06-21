# SemREC2023

1. Download [OWL2Bench](https://github.com/semrec/semrec.github.io/tree/main/Datasets_SemREC2022/ORE).
2. Download [ORE](https://github.com/semrec/semrec.github.io/tree/main/Datasets_SemREC2022/OWL2Bench).
3. Download [CaLiGraph](https://data.dws.informatik.uni-mannheim.de/CaLiGraph/CaLiGraph-for-SemREC/SemREC-2022-Datasets/).
4. Add the mentioned ontologies to a folder ```datasets```.
5. Run ```assertion_reasoner.ipynb``` (class membership) and/or ```subclass_reasoner.ipynb``` (class subsumption task) and/or ```combined_reasoner.ipynb``` (considers all relations). All notebooks employ Graph Attention Network and the adapted Graph Attention Network on the mentioned ontologies. The metrics are Precision, Recall, F1-Score, Hits@1 and Hits@10. 