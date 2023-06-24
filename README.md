# SemREC2023

1. Download [OWL2Bench](https://github.com/semrec/semrec.github.io/tree/main/Datasets_SemREC2022/ORE).
2. Download [ORE](https://github.com/semrec/semrec.github.io/tree/main/Datasets_SemREC2022/OWL2Bench).
3. Download [CaLiGraph](https://data.dws.informatik.uni-mannheim.de/CaLiGraph/CaLiGraph-for-SemREC/SemREC-2022-Datasets/).
4. Add the mentioned ontologies to a folder ```datasets```.
5. Run ```assertion_reasoner.ipynb``` (class membership) and/or ```subclass_reasoner.ipynb``` (class subsumption task) and/or ```reasoner.ipynb``` (considers all relations). All notebooks employ:

* Graph Attention Network (GAT).
* Transitive Graph Attention Network (TransGAT): an extended version of Graph Attention Network that encompasses transitivity (aggregation of node information that are 2 hops away). 
* Filter Transitive Graph Attention Network (FilterTransGAT): TransGAT with a filtering mechanism that only applies transitivtiy for "SubClassOf" edges.

The metrics are Hits@1 and Hits@10. 

```Models``` contains the trained models that were used for CaLiGraph_10e5 and CaLiGraph_Full.