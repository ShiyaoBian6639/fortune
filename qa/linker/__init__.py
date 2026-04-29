"""Stock-news linker — tags each news article with the stocks it mentions.

MVP path uses Aho-Corasick over the alias dictionary. Phase 2 adds dense
semantic matching against entity cards via bge-m3 + a logistic-regression
reranker for indirect mentions ("EV battery sector" → CATL/BYD).
"""
