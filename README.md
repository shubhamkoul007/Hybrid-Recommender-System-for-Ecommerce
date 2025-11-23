# Hybrid-Recommender-System-for-Ecommerce

Developed a hybrid recommendation engine by combining User-Based Collaborative Filtering (UBCF) with Content-Based Filtering (CBF) to generate ranked product recommendations. Implemented the CBF module using TF-IDF vectorization to convert product descriptions into weighted semantic vectors and applied cosine similarity for item–item matching. Built the UBCF pipeline by constructing user–item interaction matrices and identifying similar user neighbourhoods using cosine similarity. Merged both recommendation streams using tunable heuristic weighting for more stable and personalized outputs and packaged the full system into an interactive Flask application. 



For eg, make search of something like - The Art of Shaving Mens Hair Styling Gel, Juniper Scent, 2 Fl Oz and the hybrid recommender will combine results from user based collaborative filtering and content base recommendation and gives ranked recommendations list. 
