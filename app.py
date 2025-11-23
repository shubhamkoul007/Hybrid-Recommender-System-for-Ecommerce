from flask import Flask, request, render_template
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# load files===========================================================================================================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def content_based_recommendations(train_data, item_name, top_n=10):
    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find index of item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Cosine similarity for that item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Top N items (excluding itself)
    top_similar_items = similar_items[1:top_n + 1]

    # Extract the row indices
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Product details
    recommended_items_details = train_data.iloc[recommended_item_indices][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
    ]

    return recommended_items_details


# ----------------------------------------------------------------------
# Predefined image paths
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


@app.route("/")
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

    return render_template(
        'index.html',
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=random.choice(price)
    )


@app.route("/main")
def main():
    return render_template('main.html')


@app.route("/index")
def indexredirect():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

    return render_template(
        'index.html',
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=random.choice(price)
    )


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            return render_template('main.html', message="No recommendations available for this product.")
        else:
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

            return render_template(
                'main.html',
                content_based_rec=content_based_rec,
                truncate=truncate,
                random_product_image_urls=random_product_image_urls,
                random_price=random.choice(price)
            )


if __name__ == '__main__':
    app.run(debug=True)
