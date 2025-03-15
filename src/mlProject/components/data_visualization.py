import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlProject.entity.config_entity import DataVisualizationConfig
from wordcloud import WordCloud


class DataVisualization:
    def __init__(self, config: DataVisualizationConfig, df_merged_master: pd.DataFrame):
        self.config = config
        self.df_merged_master=df_merged_master

    def generate_top_10_product_types(self):
        plt.figure(figsize=(12, 6))
        top_categories = self.df_merged_master["product_type"].value_counts().head(10)
        sns.barplot(x=top_categories.index, y=top_categories.values, hue=top_categories.index, palette="viridis",
                    legend=False)
        plt.xticks(rotation=45)
        plt.title("Top 10 Product Categories", fontsize=14)
        plt.xlabel("Type")
        plt.ylabel("Number of Products")
        save_path = f"{self.config.root_dir}" + "/top_10_product_types.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def generate_top_10_product_brands(self):
        plt.figure(figsize=(12, 6))
        top_categories = self.df_merged_master["brand"].value_counts().head(10)
        sns.barplot(x=top_categories.index, y=top_categories.values, hue=top_categories.index, palette="viridis",
                    legend=False)
        plt.xticks(rotation=45)
        plt.title("Top 10 Product brands", fontsize=14)
        plt.xlabel("Brand")
        plt.ylabel("Number of Products")
        save_path = f"{self.config.root_dir}" + "/top_10_product_brands.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def generate_number_products_per_country(self):
        plt.figure(figsize=(12, 6))
        product_by_country = self.df_merged_master["country"].value_counts()
        sns.barplot(x=product_by_country.index, y=product_by_country.values, hue=product_by_country.index,
                    palette="viridis", legend=False)
        plt.xticks(rotation=45)
        plt.title("Products by Country", fontsize=14)
        plt.xlabel("Country")
        plt.ylabel("Number of Products")
        save_path = f"{self.config.root_dir}" + "/number_of_products_per_country.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def generate_top_5_product_categories_by_country(self):
        # For each country find the top n products.
        TOP_N = 5
        NUM_COLS = 3
        top_products_by_country = self.df_merged_master.groupby("country")["product_type"].value_counts().groupby(
            "country").head(TOP_N).reset_index(name='count')

        # Order the countries by the number of products in DESCENDING order
        country_order = self.df_merged_master["country"].value_counts().sort_values(ascending=False).index

        # Determine the number of rows and columns for the grid
        num_countries = len(country_order)
        num_rows = (num_countries + NUM_COLS - 1) // NUM_COLS

        # Create subplots
        fig, axes = plt.subplots(num_rows, NUM_COLS, figsize=(10, num_rows * 5), constrained_layout=True)

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Plot the top products by country
        for i, country in enumerate(country_order):
            ax = axes[i]
            top_products = top_products_by_country[top_products_by_country["country"] == country]
            sns.barplot(x="product_type", y="count", data=top_products, hue=top_products.index, palette="viridis",
                        ax=ax)
            ax.set_title(f"Top {TOP_N} Product Categories in {country}", fontsize=14)
            ax.set_xlabel("Product Type")
            ax.set_ylabel("Number of Products")
            ax.tick_params(axis='x', rotation=45)

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        save_path = f"{self.config.root_dir}" + "/top_5_product_categories_by_country.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def generate_word_cloud_prod_description(self):
        # Determine the number of available rows
        num_rows = self.df_merged_master["product_description"].dropna().shape[0]
        print(num_rows)

        text = " ".join(self.df_merged_master["product_description"].dropna().astype(str).sample(1500))
        wordcloud = WordCloud(width=600, height=300, background_color="white").generate(text)

        plt.figure(figsize=(6, 3))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Product Descriptions Word Cloud")

        save_path = f"{self.config.root_dir}" + "/product_description_word_cloud.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def generate_word_cloud_prod_names(self):
        ## Word cloud for product names
        text = " ".join(self.df_merged_master["item_name"].dropna().astype(str).sample(2000))
        wordcloud = WordCloud(width=600, height=300, background_color="white").generate(text)

        plt.figure(figsize=(6, 3))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Product Names Word Cloud")

        save_path = f"{self.config.root_dir}" + "/product_names_word_cloud.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
