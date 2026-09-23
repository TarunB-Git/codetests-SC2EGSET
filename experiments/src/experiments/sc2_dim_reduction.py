# %%

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# UMAP:
import umap
import umap.plot

# PCA
from sklearn.decomposition import PCA

# t-SNE:
from sklearn.manifold import TSNE

from experiments.utils.drop_fields_util import drop_unused_fields
from experiments.utils.groupby_util import groupby_fields_mean
from experiments.utils.prepare_data_util import prep_for_dim_reduction

if __name__ == "__main__":
    # %%
    csv_path = Path("./data/sc2egset.csv").resolve().as_posix()
    loaded_data = pd.read_csv(csv_path)

    plots_dir = Path("./plots").resolve()

    # Suggested fields might include: "map_name", "player_name"
    unique_games = loaded_data["game_hash"].nunique()
    groupby_fields = ["outcome", "race"]
    drop_fields = ["game_time_gameloop", "gameloop"]
    grouped_dataframes = groupby_fields_mean(data=loaded_data, fields=groupby_fields)
    grouped_dataframes = drop_unused_fields(
        grouped_dataframes=grouped_dataframes,
        fields_to_drop=drop_fields,
        unique_games=unique_games,
    )

    # %%
    random_state = 42
    dim_reduction_models = {
        "UMAP": umap.UMAP(random_state=random_state),
        "t-SNE": TSNE(random_state=random_state),
        "PCA": PCA(random_state=random_state),
    }

    dim_red_solved = {}
    for model_name, model in dim_reduction_models.items():
        for grouped_field, result_df in grouped_dataframes.items():
            grouped_dict = {
                "grouped_field": grouped_field,
                "model_name": model_name,
                "result_df": result_df,
            }

            standardized_data, unique_map_display = prep_for_dim_reduction(
                grouped_field=grouped_field, result_df=result_df
            )

            grouped_dict["map_display"] = unique_map_display

            reducer = model
            print(f"Calculating {model_name} for field {grouped_field}")
            if model_name == "UMAP":
                reducer.fit(X=standardized_data, y=result_df[grouped_field])

                grouped_dict["model"] = reducer
                dim_red_solved[f"{model_name}_{grouped_field}"] = grouped_dict

                umap.plot.output_file(
                    filename=f"{grouped_field}_interactive_bokeh_plot.html"
                )
                interactive_plot = umap.plot.interactive(
                    reducer,
                    labels=result_df[grouped_field].map(unique_map_display),
                    color_key_cmap="Paired",
                    background="black",
                )
                static_plot = umap.plot.points(
                    reducer,
                    labels=result_df[grouped_field].map(unique_map_display),
                    color_key_cmap="Paired",
                    background="white",
                )
                fig = static_plot.get_figure()
                fig.savefig(plots_dir / f"{grouped_field}_umap_plot_static.png")
                continue

            Y = reducer.fit_transform(standardized_data)
            plt.figure(figsize=(10, 7))
            plt.scatter(
                Y[:, 0],
                Y[:, 1],
                c=result_df[grouped_field]
                .map(unique_map_display)
                .astype("category")
                .cat.codes,
                cmap="Paired",
            )
            plt.colorbar()
            plt.title(f"{model_name} for {grouped_field}")
            plt.savefig(plots_dir / f"{grouped_field}_{model_name}_plot.png")
            plt.close()
