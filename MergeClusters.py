import pandas as pd

def load_and_merge_cluster_data(speaking_file, loc_file, hover_file):
    # Load datasets
    speaking_df = pd.read_csv(speaking_file)
    loc_df = pd.read_csv(loc_file)
    hover_df = pd.read_csv(hover_file)

    # Select and rename relevant columns
    speaking_df = speaking_df[["Group ID", "Participant ID", "Cluster_GMM"]]
    speaking_df.rename(columns={"Cluster_GMM": "Speaking Cluster"}, inplace=True)

    loc_df = loc_df[["Group ID", "Participant ID", "Cluster_GMM"]]
    loc_df.rename(columns={"Cluster_GMM": "Loc Cluster"}, inplace=True)

    hover_df = hover_df[["Group ID", "Participant ID", "Cluster_GMM"]]
    hover_df.rename(columns={"Cluster_GMM": "Gaze Cluster"}, inplace=True)

    # Merge on Group ID and Participant ID
    merged_df = speaking_df.merge(loc_df, on=["Group ID", "Participant ID"])
    merged_df = merged_df.merge(hover_df, on=["Group ID", "Participant ID"])

    # Sort
    merged_df.sort_values(by=["Group ID", "Participant ID"], inplace=True)

    # Save to file
    merged_df.to_csv("merged_cluster_config.csv", index=False)
    print("✅ merged_cluster_config.csv successfully created.")
    print(merged_df)

# Only run this if executing as a script
if __name__ == "__main__":
    load_and_merge_cluster_data(
        "gmm_clustering_results_speaking.csv",
        "loc_gmm_clustering_results.csv",
        "hover_gmm_user_clustering_results.csv"
    )
