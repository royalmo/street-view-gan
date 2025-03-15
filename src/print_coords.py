import matplotlib.pyplot as plt
import kagglehub

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import lib

if __name__=="__main__":
    DATASET_PATH = kagglehub.dataset_download("paulchambaz/google-street-view")
    print(f"Dataset location: {DATASET_PATH}")

    COORDS = lib.dataset.retrieve_coords(f"{DATASET_PATH}/dataset/coords.csv")
    DATASET_LENGTH = len(COORDS)
    print(f"Dataset length: {DATASET_LENGTH}")

    # Extract latitudes and longitudes
    lats, longs = zip(*COORDS)

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    ax.scatter(longs, lats, c='red', marker='o', s=[1 for _ in lats], transform=ccrs.PlateCarree())
    ax.set_title("Coordinates of the dataset images")
    plt.show()
