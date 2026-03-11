import openslide
import pandas as pd
from PIL import Image
import os

def is_background(tile: Image.Image, threshold: float = 0.35) -> bool:
    """
    Retourne True si la tuile contient plus de `threshold` % de background noir.
    """
    import numpy as np
    tile_array = np.array(tile)
    # Pixel noir = R < 20 et G < 20 et B < 20
    black_mask = (tile_array[:, :, 0] < 20) & \
                 (tile_array[:, :, 1] < 20) & \
                 (tile_array[:, :, 2] < 20)
    black_ratio = black_mask.sum() / (tile_array.shape[0] * tile_array.shape[1])
    return black_ratio > threshold

def extract_tiles_from_folder(folder_path: str, marker: str, df: pd.DataFrame, output_dir: str, tile_size: int = 224) -> pd.DataFrame:
    """
    Extrait les tuiles de toutes les images d'un dossier et les trie par label.
    
    Args:
        folder_path: dossier contenant les images (.svs, .tiff, ...)
        df: DataFrame contenant les métadonnées (colonnes: 'slide', 'final_label')
        output_dir: dossier de sortie racine
        tile_size: taille des tuiles (défaut: 224)
    
    Returns:
        DataFrame avec les coordonnées et métadonnées de toutes les tuiles extraites
    """
    df = df[df["stain"] == marker]
    df["patient_id"] = df["patient_id"].astype(str)
    
    # Créer les deux dossiers de sortie selon les labels
    labels = df['status'].unique()
    for label in labels:
        os.makedirs(os.path.join(output_dir, str(label)), exist_ok=True)
    
    # Lister toutes les images du dossier
    supported_formats = ('.svs', '.tiff', '.tif', '.ndpi', '.scn', '.mrxs')
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(supported_formats)
    ]
    
    if not image_files:
        raise ValueError(f"Aucune image trouvée dans {folder_path}")
    
    print(f"{len(image_files)} image(s) trouvée(s) dans '{folder_path}'")
    
    all_tiles_records = []


    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        
        # Récupérer le label dans le DataFrame
        image_name = image_name.replace("_masked.svs", "")

        # DEBUG - à retirer ensuite
        print(f"image_name nettoyé : '{image_name}'")
        print(f"Exemples patient_id : {df['patient_id'].values[:5]}")

        row = df[df['patient_id'] == image_name]

        
        if row.empty:
            #print(f"⚠️  '{image_name}' non trouvée dans le DataFrame, ignorée.")
            continue
        
        label = row['status'].values[0]
        print(f"\n→ Traitement : {image_name} | Label : {label}")
        
        # Ouvrir l'image
        try:
            slide = openslide.OpenSlide(image_path)
        except Exception as e:
            print(f"⚠️  Impossible d'ouvrir '{image_name}' : {e}")
            continue
        
        width, height = slide.dimensions
        print(f"  Dimensions : {width} x {height}")
        
        # Dossier de sortie selon le label (Atypical ou Normal)
        save_dir = os.path.join(output_dir, str(label))
        
        # Extraction des tuiles
        tile_id = 0
        image_stem = os.path.splitext(image_name)[0]
        
        for y in range(0, height - tile_size + 1, tile_size):
            for x in range(0, width - tile_size + 1, tile_size):
                
                tile = slide.read_region(
                    location=(x, y),
                    level=0,
                    size=(tile_size, tile_size)
                ).convert("RGB")
                
                #Filtre background
                if is_background(tile, threshold=0.40):
                    continue  # On skip : pas de save, pas d'ajout au DataFrame

                tile_name = f"{image_stem}_tile_{tile_id}.png"
                tile.save(os.path.join(save_dir, tile_name))
                
                all_tiles_records.append({
                    'slide':        image_name,
                    'tile_id':      tile_id,
                    'dataset_uid':  tile_name,
                    'final_label':  label,
                    'x':            x,
                    'y':            y,
                    'tile_path':    os.path.join(save_dir, tile_name)
                })
                
                tile_id += 1
        
        slide.close()
        print(f"  ✅ {tile_id} tuiles extraites → '{save_dir}'")
    
    # Résumé final
    tiles_df = pd.DataFrame(all_tiles_records)
    print(tiles_df.head)
    print(tiles_df.columns)
    total = len(tiles_df)
    for label in labels:
        count = len(tiles_df[tiles_df['final_label'] == label])
        print(f"\n{label} : {count} tuiles ({100*count/total:.1f}%)")
    print(f"Total : {total} tuiles")
    
    return tiles_df


# --- Exemple d'utilisation ---
list_markers = ["BCL2"] #, "BCL6", "HE", "MUM1", "MYC"
df_meta = pd.read_csv(f"data\\dataset_list.csv")
print(df_meta.head)
output_path = "images_optimus"
os.makedirs(output_path, exist_ok=True)

for marker in list_markers :
    path = os.path.join("data", marker)
    df_markers = df_meta[df_meta["stain"] == marker]
    out_path = os.path.join(output_path, marker)
    os.makedirs(out_path, exist_ok=True)
    tiles_df = extract_tiles_from_folder(
        folder_path=path,
        marker=marker,
        df=df_meta,
        output_dir=out_path,
        tile_size=224
    )
    tiles_df.to_csv(f'{output_path}\\{marker}_coordinates.csv', index=False)
