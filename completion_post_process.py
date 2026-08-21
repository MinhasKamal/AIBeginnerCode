from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def plot_np_arr(
            np_arr_list: list,
            file_name: str
        ):
    fig, axes = plt.subplots(1, len(np_arr_list), figsize=(6 * len(np_arr_list), 10))

    if len(np_arr_list) > 1:
        for index, np_arr in enumerate(np_arr_list):
            axes[index].imshow(np_arr)
            # axes[index].imshow(np_arr, cmap="gray")
    else :
        axes.imshow(np_arr_list[0])
        # axes.imshow(np_arr_list[0], cmap="gray")
    
    plt.savefig(f"{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.clf()
    
    return


def load_img(
            path: str
        ) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img)


def remove_background(
            img: np.ndarray
        ) -> np.ndarray:
    print(img.shape)
    # if img.ndim != 3 or img.shape[2] != 3:
    #     raise ValueError("Input image must have shape (H, W, 3).")

    # Alpha = 255 everywhere
    alpha = np.full(img.shape[:2], 255, dtype=np.uint8)

    # Make pixels transparent where R=G=B=0
    alpha[np.all(img <= 5, axis=-1)] = 0

    # Stack RGB and alpha
    return np.dstack((img, alpha))
    

if __name__ == "__main__":

    # completed_opaque_img = load_img("/workspace/minhas/dataset/test/sd1.5_completion_5_a1/compla1.png_990_1.5.png")
    occluded_opaque_img = load_img("/workspace/minhas/dataset/test/sd1.5_completion_5_a1/compla1.png_in.png")
    # occluded_img = load_img("/workspace/minhas_dgx/hypersim_object_completion/occluded/a1.png")
    
    # resedue_img = unoccluded_img - occluded_img
    
    # plot_np_arr([occluded_img, unoccluded_img, resedue_img], "test.pdf")
    
    # Image.fromarray(resedue_img.astype(np.uint8)).save("resedue.png")

    occluded_transparesnt_img = remove_background(occluded_opaque_img)
    # completed_transparesnt_img = remove_background(opaque_img)
    
    plot_np_arr([occluded_opaque_img, occluded_transparesnt_img], "completion_post")
    # plot_np_arr([opaque_img, transparesnt_img], "completion_post")
