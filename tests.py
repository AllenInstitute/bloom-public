# Generates test data for the model.

import os
import random
from typing import List

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def dummy_dna_coords(n_peaks: int) -> List[str]:
    """
    Generate dummy BED format values.

    Parameters
    ----------
    n_peaks : int
        Number of dummy BED values to generate.

    Returns
    -------
    List[str]
        List of dummy BED format values.
    """
    fixed_length = 500  # Fixed length for the peaks
    jitter = 100
    bed_values = []
    for _ in range(n_peaks):
        chromosome = f"chr{random.randint(1, 20)}"
        start = random.randint(1, 10_000_000) + random.randint(0, jitter)
        end = start + fixed_length + random.randint(0, jitter)
        bed_values.append(f"{chromosome}:{start}-{end}")
    return bed_values


def generate_h5ad(
    n_cells: int = int(2e4),
    n_genes: int = int(1e4),
    n_peaks: int = int(4e4),
    n_embedding_dim: int = int(1e2),
    path: str | None = None,
):
    """
    Generate dummy data for testing. All files are saved as h5ad files.

    Parameters
    ----------
    n_cells : int, optional
        Number of cells. Default is 1e4. (Pilot has about 1e6 cells)
    n_genes : int, optional
        Number of genes. Default is 1e4. (Pilot has about 1e4 genes)
    n_peaks : int, optional
        Number of peaks. Default is 4e4. (Pilot has about 2e6 peaks)
    n_embedding_dim : int, optional
        Dimension of the embeddings. Default is 1e3. (Pilot has about 4e3)
    """

    # only contains cell ids
    obs = pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(n_cells)],
    })
    obs.set_index("cell_id", inplace=True)

    # ATAC values
    sparsity = 0.98
    X = np.random.rand(n_cells, n_peaks) > sparsity
    X = sp.csr_matrix(X, dtype=np.int8)
    atac_var = pd.DataFrame({"peak_id": dummy_dna_coords(n_peaks)})
    atac_var.set_index("peak_id", inplace=True)
    atac = ad.AnnData(X, obs=obs, var=atac_var)

    # RNA values
    sparsity = 0.60
    X = np.random.rand(n_cells, n_genes) > sparsity
    X = sp.csr_matrix(X, dtype=float)
    rna_var = pd.DataFrame({"gene_id": dummy_dna_coords(n_genes)})
    rna_var.set_index("gene_id", inplace=True)
    rna = ad.AnnData(X, obs=obs, var=rna_var)

    # Peak embeddings
    emb_peaks = ad.AnnData(X=np.random.rand(n_peaks, n_embedding_dim), obs=atac_var)

    # Gene embeddings
    emb_genes = ad.AnnData(X=np.random.rand(n_genes, n_embedding_dim), obs=rna_var)

    # Cell metadata
    cell_metadata = obs.copy()
    cell_metadata["species"] = pd.Series(
        np.random.choice(["human", "mouse"], n_cells, replace=True),
        index=obs.index,
    ).astype("category")
    cell_metadata["species_id"] = cell_metadata["species"].cat.codes

    cell_metadata["celltype"] = pd.Series(
        np.random.choice(["A", "B", "C", "D"], n_cells, replace=True),
        index=obs.index,
    ).astype("category")
    cell_metadata["celltype_id"] = cell_metadata["celltype"].cat.codes

    X = sp.csr_matrix(np.zeros((n_cells, 1), dtype=int))  # empty matrix to retain adata structure
    cell_metadata = ad.AnnData(X, obs=cell_metadata)

    for adata in [atac, rna, emb_peaks, emb_genes, cell_metadata]:
        print(adata)

    assert np.array_equal(cell_metadata.obs.index, atac.obs.index), "atac"
    assert np.array_equal(cell_metadata.obs.index, rna.obs.index), "rna"
    assert np.array_equal(emb_peaks.obs.index, atac.var.index), "emb_peaks"
    assert np.array_equal(emb_genes.obs.index, rna.var.index), "emb_genes"

    if not os.path.exists(path):
        print("Creating path: ", path)
        os.makedirs(path)

    atac.write_h5ad(path + "atac.h5ad")
    rna.write_h5ad(path + "rna.h5ad")
    emb_peaks.write_h5ad(path + "emb_peaks.h5ad")
    emb_genes.write_h5ad(path + "emb_genes.h5ad")
    cell_metadata.write_h5ad(path + "cell_metadata.h5ad")
    print("Data saved to: ", path)
    return


if __name__ == "__main__":
    save_path = "./"

    # Generate the data
    generate_h5ad(
        n_cells=int(1e4),
        n_genes=int(1e4),
        n_peaks=int(4e4),
        n_embedding_dim=int(1e2),
        path=save_path + "dummy/",
    )