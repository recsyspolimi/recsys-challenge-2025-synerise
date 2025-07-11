from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from models.hybrid.utils import load_clients_and_embeddings

def regularize_embeddings(embeddings, old_embeddings, alpha_old=0.3):
    """
    Regularizes the embeddings by combining current and old embeddings with a specified alpha.

    Args:
        embeddings (np.ndarray): Current embeddings.
        old_embeddings (np.ndarray): Old embeddings.
        alpha_old (float): Weight for the old embeddings.

    Returns:
        np.ndarray: Regularized embeddings.
    """
    assert embeddings.shape == old_embeddings.shape, "Current and old embeddings must have the same shape."
    return alpha_old * old_embeddings + (1 - alpha_old) * embeddings

def main():
    parser = ArgumentParser()
    parser.add_argument('--sra_dir', type=str, required=True, help="Folder where SRA embeddings are saved")
    parser.add_argument("--sra_old2new_dir", type=str, required=True, help="Folder where SRA Old2New embeddings are saved")
    parser.add_argument("--mtl_dir", type=str, required=True, help="Folder where MTL embeddings are saved")
    parser.add_argument("--mtl_old2new_dir", type=str, required=True, help="Folder where MTL Old2New embeddings are saved")
    parser.add_argument("--fip_dir", type=str, required=True, help="Folder where FIP embeddings are saved")
    parser.add_argument("--prop_mtl_dir", type=str, required=True, help="Folder where Propensity-Focused MTL embeddings are saved")
    parser.add_argument("--churn_mtl_dir", type=str, required=True, help="Folder where Churn-Focused MTL embeddings are saved")
    parser.add_argument("--bpr_sku_dir", type=str, required=True, help="Folder where BPR SKU embeddings are saved")
    parser.add_argument("--bpr_category_dir", type=str, required=True, help="Folder where BPR Category embeddings are saved")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Folder where the final concatenated embeddings will be saved")
    args = parser.parse_args()

    # First let's focus on the SRA and MTL embeddings
    sra_client_ids, sra_embeddings = load_clients_and_embeddings(args.sra_dir)
    sra_client_ids_old, sra_embeddings_old = load_clients_and_embeddings(args.sra_old2new_dir)

    assert np.array_equal(sra_client_ids, sra_client_ids_old), "Client IDs in SRA and SRA Old2New must match."

    sra_regularized_embeddings = regularize_embeddings(sra_embeddings, sra_embeddings_old)

    mtl_client_ids, mtl_embeddings = load_clients_and_embeddings(args.mtl_dir)
    mtl_client_ids_old, mtl_embeddings_old = load_clients_and_embeddings(args.mtl_old2new_dir)

    mtl_regularized_embeddings = regularize_embeddings(mtl_embeddings, mtl_embeddings_old)

    assert np.array_equal(sra_client_ids, sra_client_ids_old), "Client IDs in SRA and SRA Old2New must match."
    assert np.array_equal(mtl_client_ids, mtl_client_ids_old), "Client IDs in MTL and MTL Old2New must match."
    assert np.array_equal(sra_client_ids, mtl_client_ids), "Client IDs in SRA and MTL must match."
    
    hybrid_reg_client_ids = sra_client_ids
    hybrid_reg_embeddings = np.concatenate((sra_regularized_embeddings, mtl_regularized_embeddings), axis=1)

    # Now let's load the other embeddings
    fip_client_ids, fip_embeddings = load_clients_and_embeddings(args.fip_dir)
    prop_MTL_client_ids, prop_MTL_embeddings = load_clients_and_embeddings(args.prop_mtl_dir)
    churn_MTL_client_ids, churn_MTL_embeddings = load_clients_and_embeddings(args.churn_mtl_dir)
    bpr_sku_client_ids, bpr_sku_embeddings = load_clients_and_embeddings(args.bpr_sku_dir)
    bpr_category_client_ids, bpr_category_embeddings = load_clients_and_embeddings(args.bpr_category_dir)
    
    assert np.array_equal(hybrid_reg_client_ids, fip_client_ids)
    assert np.array_equal(hybrid_reg_client_ids, prop_MTL_client_ids)
    assert np.array_equal(hybrid_reg_client_ids, churn_MTL_client_ids)
    assert np.array_equal(hybrid_reg_client_ids, bpr_sku_client_ids)
    assert np.array_equal(hybrid_reg_client_ids, bpr_category_client_ids)
    
    gru3H_concat_gru_future = np.concatenate((prop_MTL_embeddings, fip_embeddings), axis=1)
    
    final_embeddings = np.concatenate(
        (hybrid_reg_embeddings, gru3H_concat_gru_future, churn_MTL_embeddings, bpr_sku_embeddings, bpr_category_embeddings),
        axis=1
    )
    
    print(f"Final embeddings shape: {final_embeddings.shape}")

    assert final_embeddings.shape[1] == 744, "Final embeddings should have 744 features per client."
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "embeddings.npy", final_embeddings)
    np.save(output_path / "client_ids.npy", hybrid_reg_client_ids)


if __name__ == "__main__":
    main()