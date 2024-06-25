import os
import argparse
import time
import torch
import warnings
import numpy as np
from Bio import PDB
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import MMCIFIO
from utils.helpers import (
    read_fasta_ids,
    remove_hetatms,
    get_max_residue_id,
    rename_predicted_atoms,
    add_predicted_atoms,
    get_all_metalbinding_resids,
    get_bb,
    create_grid_fromBB,
    get_probability_mean,
    write_cubefile,
    find_unique_sites,
    get_all_protein_resids,
    maxprobability,
)
from utils.voxelization import processStructures as processStructures
from utils.model import Model
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress specific PDBConstructionWarnings
warnings.simplefilter("ignore", PDBConstructionWarning)


def process_cif_file(file_path, output_folder, model, device, args):
    base_name = os.path.basename(file_path).replace("_with_", "_").split(".")[0]
    ion_type = file_path.split("_with_")[-1].split(".")[0]

    try:
        if len(args.id) == 0 and not args.metalbinding:
            print(f"No resid passed for {file_path}, using whole protein")
            ids = get_all_protein_resids(file_path)
        elif len(args.id) == 0 and args.metalbinding:
            print(
                f"Using all residues that can bind metals with their sidechain for {file_path}"
            )
            ids = get_all_metalbinding_resids(file_path)
        else:
            print(f"Using the following indexes for {file_path}", args.id)
            ids = args.id

        if ids is None:
            print(f"Skipping {file_path} due to error in reading residues")
            return

        voxels, prot_centers, _, _ = processStructures(file_path, ids)
        voxels = voxels.to(device)
        outputs = torch.zeros([voxels.size()[0], 1, 32, 32, 32])
        with warnings.catch_warnings(), torch.no_grad():
            warnings.filterwarnings("ignore")
            for i in range(0, voxels.size()[0], args.batch_size):
                batch = voxels[i : i + args.batch_size]
                o = model(batch)
                outputs[i : i + args.batch_size] = o.cpu()

        outputs = outputs.flatten().numpy()
        prot_v = np.vstack(prot_centers)

        bb = get_bb(prot_v)
        grid, box_N = create_grid_fromBB(bb)
        probability_values = get_probability_mean(grid, prot_v, outputs)

        if args.writecube:
            write_cubefile(
                bb,
                probability_values,
                box_N,
                outname=os.path.join(output_folder, f"{base_name}.cube"),
                gridres=1,
            )

        if args.writeprobes:
            find_unique_sites(
                probability_values,
                grid,
                writeprobes=args.writeprobes,
                probefile=os.path.join(output_folder, f"{base_name}_probes.pdb"),
                threshold=args.threshold,
                p=args.pthreshold,
            )

        if args.maxp:
            maxprobability(probability_values, grid, file_path, base_name)

        # Process the predicted probe file and the input file
        parser = MMCIFParser()
        structure = parser.get_structure(base_name, file_path)
        remove_hetatms(structure)

        probe_file_path = os.path.join(output_folder, f"{base_name}_probes.pdb")
        if os.path.exists(probe_file_path):
            probes_parser = PDB.PDBParser()
            probes_structure = probes_parser.get_structure(
                base_name + "_probes", probe_file_path
            )
            max_residue_id = get_max_residue_id(structure)
            new_residues = rename_predicted_atoms(
                probes_structure, ion_type, max_residue_id + 1
            )
            add_predicted_atoms(structure, new_residues)
        else:
            print(
                f"No probe file generated for {file_path}, using original structure with HETATMs removed"
            )

        # Save the resulting structure into a new CIF file
        io = MMCIFIO()
        output_file_path = os.path.join(output_folder, f"{base_name}_metal3d.cif")
        io.set_structure(structure)
        io.save(output_file_path)
        print(f"Processed {file_path} and saved results to {output_file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Inference using Metal3D for multiple CIF files based on FASTA input"
    )
    parser.add_argument(
        "--id", nargs="+", help="indexes of CA atoms to predict", type=int, default=[]
    )
    parser.add_argument(
        "--metalbinding",
        help="uses all residues that have sidechains that can coordinate metals",
        action="store_true",
    )
    parser.add_argument("--writecube", help="write a cube file", action="store_true")
    parser.add_argument(
        "--cubefile", help="name of cube file", default="Metal3D_pred.cube"
    )
    parser.add_argument(
        "--no-clean", help="do not clean the pdb file", action="store_false"
    )
    parser.add_argument("--batch-size", help="batchsize", default=128, type=int)
    parser.add_argument("--writeprobes", help="write probe files", action="store_true")
    parser.add_argument(
        "--probefile",
        help="name of files with predicted metal positions",
        default="probes_Metal3D.pdb",
    )
    parser.add_argument(
        "--probeprobabilities",
        nargs="+",
        help="probabilities to predict probes",
        type=float,
        default=[0.1, 0.2, 0.3, 0.5, 0.75],
    )
    parser.add_argument("--threshold", help="cluster threshold", type=float, default=7)
    parser.add_argument("--pthreshold", help="p threshold", type=float, default=0.10)
    parser.add_argument(
        "--maxp", help="print max probability point", action="store_true"
    )
    parser.add_argument("--label", help="label for the maxp file", default="metal3d")
    parser.add_argument(
        "--softexit",
        help="dont ask for confirmation before exiting",
        action="store_true",
    )
    parser.add_argument(
        "--base_folder", help="Base folder containing ion subfolders", required=True
    )
    parser.add_argument(
        "--device", help="Device to use for computation", default="cuda:0"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Model()
    model.to(device)
    model.load_state_dict(
        torch.load(f"weights/metal_0.5A_v3_d0.2_16Abox.pth", map_location=device)
    )
    model.eval()
    ion_list = ["FE"]
    for ion in ion_list:
        # Read IDs from the FASTA file
        fasta_ids = read_fasta_ids(
            f"/host/protein-metal-ion-binding/multi_modal_binding/data_processing/biolip2_latest/{ion}/{ion}_test_no_label.fasta"
        )
        folder = os.path.join(args.base_folder, ion, "ground_truth")
        output_folder = os.path.join(args.base_folder, ion, "metal3d")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for fasta_id in fasta_ids:
            cif_file = f"{fasta_id}_with_{ion}.cif"
            file_path = os.path.join(folder, cif_file)

            if os.path.exists(file_path):
                process_cif_file(file_path, output_folder, model, device, args)
            else:
                print(f"CIF file not found for ID {fasta_id} and ion {ion}")

    print("--- %s seconds ---" % (time.time() - start_time))

    if not args.softexit:
        input("Press Enter to end program...")
