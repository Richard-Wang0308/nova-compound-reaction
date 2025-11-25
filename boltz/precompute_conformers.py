"""
Precompute conformers for Boltz-2 molecules in parallel.
This avoids expensive RDKit conformer generation during inference.
"""
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import bittensor as bt


def compute_conformer_fast(smiles: str, max_attempts: int = 5, max_iters: int = 500) -> Optional[Dict]:
    """
    Compute conformer for a single SMILES string with fast settings.
    
    Args:
        smiles: SMILES string
        max_attempts: Maximum embedding attempts (reduced for speed)
        max_iters: Maximum UFF optimization iterations (reduced for speed)
    
    Returns:
        Dict with coords, atom_types, bonds, and metadata, or None if failed
    """
    try:
        # Parse SMILES
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = AllChem.AddHs(mol)
        
        # Set atom names (required by Boltz)
        canonical_order = AllChem.CanonicalRankAtoms(mol)
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(atom_name) > 4:
                # Skip molecules with atom names > 4 chars
                return None
            atom.SetProp("name", atom_name)
        
        # Generate conformer with fast settings
        options = AllChem.ETKDGv3()
        options.clearConfs = False
        options.maxAttempts = max_attempts  # Reduced from default
        
        conf_id = AllChem.EmbedMolecule(mol, options)
        
        if conf_id == -1:
            # Try with random coords as fallback
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)
            if conf_id == -1:
                return None
        
        # Fast UFF optimization (reduced iterations)
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iters)
        except (RuntimeError, ValueError):
            # Force field issues - use conformer as-is
            pass
        
        # Extract conformer data
        conformer = mol.GetConformer(conf_id)
        num_atoms = mol.GetNumAtoms()
        
        # Get coordinates
        coords = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)], dtype=np.float32)
        
        # Get atom types (atomic numbers)
        atom_types = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
        
        # Get bonds (adjacency list)
        bonds = []
        for bond in mol.GetBonds():
            bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), int(bond.GetBondType())))
        
        # Get molecular weight (for affinity correction)
        mol_no_h = AllChem.RemoveHs(mol, sanitize=False)
        mw = Descriptors.MolWt(mol_no_h)
        
        return {
            "coords": coords,
            "atom_types": atom_types,
            "bonds": bonds,
            "smiles": smiles,
            "mw": mw,
            "num_atoms": num_atoms,
        }
    
    except Exception as e:
        bt.logging.debug(f"Failed to compute conformer for {smiles}: {e}")
        return None


def precompute_conformers_batch(
    items: List[Tuple[str, str]],  # List of (product_name, smiles)
    output_dir: Path,
    max_workers: int = 32,
    shard_size: int = 1000,
    chunk_size: int = 100,
) -> Dict[str, int]:
    """
    Precompute conformers for a batch of molecules in parallel.
    
    Args:
        items: List of (product_name, smiles) tuples
        output_dir: Directory to save precomputed conformers
        max_workers: Number of parallel workers
        shard_size: Number of records per shard file
        chunk_size: Chunk size for ProcessPoolExecutor
    
    Returns:
        Dict with stats: {"success": count, "failed": count, "shards": count}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create index mapping product_name -> shard_id, offset
    index = {}
    shard_id = 0
    shard_records = []
    stats = {"success": 0, "failed": 0, "shards": 0}
    
    bt.logging.info(f"Precomputing conformers for {len(items)} molecules with {max_workers} workers...")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(compute_conformer_fast, smiles): (name, smiles)
            for name, smiles in items
        }
        
        # Process results as they complete
        for future in as_completed(future_to_item):
            name, smiles = future_to_item[future]
            try:
                result = future.result()
                
                if result is not None:
                    # Add metadata
                    result["product_name"] = name
                    shard_records.append(result)
                    index[name] = (shard_id, len(shard_records) - 1)
                    stats["success"] += 1
                    
                    # Write shard when full
                    if len(shard_records) >= shard_size:
                        shard_path = output_dir / f"shard_{shard_id:04d}.pt"
                        torch.save(shard_records, shard_path)
                        bt.logging.debug(f"Wrote shard {shard_id} with {len(shard_records)} records")
                        stats["shards"] += 1
                        shard_records = []
                        shard_id += 1
                else:
                    stats["failed"] += 1
                    
            except Exception as e:
                bt.logging.warning(f"Error processing {name}: {e}")
                stats["failed"] += 1
    
    # Write final shard if any remaining records
    if shard_records:
        shard_path = output_dir / f"shard_{shard_id:04d}.pt"
        torch.save(shard_records, shard_path)
        bt.logging.debug(f"Wrote final shard {shard_id} with {len(shard_records)} records")
        stats["shards"] += 1
    
    # Save index
    index_path = output_dir / "index.pt"
    torch.save(index, index_path)
    
    bt.logging.info(
        f"Precomputation complete: {stats['success']} success, {stats['failed']} failed, "
        f"{stats['shards']} shards written"
    )
    
    return stats


def load_precomputed_conformer(product_name: str, precomputed_dir: Path) -> Optional[Dict]:
    """
    Load a precomputed conformer from shards.
    
    Args:
        product_name: Name of the product/molecule
        precomputed_dir: Directory containing precomputed shards
    
    Returns:
        Dict with conformer data or None if not found
    """
    precomputed_dir = Path(precomputed_dir)
    index_path = precomputed_dir / "index.pt"
    
    if not index_path.exists():
        return None
    
    try:
        # Load index
        index = torch.load(index_path)
        
        if product_name not in index:
            return None
        
        shard_id, offset = index[product_name]
        
        # Load shard
        shard_path = precomputed_dir / f"shard_{shard_id:04d}.pt"
        if not shard_path.exists():
            return None
        
        shard = torch.load(shard_path)
        if offset >= len(shard):
            return None
        
        return shard[offset]
    
    except Exception as e:
        bt.logging.debug(f"Error loading precomputed conformer for {product_name}: {e}")
        return None

