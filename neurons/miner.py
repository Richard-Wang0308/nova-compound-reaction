import os
import sys
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import sqlite3
import requests
from concurrent.futures import ThreadPoolExecutor

from typing import Any, Dict, List, Optional, Tuple, cast
from types import SimpleNamespace
from collections import defaultdict
from functools import lru_cache

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
import pandas as pd
from rdkit import Chem

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Set DEVICE_OVERRIDE before importing PSICHIC to ensure correct device assignment
# This ensures module-level device variables in PSICHIC are set correctly
os.environ.setdefault('DEVICE_OVERRIDE', 'cuda:0')  # Default to GPU 0 for PSICHIC

from config.config_loader import load_config
from utils import (
    get_sequence_from_protein_code,
    get_protein_treat_model,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    compute_maccs_entropy,
    is_reaction_allowed,
)
from PSICHIC.wrapper import PsichicWrapper
from btdr import QuicknetBittensorDrandTimelock
from combinatorial_db.reactions import get_reaction_info, get_smiles_from_reaction
from boltz.wrapper import BoltzWrapper
import torch
import gc

# ----------------------------------------------------------------------------
# 1. CONFIG & ARGUMENT PARSING
# ----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments and merges with config defaults.

    Returns:
        argparse.Namespace: The combined configuration object.
    """
    parser = argparse.ArgumentParser()
    # Add override arguments for network.
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    # Adds override arguments for netuid.
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    # Bittensor standard argument additions.
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    # Parse combined config
    config = bt.config(parser)

    # Load protein selection params
    config.update(load_config())

    # Final logging dir
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    # Ensure the logging directory exists.
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """
    Constructs the path for GitHub operations from environment variables.
    
    Returns:
        str: The fully qualified GitHub path (owner/repo/branch/path).
    Raises:
        ValueError: If the final path exceeds 100 characters.
    """
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')  # e.g., "nova"
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')  # e.g., "main"
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')  # e.g., "metanova-labs"
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')  # e.g., "data/results" or ""

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path


# ----------------------------------------------------------------------------
# 2. LOGGING SETUP
# ----------------------------------------------------------------------------

def setup_logging(config: argparse.Namespace) -> None:
    """
    Sets up Bittensor logging.

    Args:
        config (argparse.Namespace): The miner configuration object.
    """
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)


# ----------------------------------------------------------------------------
# 3. BITTENSOR & NETWORK SETUP
# ----------------------------------------------------------------------------

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """
    Initializes wallet, subtensor, and metagraph. Fetches the epoch length
    and calculates the miner UID.

    Args:
        config (argparse.Namespace): The miner configuration object.

    Returns:
        tuple: A 5-element tuple of
            (wallet, subtensor, metagraph, miner_uid, epoch_length).
    """
    bt.logging.info("Setting up Bittensor objects.")

    # Initialize wallet
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # Initialize subtensor (asynchronously)
    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            bt.logging.info(f"Connected to subtensor network: {config.network}")
            
            # Sync metagraph
            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info(f"Metagraph synced successfully.")

            bt.logging.info(f"Subtensor: {subtensor}")
            bt.logging.info(f"Metagraph synced: {metagraph}")

            # Get miner UID
            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            # Query epoch length
            node = SubstrateInterface(url=config.network)
            # Set epoch_length to tempo + 1
            epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
            bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error("Please check your network connection and the subtensor network status")
        raise

# ----------------------------------------------------------------------------
# 4. COMBINATORIAL GENERATION & EVOLUTIONARY STRATEGY
# ----------------------------------------------------------------------------

def get_db_path() -> str:
    """Get the path to the combinatorial database."""
    db_path = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
    
    # Check if database exists and is not empty
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        bt.logging.error(f"Database file not found or empty at {db_path}")
        bt.logging.info("Attempting to download database from HuggingFace...")
        try:
            import requests
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = "https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite"
            response = requests.get(db_url, timeout=30)
            if response.status_code == 200:
                with open(db_path, 'wb') as f:
                    f.write(response.content)
                bt.logging.info(f"Database downloaded successfully to {db_path}")
            else:
                bt.logging.error(f"Failed to download database: HTTP {response.status_code}")
        except Exception as e:
            bt.logging.error(f"Error downloading database: {e}")
            bt.logging.error("Please ensure the database file exists at: combinatorial_db/molecules.sqlite")
    
    return db_path


@lru_cache(maxsize=32)  # Reduced from 1024 to prevent OOM - cache only recent role queries
def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    """
    Get all molecules that have the specified role_mask.

    Args:
        role_mask: The role mask to filter by
        db_path: Path to the molecules database

    Returns:
        List of tuples (mol_id, smiles, role_mask) for molecules that match the role
    """
    try:
        import sqlite3
        abs_db_path = os.path.abspath(db_path)
        conn = sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
            (role_mask, role_mask)
        )
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []


def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """
    Build component weights based on scores of molecules containing them.
    Returns dict with 'A', 'B', 'C' keys mapping to {component_id: weight}
    """
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    # Extract component IDs and scores
    # Support both 'boltz_score' (final scorer) and 'combined_score' (legacy)
    score_column = 'boltz_score' if 'boltz_score' in top_pool.columns else 'combined_score'
    for _, row in top_pool.iterrows():
        name = row['product_name']
        score = row[score_column]
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += max(0, score)  # Only positive contributions
                weights['B'][B_id] += max(0, score)
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += max(0, score)
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    # Normalize by count and add smoothing
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                weights[role][comp_id] = weights[role][comp_id] / counts[role][comp_id] + 0.1  # Smoothing
    
    return weights


def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.7) -> pd.DataFrame:
    """
    Select diverse elite molecules: top by score, but ensure diversity in component space.
    """
    if top_pool.empty or n_elites <= 0:
        return pd.DataFrame()
    
    # Take top candidates (more than needed for diversity filtering)
    top_candidates = top_pool.head(min(len(top_pool), n_elites * 3))
    if len(top_candidates) <= n_elites:
        return top_candidates
    
    # Score threshold: at least min_score_ratio of max score
    # Support both 'boltz_score' (final scorer) and 'combined_score' (legacy)
    score_column = 'boltz_score' if 'boltz_score' in top_candidates.columns else 'combined_score'
    max_score = top_candidates[score_column].max()
    threshold = max_score * min_score_ratio
    candidates = top_candidates[top_candidates[score_column] >= threshold]
    
    # Select diverse set: prefer molecules with different components
    selected = []
    used_components = {'A': set(), 'B': set(), 'C': set()}
    
    # First, add top scorer
    if not candidates.empty:
        top_row = candidates.iloc[0]
        selected.append(candidates.index[0])
        parts = top_row['product_name'].split(":")
        if len(parts) >= 4:
            try:
                used_components['A'].add(int(parts[2]))
                used_components['B'].add(int(parts[3]))
                if len(parts) > 4:
                    used_components['C'].add(int(parts[4]))
            except (ValueError, IndexError):
                pass
    
    # Then add diverse molecules
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx in selected:
            continue
        
        parts = row['product_name'].split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                C_id = int(parts[4]) if len(parts) > 4 else None
                
                # Prefer molecules with new components
                is_diverse = (A_id not in used_components['A'] or 
                             B_id not in used_components['B'] or
                             (C_id is not None and C_id not in used_components['C']))
                
                if is_diverse or len(selected) < n_elites * 0.5:  # Always take some top ones
                    selected.append(idx)
                    used_components['A'].add(A_id)
                    used_components['B'].add(B_id)
                    if C_id is not None:
                        used_components['C'].add(C_id)
            except (ValueError, IndexError):
                # If parsing fails, just add it
                if len(selected) < n_elites:
                    selected.append(idx)
    
    # Fill remaining slots with top scorers
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)
    
    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)


def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple], 
                               molecules_C: List[Tuple], is_three_component: bool, 
                               component_weights: dict = None) -> List[str]:
    """Generate molecules by combining components from pools."""
    A_ids = [a[0] for a in molecules_A]
    B_ids = [b[0] for b in molecules_B]
    C_ids = [c[0] for c in molecules_C] if is_three_component else None

    # Use weighted sampling if component weights are provided
    if component_weights:
        # Build weights for each component pool
        weights_A = [component_weights.get('A', {}).get(aid, 1.0) for aid in A_ids]
        weights_B = [component_weights.get('B', {}).get(bid, 1.0) for bid in B_ids]
        weights_C = [component_weights.get('C', {}).get(cid, 1.0) for cid in C_ids] if is_three_component else None
        
        # Normalize weights
        if weights_A:
            sum_w = sum(weights_A)
            weights_A = [w / sum_w if sum_w > 0 else 1.0/len(weights_A) for w in weights_A]
        if weights_B:
            sum_w = sum(weights_B)
            weights_B = [w / sum_w if sum_w > 0 else 1.0/len(weights_B) for w in weights_B]
        if weights_C:
            sum_w = sum(weights_C)
            weights_C = [w / sum_w if sum_w > 0 else 1.0/len(weights_C) for w in weights_C]
        
        picks_A = random.choices(A_ids, weights=weights_A, k=n) if weights_A else random.choices(A_ids, k=n)
        picks_B = random.choices(B_ids, weights=weights_B, k=n) if weights_B else random.choices(B_ids, k=n)
        if is_three_component:
            picks_C = random.choices(C_ids, weights=weights_C, k=n) if weights_C else random.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    else:
        # Uniform random sampling
        picks_A = random.choices(A_ids, k=n)
        picks_B = random.choices(B_ids, k=n)
        if is_three_component:
            picks_C = random.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]

    names = list(dict.fromkeys(names))  # Remove duplicates
    return names


def _parse_components(name: str) -> tuple:
    """Parse component IDs from molecule name."""
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    try:
        A = int(parts[2])
        B = int(parts[3])
        C = int(parts[4]) if len(parts) > 4 else None
        return A, B, C
    except (ValueError, IndexError):
        return None, None, None


def generate_offspring_from_elites(rxn_id: int, n: int, elite_names: list[str],
                                   molecules_A, molecules_B, molecules_C, is_three_component: bool,
                                   mutation_prob: float = 0.1,
                                   avoid_names: set[str] = None) -> list[str]:
    """Generate offspring molecules from elite parents with mutation."""
    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    for name in elite_names:
        A, B, C = _parse_components(name)
        if A is not None: elite_As.add(A)
        if B is not None: elite_Bs.add(B)
        if C is not None and is_three_component: elite_Cs.add(C)

    pool_A_ids = [a[0] for a in molecules_A]
    pool_B_ids = [b[0] for b in molecules_B]
    pool_C_ids = [c[0] for c in molecules_C] if is_three_component else []

    out = []
    local_names = set()
    for _ in range(n):
        use_mutA = (not elite_As) or (random.random() < mutation_prob)
        use_mutB = (not elite_Bs) or (random.random() < mutation_prob)
        use_mutC = (not elite_Cs) or (random.random() < mutation_prob)

        A = random.choice(pool_A_ids) if use_mutA else random.choice(list(elite_As))
        B = random.choice(pool_B_ids) if use_mutB else random.choice(list(elite_Bs))
        if is_three_component:
            C = random.choice(pool_C_ids) if use_mutC else random.choice(list(elite_Cs))
            name = f"rxn:{rxn_id}:{A}:{B}:{C}"
        else:
            name = f"rxn:{rxn_id}:{A}:{B}"

        if avoid_names and name in avoid_names:
            continue
        if name in local_names:
            continue

        out.append(name)
        local_names.add(name)
        if avoid_names is not None:
            avoid_names.add(name)
    return out


def generate_valid_molecules_batch(rxn_ids: List[int], n_samples_per_reaction: int, db_path: str, config: Any,
                                 batch_size: int = 400,  # Increased for RTX 3090 efficiency
                                 elite_names: Dict[int, list[str]] = None, elite_frac: float = 0.5, 
                                 mutation_prob: float = 0.1,
                                 seen_inchikeys: set[str] = None, 
                                 component_weights: Dict[int, dict] = None) -> dict:
    """
    Generate valid molecules combinatorially with validation for multiple reactions.

    Args:
        rxn_ids: List of reaction IDs to generate molecules for
        n_samples_per_reaction: Number of samples to generate per reaction
        elite_names: Dict mapping rxn_id to list of elite molecule names
        component_weights: Dict mapping rxn_id to component weights dict
    """
    all_valid_molecules = []
    all_valid_smiles = []
    seen_keys = seen_inchikeys.copy() if seen_inchikeys else set()
    
    # Generate molecules for each reaction
    for rxn_id in rxn_ids:
        reaction_info = get_reaction_info(rxn_id, db_path)
        if not reaction_info:
            bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
            continue
        
        smarts, roleA, roleB, roleC = reaction_info
        is_three_component = roleC is not None and roleC != 0
        
        molecules_A = get_molecules_by_role(roleA, db_path)
        molecules_B = get_molecules_by_role(roleB, db_path)
        molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []
        
        if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
            bt.logging.error(f"No molecules found for rxn_id {rxn_id}, roles A={roleA}, B={roleB}, C={roleC}")
            continue

        valid_molecules = []
        valid_smiles = []
        rxn_elite_names = elite_names.get(rxn_id) if elite_names else None
        rxn_component_weights = component_weights.get(rxn_id) if component_weights else None
        
        while len(valid_molecules) < n_samples_per_reaction:
            needed = n_samples_per_reaction - len(valid_molecules)
            batch_size_actual = min(max(batch_size, 400), needed * 2)  # Increased for RTX 3090
            
            emitted_names = set()
            if rxn_elite_names:
                n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
                n_rand = batch_size_actual - n_elite

                elite_batch = generate_offspring_from_elites(
                    rxn_id=rxn_id,
                    n=n_elite,
                    elite_names=rxn_elite_names,
                    molecules_A=molecules_A,
                    molecules_B=molecules_B,
                    molecules_C=molecules_C,
                    is_three_component=is_three_component,
                    mutation_prob=mutation_prob,
                    avoid_names=emitted_names,
                )
                emitted_names.update(elite_batch)

                rand_batch = generate_molecules_from_pools(
                    rxn_id, n_rand, molecules_A, molecules_B, molecules_C, is_three_component, rxn_component_weights
                )
                rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
                batch_molecules = elite_batch + rand_batch
            else:
                batch_molecules = generate_molecules_from_pools(
                    rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, rxn_component_weights
                )
            
            # Validate molecules (CPU-intensive RDKit operations)
            for idx, name in enumerate(batch_molecules):
                try:
                    smiles = get_smiles_from_reaction(name)
                    if not smiles:
                        continue
                    
                    if get_heavy_atom_count(smiles) < config.min_heavy_atoms:
                        continue
                    
                    # Check InChIKey for uniqueness
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        continue
                    key = Chem.MolToInchiKey(mol)
                    if key in seen_keys:
                        continue
                    
                    seen_keys.add(key)
                    valid_molecules.append(name)
                    valid_smiles.append(smiles)
                    
                    if len(valid_molecules) >= n_samples_per_reaction:
                        break
                except Exception:
                    continue
                
                # Yield every 100 molecules to prevent CPU spinning (if called from async context)
                # Note: This function is synchronous, but we can't await here
                # The caller should handle this by running in executor or batching
        
        all_valid_molecules.extend(valid_molecules)
        all_valid_smiles.extend(valid_smiles)
    
    return {"molecules": all_valid_molecules, "smiles": all_valid_smiles}


# ----------------------------------------------------------------------------
# 5. INFERENCE AND SUBMISSION LOGIC - TWO-STAGE PIPELINE
# ----------------------------------------------------------------------------

def setup_gpu_devices():
    """
    Setup GPU device assignments for two-stage pipeline.
    
    Note: For proper GPU assignment, set CUDA_VISIBLE_DEVICES before running:
    - For PSICHIC on GPU 0 and Boltz-2 on GPU 1, the system should see both GPUs
    - Boltz-2 device selection may require setting device in predict() call or
      using torch.cuda.set_device(1) before Boltz-2 operations
    """
    # Set PSICHIC to use GPU 0
    os.environ['DEVICE_OVERRIDE'] = 'cuda:0'
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            bt.logging.warning(f"Only {gpu_count} GPU(s) available. Two-stage pipeline works best with 2 GPUs.")
            bt.logging.warning("Both stages may run on the same GPU, which could cause performance issues.")
        else:
            bt.logging.info(f"GPU setup: PSICHIC -> GPU 0, Boltz-2 -> GPU 1 ({gpu_count} GPUs available)")
            # Log GPU memory info
            for i in range(min(2, gpu_count)):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                bt.logging.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        bt.logging.error("CUDA not available. Two-stage pipeline requires GPU support.")


async def stage1_psichic_prefilter(
    state: Dict[str, Any],
    psichic_queue: asyncio.Queue,
    db_path: str,
    n_samples_per_reaction: int,
    elite_names_dict: Dict[int, list],
    component_weights_dict: Dict[int, dict],
    elite_frac: float,
    mutation_prob: float,
    seen_inchikeys: set,
    iteration: int
) -> None:
    """
    Stage 1: PSICHIC pre-filter on GPU 0.
    Scores large batches (10K-200K molecules) and sends top 10% of each batch to Boltz-2.
    Does not maintain a top pool - that's handled by Boltz-2.
    """
    # Generate molecules combinatorially (CPU-intensive - run in executor to prevent blocking)
    # This prevents the CPU-intensive RDKit operations from blocking the async event loop
    loop = asyncio.get_event_loop()
    executor = state.get('molecule_gen_executor', None)
    if executor is None:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="molecule-gen")
        state['molecule_gen_executor'] = executor
    
    sampler_data = await loop.run_in_executor(
        executor,
        generate_valid_molecules_batch,
        [4, 5],
        n_samples_per_reaction,
        db_path,
        state['config'],
        400,  # batch_size
        elite_names_dict,
        elite_frac,
        mutation_prob,
        seen_inchikeys,
        component_weights_dict
    )
    
    # Yield control after CPU-intensive generation
    await asyncio.sleep(0.01)
    
    if not sampler_data or not sampler_data.get("molecules"):
        return
    
    filtered_names = sampler_data["molecules"]
    filtered_smiles = sampler_data["smiles"]
    
    # Count molecules per reaction for logging
    rxn4_count = sum(1 for name in filtered_names if name.startswith("rxn:4:"))
    rxn5_count = sum(1 for name in filtered_names if name.startswith("rxn:5:"))
    bt.logging.info(f"Stage 1 (PSICHIC) Iteration {iteration}: Processing {len(filtered_names)} molecules (rxn4: {rxn4_count}, rxn5: {rxn5_count})")
    
    # PSICHIC scoring on GPU 0 - process incrementally and send top 10% of each batch to Boltz-2
    PSICHIC_BATCH_SIZE = 512  # Large batch size for RTX 3090 GPU 0 (optimized for 2x RTX 3090)
    TOP_PERCENTAGE_FOR_BOLTZ = 0.10  # Send top 10% of each batch to Boltz-2 immediately
    
    # Initialize models for all proteins first
    for target_protein in state['current_challenge_targets']:
        if target_protein not in state['psichic_models']:
            try:
                treat_model = get_protein_treat_model(target_protein)
                treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                target_sequence = get_sequence_from_protein_code(target_protein)
                model = PsichicWrapper(model_path=treat_model_path)
                model.run_challenge_start(target_sequence)
                state['psichic_models'][target_protein] = model
                bt.logging.info(f"Initialized PSICHIC model ({treat_model}) for target: {target_protein} on GPU 0")
            except Exception as e:
                bt.logging.error(f"Error initializing PSICHIC model for target {target_protein}: {e}")
                continue

    for antitarget_protein in state['current_challenge_antitargets']:
        if antitarget_protein not in state['psichic_models']:
            try:
                treat_model = get_protein_treat_model(antitarget_protein)
                treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                model = PsichicWrapper(model_path=treat_model_path)
                model.run_challenge_start(antitarget_sequence)
                state['psichic_models'][antitarget_protein] = model
                bt.logging.info(f"Initialized PSICHIC model ({treat_model}) for antitarget: {antitarget_protein} on GPU 0")
            except Exception as e:
                bt.logging.error(f"Error initializing PSICHIC model for antitarget {antitarget_protein}: {e}")
                continue
    
    # Process molecules in batches and send top 10% of each batch to Boltz-2 immediately
    all_batch_dfs = []
    batch_count = 0
    
    for batch_start in range(0, len(filtered_smiles), PSICHIC_BATCH_SIZE):
        batch_end = min(batch_start + PSICHIC_BATCH_SIZE, len(filtered_smiles))
        batch_smiles = filtered_smiles[batch_start:batch_end]
        batch_names = filtered_names[batch_start:batch_end]
        batch_count += 1
        
        # Score this batch against all target proteins
        batch_target_scores = []
        for target_protein in state['current_challenge_targets']:
            if target_protein in state['psichic_models']:
                batch_results = state['psichic_models'][target_protein].run_validation(list(batch_smiles))
                batch_target_scores.append(batch_results[state['psichic_result_column_name']].tolist())
        
        # Score this batch against all antitarget proteins
        batch_antitarget_scores = []
        for antitarget_protein in state['current_challenge_antitargets']:
            if antitarget_protein in state['psichic_models']:
                batch_results = state['psichic_models'][antitarget_protein].run_validation(list(batch_smiles))
                batch_antitarget_scores.append(batch_results[state['psichic_result_column_name']].tolist())
        
        # Calculate PSICHIC scores for this batch
        batch_df = pd.DataFrame({
            'product_name': batch_names,
            'smiles': batch_smiles
        })
        if batch_target_scores:
            batch_df['target_affinity'] = pd.DataFrame(batch_target_scores).mean(axis=0)
        else:
            batch_df['target_affinity'] = -math.inf
        if batch_antitarget_scores:
            batch_df['antitarget_affinity'] = pd.DataFrame(batch_antitarget_scores).mean(axis=0)
        else:
            batch_df['antitarget_affinity'] = math.inf
        batch_df['psichic_score'] = batch_df['target_affinity'] - state['config'].antitarget_weight * batch_df['antitarget_affinity']
        
        # Send top 10% of this batch to Boltz-2 immediately (concurrent processing)
        batch_df_sorted = batch_df.sort_values(by="psichic_score", ascending=False)
        n_candidates_to_send = max(1, int(len(batch_df_sorted) * TOP_PERCENTAGE_FOR_BOLTZ))
        top_batch_candidates = batch_df_sorted.head(n_candidates_to_send)
        
        if not top_batch_candidates.empty:
            try:
                queue_size_before = psichic_queue.qsize()
                await psichic_queue.put({
                    'molecules': top_batch_candidates[['product_name', 'smiles', 'psichic_score']].to_dict('records'),
                    'iteration': iteration,
                    'batch': batch_count
                })
                queue_size_after = psichic_queue.qsize()
                bt.logging.info(f"Stage 1: Sent top {len(top_batch_candidates)} candidates (top {TOP_PERCENTAGE_FOR_BOLTZ*100:.0f}% of batch {batch_count}/{len(filtered_smiles)//PSICHIC_BATCH_SIZE + 1}) to Boltz-2 queue (queue size: {queue_size_before} -> {queue_size_after})")
            except Exception as e:
                bt.logging.error(f"Error sending incremental candidates to Boltz-2 queue: {e}")
                traceback.print_exc()
        
        # Periodic memory cleanup after each batch to prevent OOM
        if batch_count % 10 == 0:  # Every 10 batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # No need to maintain PSICHIC top pool - Boltz-2 maintains the final top pool


async def stage2_boltz_scorer(
    state: Dict[str, Any],
    psichic_queue: asyncio.Queue,
    boltz_wrapper: BoltzWrapper
) -> pd.DataFrame:
    """
    Stage 2: Boltz-2 final scorer on GPU 1.
    Scores top candidates from PSICHIC (500-2000 molecules) with smaller batches.
    """
    BOLTZ_BATCH_SIZE = 50  # Smaller batch size for Boltz-2 to avoid OOM on GPU 1 (RTX 3090)
    # Boltz-2 is memory-intensive due to 3D attention layers, so use conservative batch size
    MINER_UID = 0  # Dummy UID for miner's own molecules
    
    # Set GPU 1 for Boltz-2 (this should be set before BoltzWrapper initialization)
    # Note: CUDA_VISIBLE_DEVICES must be set before importing torch in Boltz
    # For now, we'll rely on device parameter in predict function if available
    
    while not state['shutdown_event'].is_set():
        try:
            # Get candidates from PSICHIC queue
            try:
                candidate_data = await asyncio.wait_for(psichic_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # Sleep to prevent CPU spinning when queue is empty
                await asyncio.sleep(0.5)
                continue
            
            if not candidate_data or not candidate_data.get('molecules'):
                continue
            
            candidates = candidate_data['molecules']
            iteration = candidate_data.get('iteration', 0)
            batch_num = candidate_data.get('batch', 'unknown')
            bt.logging.info(f"Stage 2 (Boltz-2): Received {len(candidates)} candidates from PSICHIC batch {batch_num} (iteration {iteration}) - starting concurrent processing on GPU 1")
            
            # Process in batches to avoid GPU OOM
            all_boltz_scores = []
            all_names = []
            all_smiles = []
            
            for batch_start in range(0, len(candidates), BOLTZ_BATCH_SIZE):
                batch_end = min(batch_start + BOLTZ_BATCH_SIZE, len(candidates))
                batch_candidates = candidates[batch_start:batch_end]
                
                # Prepare data for Boltz-2
                batch_names = [c['product_name'] for c in batch_candidates]
                batch_smiles = [c['smiles'] for c in batch_candidates]
                
                # Create valid_molecules_by_uid format for Boltz-2
                # Format: {uid: {'smiles': [...], 'names': [...]}}
                valid_molecules_for_boltz = {
                    MINER_UID: {
                        'smiles': batch_smiles,
                        'names': batch_names
                    }
                }
                
                # Create score_dict structure
                score_dict_boltz = {
                    MINER_UID: {
                        'boltz_score': None,
                        'entropy_boltz': None
                    }
                }
                
                # Score with Boltz-2 on GPU 1
                try:
                    # Note: Device assignment is handled by BoltzWrapper(device_id=1)
                    # PyTorch Lightning will use GPU 1 via devices=[1] parameter
                    final_block_hash = "0x" + hashlib.sha256(str(iteration).encode()).hexdigest()[:64]
                    
                    # Create modified config dict for Boltz-2 to score all molecules in batch
                    # Override num_molecules_boltz to score all molecules (not just 1)
                    # Boltz expects a dict with specific keys (same format as validator uses)
                    config_obj = state['config']
                    boltz_config = {
                        'weekly_target': getattr(config_obj, 'weekly_target', None),
                        'num_antitargets': getattr(config_obj, 'num_antitargets', 1),
                        'binding_pocket': getattr(config_obj, 'binding_pocket', None),
                        'max_distance': getattr(config_obj, 'max_distance', None),
                        'force': getattr(config_obj, 'force', False),
                        'num_molecules_boltz': len(batch_smiles),  # Override: score all molecules in batch
                        'boltz_metric': getattr(config_obj, 'boltz_metric', 'affinity_probability_binary'),
                        'sample_selection': 'first',  # Use 'first' to get all molecules
                    }
                    
                    # Score molecules with Boltz-2 (same method as validator uses)
                    # Run in thread pool executor to avoid blocking async event loop
                    # This allows PSICHIC to continue processing next batch while Boltz-2 runs on GPU 1
                    loop = asyncio.get_event_loop()
                    executor = state.get('boltz_executor', ThreadPoolExecutor(max_workers=1))
                    state['boltz_executor'] = executor  # Store for reuse
                    
                    # Run blocking Boltz-2 call in executor (non-blocking for async loop)
                    await loop.run_in_executor(
                        executor,
                        boltz_wrapper.score_molecules_target,
                        valid_molecules_for_boltz,
                        score_dict_boltz,
                        boltz_config,
                        final_block_hash
                    )
                    
                    # Get per-molecule Boltz scores from wrapper
                    per_molecule_scores = boltz_wrapper.per_molecule_metric.get(MINER_UID, {})
                    
                    # Store results - use per-molecule scores if available, otherwise use average
                    for i, (name, smiles) in enumerate(zip(batch_names, batch_smiles)):
                        all_names.append(name)
                        all_smiles.append(smiles)
                        # Get per-molecule score if available, otherwise use average
                        if smiles in per_molecule_scores:
                            all_boltz_scores.append(float(per_molecule_scores[smiles]))
                        else:
                            avg_score = score_dict_boltz[MINER_UID].get('boltz_score', -math.inf)
                            all_boltz_scores.append(float(avg_score) if avg_score != -math.inf else -math.inf)
                    
                    # Clear GPU memory after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()  # Force Python garbage collection to free RAM (prevent OOM)
                    
                    bt.logging.debug(f"Boltz-2 batch {batch_start//BOLTZ_BATCH_SIZE + 1}: Scored {len(batch_smiles)} molecules")
                    
                except Exception as e:
                    bt.logging.error(f"Error in Boltz-2 scoring batch: {e}")
                    traceback.print_exc()
                    # Continue with next batch - assign -inf scores for failed batch
                    for name, smiles in zip(batch_names, batch_smiles):
                        all_names.append(name)
                        all_smiles.append(smiles)
                        all_boltz_scores.append(-math.inf)
                    continue
            
            if not all_names:
                continue
            
            # Create DataFrame with Boltz-2 scores
            boltz_df = pd.DataFrame({
                'product_name': all_names,
                'smiles': all_smiles,
                'boltz_score': all_boltz_scores
            })
            
            # Update Boltz-2 top pool (this is the final pool for submission)
            # Use state to share the pool
            current_pool = state.get('boltz_top_pool', pd.DataFrame(columns=["product_name", "smiles", "boltz_score"]))
            updated_pool = pd.concat([current_pool, boltz_df])
            updated_pool = updated_pool.drop_duplicates(subset=["product_name"], keep="first")
            updated_pool = updated_pool.sort_values(by="boltz_score", ascending=False)
            BOLTZ_TOP_POOL_SIZE = 100  # Final top pool based on Boltz-2 scores
            updated_pool = updated_pool.head(BOLTZ_TOP_POOL_SIZE)
            state['boltz_top_pool'] = updated_pool  # Update shared pool
            
            # Update best candidate based on Boltz-2 score
            if not updated_pool.empty:
                top_molecule = updated_pool.iloc[0]
                final_score = top_molecule['boltz_score']
                
                if final_score > state['best_score']:
                    state['best_score'] = final_score
                    candidate_name = top_molecule['product_name']
                    state['candidate_product'] = candidate_name
                    reaction_type = candidate_name.split(":")[1] if ":" in candidate_name else "unknown"
                    bt.logging.info(f"Stage 2: New best Boltz-2 score: {final_score:.4f}, Candidate: {candidate_name}, Reaction: {reaction_type}")
                    bt.logging.info(f"Boltz-2 top pool stats - Avg: {updated_pool['boltz_score'].mean():.4f}, Max: {updated_pool['boltz_score'].max():.4f}")
            
            await asyncio.sleep(0.1)  # Small delay to prevent tight loop
            
        except Exception as e:
            bt.logging.error(f"Error in Boltz-2 scorer: {e}")
            traceback.print_exc()
            await asyncio.sleep(1)
    
    return state.get('boltz_top_pool', pd.DataFrame())


async def run_two_stage_pipeline(state: Dict[str, Any]) -> None:
    """
    Two-stage scoring pipeline:
    - Stage 1 (PSICHIC on GPU 0): Pre-screen large batches (10K-200K molecules)
    - Stage 2 (Boltz-2 on GPU 1): Score top candidates (500-2000) for final ranking
    
    Args:
        state (dict): A shared state dict containing references to:
            'psichic_models', 'current_challenge_targets', 'current_challenge_antitargets',
            'psichic_result_column_name', 'best_score', 'candidate_product',
            'last_submitted_product', 'shutdown_event', etc.
    """
    bt.logging.info("Starting two-stage scoring pipeline: PSICHIC (GPU 0) -> Boltz-2 (GPU 1)")
    setup_gpu_devices()
    
    # Only reactions 4 and 5 are allowed - generate molecules for both
    allowed_reactions = [4, 5]
    db_path = get_db_path()
    
    # Verify database is accessible
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        bt.logging.error(f"Database file is missing or empty at {db_path}. Cannot continue.")
        state['shutdown_event'].set()
        return
    
    # Initialize Boltz-2 wrapper (reused across iterations)
    # Pass device_id=1 to use GPU 1 (PyTorch Lightning will handle device assignment)
    # Note: CUDA_VISIBLE_DEVICES="0,1" must be set before running (both GPUs visible)
    bt.logging.info("Initializing Boltz-2 wrapper for GPU 1...")
    boltz_wrapper = BoltzWrapper(device_id=1)  # Use GPU 1 for Boltz-2
    state['boltz_wrapper'] = boltz_wrapper
    bt.logging.info("Boltz-2 wrapper initialized (will use GPU 1 via PyTorch Lightning devices=[1])")
    
    # Create thread pool executor for running Boltz-2 (blocking calls) without blocking async loop
    # This allows PSICHIC to continue processing while Boltz-2 runs concurrently on GPU 1
    state['boltz_executor'] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="boltz-scorer")
    bt.logging.info("Boltz-2 thread pool executor created for concurrent processing")
    
    # Initialize evolutionary strategy state
    # Use Boltz-2 top pool for genetic algorithm (since Boltz-2 is the final scorer)
    boltz_top_pool = pd.DataFrame(columns=["product_name", "smiles", "boltz_score"])
    state['boltz_top_pool'] = boltz_top_pool  # Share via state for updates
    seen_inchikeys = set()
    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.25
    prev_avg_score = None
    score_improvement_rate = 0.0
    
    # Two-stage pipeline hyperparameters
    # Stage 1: Generate and pre-screen with PSICHIC
    # Generate 10K-50K molecules per iteration for PSICHIC pre-filtering (adjustable based on GPU capacity)
    n_samples_per_reaction = 10000  # 10K per reaction = 20K total for PSICHIC pre-filtering (optimized for 2x RTX 3090)
    
    # Queue for communication between stages
    psichic_queue = asyncio.Queue(maxsize=10)  # Queue for passing candidates to Boltz-2
    
    # Start Stage 2 (Boltz-2 scorer) as background task
    boltz_task = asyncio.create_task(
        stage2_boltz_scorer(state, psichic_queue, boltz_wrapper)
    )
    state['boltz_task'] = boltz_task  # Store in state for cleanup
    bt.logging.info("Stage 2 (Boltz-2) scorer task started on GPU 1")

    while not state['shutdown_event'].is_set():
        try:
            iteration += 1
            bt.logging.debug(f"Two-stage pipeline iteration {iteration}")
            
            # Get current Boltz-2 top pool from state (updated by stage2_boltz_scorer)
            current_boltz_pool = state.get('boltz_top_pool', pd.DataFrame(columns=["product_name", "smiles", "boltz_score"]))
            
            # Build component weights for each reaction from Boltz-2 top pool (final scorer)
            component_weights_rxn4 = build_component_weights(current_boltz_pool, 4) if not current_boltz_pool.empty else None
            component_weights_rxn5 = build_component_weights(current_boltz_pool, 5) if not current_boltz_pool.empty else None
            
            # Select diverse elites from Boltz-2 top pool (final scorer)
            n_elites = min(50, len(current_boltz_pool))  # Use up to 50 elites from top pool
            elite_df = select_diverse_elites(current_boltz_pool, n_elites) if not current_boltz_pool.empty else pd.DataFrame()
            if not elite_df.empty:
                elite_names_rxn4 = [name for name in elite_df["product_name"].tolist() if name.startswith("rxn:4:")]
                elite_names_rxn5 = [name for name in elite_df["product_name"].tolist() if name.startswith("rxn:5:")]
            else:
                elite_names_rxn4 = None
                elite_names_rxn5 = None
            
            # Adaptive sampling: adjust based on Boltz-2 score improvement (final scorer)
            if prev_avg_score is not None and not current_boltz_pool.empty:
                current_avg = current_boltz_pool['boltz_score'].mean()
                score_improvement_rate = (current_avg - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
                
                # If improving well, increase exploitation; if stagnating, increase exploration
                if score_improvement_rate > 0.01:  # Good improvement
                    elite_frac = min(0.7, elite_frac * 1.1)
                    mutation_prob = max(0.05, mutation_prob * 0.95)
                elif score_improvement_rate < -0.01:  # Declining
                    elite_frac = max(0.2, elite_frac * 0.9)
                    mutation_prob = min(0.4, mutation_prob * 1.1)
            
            # Stage 1: Generate and pre-screen with PSICHIC
            elite_names_dict = {4: elite_names_rxn4, 5: elite_names_rxn5} if elite_names_rxn4 or elite_names_rxn5 else None
            component_weights_dict = {4: component_weights_rxn4, 5: component_weights_rxn5} if component_weights_rxn4 or component_weights_rxn5 else None
            
            await stage1_psichic_prefilter(
                state=state,
                psichic_queue=psichic_queue,
                db_path=db_path,
                n_samples_per_reaction=n_samples_per_reaction,
                elite_names_dict=elite_names_dict,
                component_weights_dict=component_weights_dict,
                elite_frac=elite_frac,
                mutation_prob=mutation_prob,
                seen_inchikeys=seen_inchikeys,
                iteration=iteration
            )
            
            # Yield control to event loop to prevent CPU spinning
            await asyncio.sleep(0.1)
            
            # Periodic memory cleanup to prevent OOM
            if iteration % 5 == 0:  # Every 5 iterations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Clear LRU cache periodically to free memory
                get_molecules_by_role.cache_clear()
            
            # Update seen InChIKeys from Boltz-2 top pool (final scorer)
            if not current_boltz_pool.empty:
                for smile in current_boltz_pool['smiles'].tolist():
                    try:
                        mol = Chem.MolFromSmiles(smile)
                        if mol:
                            key = Chem.MolToInchiKey(mol)
                            seen_inchikeys.add(key)
                    except Exception:
                        pass
            
            # Track Boltz-2 score improvement (final scorer)
            current_avg_score = current_boltz_pool['boltz_score'].mean() if not current_boltz_pool.empty else None
            if current_avg_score is not None:
                if prev_avg_score is not None:
                    score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
                prev_avg_score = current_avg_score
                
                if not current_boltz_pool.empty:
                    bt.logging.info(f"Boltz-2 top pool - Avg: {current_boltz_pool['boltz_score'].mean():.4f}, Max: {current_boltz_pool['boltz_score'].max():.4f}, Size: {len(current_boltz_pool)}, Improvement: {score_improvement_rate*100:.2f}%")
            
            # Check for submission (based on Boltz-2 scores in state)
                    current_block = await state['subtensor'].get_current_block()
                    next_epoch_block = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
                    blocks_until_epoch = next_epoch_block - current_block
                    
            bt.logging.debug(f"Current block: {current_block}, Blocks until epoch: {blocks_until_epoch}")
            
            if state['candidate_product'] and blocks_until_epoch <= 20:
                bt.logging.info(f"Close to epoch end ({blocks_until_epoch} blocks remaining), attempting submission...")
                if state['candidate_product'] != state['last_submitted_product']:
                    bt.logging.info("Attempting to submit new candidate...")
                    try:
                        await submit_response(state)
                    except Exception as e:
                        bt.logging.error(f"Error submitting response: {e}")
                else:
                    bt.logging.info("Skipping submission - same product as last submission")

            await asyncio.sleep(2)

        except Exception as e:
            bt.logging.error(f"Error in two-stage pipeline: {e}")
            traceback.print_exc()
            state['shutdown_event'].set()
    
    # Cleanup: Cancel Boltz-2 task and clean up resources
    if 'boltz_task' in state and state['boltz_task']:
        if not state['boltz_task'].done():
            state['boltz_task'].cancel()
            try:
                await state['boltz_task']
            except asyncio.CancelledError:
                pass
    
    # Shutdown thread pool executors
    if 'boltz_executor' in state and state.get('boltz_executor'):
        state['boltz_executor'].shutdown(wait=True)
        bt.logging.info("Boltz-2 thread pool executor shut down")
    
    if 'molecule_gen_executor' in state and state.get('molecule_gen_executor'):
        state['molecule_gen_executor'].shutdown(wait=True)
        bt.logging.info("Molecule generation thread pool executor shut down")
    
    if 'boltz_wrapper' in state and state['boltz_wrapper']:
        try:
            state['boltz_wrapper'].cleanup_model()
        except Exception:
            pass


async def submit_response(state: Dict[str, Any]) -> None:
    """
    Encrypts and submits the current candidate product as a chain commitment and uploads
    the encrypted response to GitHub. If the chain accepts the commitment, we finalize it.

    Args:
        state (dict): Shared state dictionary containing references to:
            'bdt', 'miner_uid', 'candidate_product', 'subtensor', 'wallet', 'config',
            'github_path', etc.
    """
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    # Validate that the candidate uses allowed reactions (4 or 5)
    if not (is_reaction_allowed(candidate_product, "rxn:4") or is_reaction_allowed(candidate_product, "rxn:5")):
        bt.logging.warning(f"Candidate product '{candidate_product}' does not use allowed reactions (4 or 5). Skipping submission.")
        return

    bt.logging.info(f"Starting submission process for product: {candidate_product}")
    
    # 1) Encrypt the response
    current_block = await state['subtensor'].get_current_block()
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
    bt.logging.info(f"Encrypted response generated successfully")

    # 2) Create temp file, write content
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, 'w+') as f:
        f.write(str(encrypted_response))
        f.flush()

        # Read, base64-encode
        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()

        # Generate short hash-based filename
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Prepared commit content: {commit_content}")

        # 3) Attempt chain commitment
        bt.logging.info(f"Attempting chain commitment...")
        try: 
            commitment_status = await state['subtensor'].set_commitment(
                wallet=state['wallet'],
                netuid=state['config'].netuid,
                data=commit_content
            )
            bt.logging.info(f"Chain commitment status: {commitment_status}")
        except MetadataError:
            bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
            return

        # 4) If chain commitment success, upload to GitHub
        if commitment_status:
            try:
                bt.logging.info(f"Commitment set successfully for {commit_content}")
                bt.logging.info("Attempting GitHub upload...")
                github_status = upload_file_to_github(filename, encoded_content)
                if github_status:
                    bt.logging.info(f"File uploaded successfully to {commit_content}")
                    state['last_submitted_product'] = candidate_product
                    state['last_submission_time'] = datetime.datetime.now()
                else:
                    bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
            except Exception as e:
                bt.logging.error(f"Failed to upload file for {commit_content}: {e}")


# ----------------------------------------------------------------------------
# 6. MAIN MINING LOOP
# ----------------------------------------------------------------------------

async def run_miner(config: argparse.Namespace) -> None:
    """
    The main mining loop, orchestrating:
      - Bittensor objects initialization
      - Model initialization
      - Fetching new proteins each epoch
      - Running inference and submissions
      - Periodically syncing metagraph

    Args:
        config (argparse.Namespace): The miner configuration object.
    """

    # 1) Setup wallet, subtensor, metagraph, etc.
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)

    # 2) Prepare shared state
    state: Dict[str, Any] = {
        # environment / config
        'config': config,
        'psichic_result_column_name': 'predicted_binding_affinity',
        'submission_interval': 1200,

        # GitHub
        'github_path': load_github_path(),

        # Bittensor
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,

        # Models - one instance per protein
        'psichic_models': {},  # Dictionary mapping protein codes to their PSICHIC instances
        'bdt': QuicknetBittensorDrandTimelock(),

        # Inference state
        'candidate_product': None,
        'best_score': float('-inf'),
        'last_submitted_product': None,
        'last_submission_time': None,
        'shutdown_event': asyncio.Event(),

        # Challenges
        'current_challenge_targets': [],
        'last_challenge_targets': [],
        'current_challenge_antitargets': [],
        'last_challenge_antitargets': [],
    }

    bt.logging.info("Entering main miner loop...")

    # 3) If we start mid-epoch, obtain most recent proteins from block hash
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    next_boundary = last_boundary + epoch_length

    # If we start too close to epoch end, wait for next epoch
    if next_boundary - current_block < 20:
        bt.logging.info(f"Too close to epoch end, waiting for next epoch to start...")
        block_to_check = next_boundary
        await asyncio.sleep(12*10)
    else:
        block_to_check = last_boundary

    block_hash = await subtensor.determine_block_hash(block_to_check)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets
    )

    if startup_proteins:
        state['current_challenge_targets'] = startup_proteins["targets"]
        state['last_challenge_targets'] = startup_proteins["targets"]
        state['current_challenge_antitargets'] = startup_proteins["antitargets"]
        state['last_challenge_antitargets'] = startup_proteins["antitargets"]
        bt.logging.info(f"Startup targets: {startup_proteins['targets']}, antitargets: {startup_proteins['antitargets']}")

        # Initialize models for all proteins
        try:
            for target_protein in startup_proteins["targets"]:
                # Determine which TREAT model to use based on protein type
                treat_model = get_protein_treat_model(target_protein)
                treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                
                target_sequence = get_sequence_from_protein_code(target_protein)
                model = PsichicWrapper(model_path=treat_model_path)
                model.run_challenge_start(target_sequence)
                state['psichic_models'][target_protein] = model
                bt.logging.info(f"Initialized model ({treat_model}) for target: {target_protein}")

            for antitarget_protein in startup_proteins["antitargets"]:
                # Determine which TREAT model to use based on protein type
                treat_model = get_protein_treat_model(antitarget_protein)
                treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                
                antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                model = PsichicWrapper(model_path=treat_model_path)
                model.run_challenge_start(antitarget_sequence)
                state['psichic_models'][antitarget_protein] = model
                bt.logging.info(f"Initialized model ({treat_model}) for antitarget: {antitarget_protein}")
        except Exception as e:
            try:
                os.system(
                    f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/TREAT1/model.pt')} "
                    f"https://huggingface.co/Metanova/TREAT-1/resolve/main/model.pt"
                )
                # Retry initialization after download
                for target_protein in state['current_challenge_targets']:
                    if target_protein not in state['psichic_models']:
                        # Determine which TREAT model to use based on protein type
                        treat_model = get_protein_treat_model(target_protein)
                        treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                        
                        target_sequence = get_sequence_from_protein_code(target_protein)
                        model = PsichicWrapper(model_path=treat_model_path)
                        model.run_challenge_start(target_sequence)
                        state['psichic_models'][target_protein] = model
                        bt.logging.info(f"Initialized model ({treat_model}) for target: {target_protein}")

                for antitarget_protein in state['current_challenge_antitargets']:
                    if antitarget_protein not in state['psichic_models']:
                        # Determine which TREAT model to use based on protein type
                        treat_model = get_protein_treat_model(antitarget_protein)
                        treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                        
                        antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                        model = PsichicWrapper(model_path=treat_model_path)
                        model.run_challenge_start(antitarget_sequence)
                        state['psichic_models'][antitarget_protein] = model
                        bt.logging.info(f"Initialized model ({treat_model}) for antitarget: {antitarget_protein}")
                bt.logging.info("Models re-downloaded and initialized successfully.")
            except Exception as e2:
                bt.logging.error(f"Error initializing models after re-download attempt: {e2}")

        # 4) Launch the inference loop
        try:
            state['inference_task'] = asyncio.create_task(run_two_stage_pipeline(state))
            bt.logging.debug("Inference started on startup proteins.")
        except Exception as e:
            bt.logging.error(f"Error starting inference: {e}")

    # 5) Main epoch-based loop
    while True:
        try:
            current_block = await subtensor.get_current_block()

            # If we are at an epoch boundary, fetch new proteins
            if current_block % epoch_length == 0:
                bt.logging.info(f"Found epoch boundary at block {current_block}.")
                
                block_hash = await subtensor.determine_block_hash(current_block)
                
                new_proteins = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    weekly_target=config.weekly_target,
                    num_antitargets=config.num_antitargets
                )
                if (new_proteins and 
                    (new_proteins["targets"] != state['last_challenge_targets'] or 
                     new_proteins["antitargets"] != state['last_challenge_antitargets'])):
                    state['current_challenge_targets'] = new_proteins["targets"]
                    state['last_challenge_targets'] = new_proteins["targets"]
                    state['current_challenge_antitargets'] = new_proteins["antitargets"]
                    state['last_challenge_antitargets'] = new_proteins["antitargets"]
                    bt.logging.info(f"New proteins - targets: {new_proteins['targets']}, antitargets: {new_proteins['antitargets']}")

                # Cancel old inference, reset relevant state
                if 'inference_task' in state and state['inference_task']:
                    if not state['inference_task'].done():
                        state['shutdown_event'].set()
                        bt.logging.debug("Shutdown event set for old inference task.")
                        await state['inference_task']

                # Reset best score and candidate
                state['candidate_product'] = None
                state['best_score'] = float('-inf')
                state['last_submitted_product'] = None
                state['shutdown_event'] = asyncio.Event()

                # Initialize models for new proteins
                try:
                    for target_protein in state['current_challenge_targets']:
                        if target_protein not in state['psichic_models']:
                            # Determine which TREAT model to use based on protein type
                            treat_model = get_protein_treat_model(target_protein)
                            treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                            
                            target_sequence = get_sequence_from_protein_code(target_protein)
                            model = PsichicWrapper(model_path=treat_model_path)
                            model.run_challenge_start(target_sequence)
                            state['psichic_models'][target_protein] = model
                            bt.logging.info(f"Initialized model ({treat_model}) for target: {target_protein}")

                    for antitarget_protein in state['current_challenge_antitargets']:
                        if antitarget_protein not in state['psichic_models']:
                            # Determine which TREAT model to use based on protein type
                            treat_model = get_protein_treat_model(antitarget_protein)
                            treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                            
                            antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                            model = PsichicWrapper(model_path=treat_model_path)
                            model.run_challenge_start(antitarget_sequence)
                            state['psichic_models'][antitarget_protein] = model
                            bt.logging.info(f"Initialized model ({treat_model}) for antitarget: {antitarget_protein}")
                except Exception as e:
                    try:
                        os.system(
                            f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/TREAT1/model.pt')} "
                            f"https://huggingface.co/Metanova/TREAT-1/resolve/main/model.pt"
                        )
                        # Retry initialization after download
                        for target_protein in state['current_challenge_targets']:
                            if target_protein not in state['psichic_models']:
                                # Determine which TREAT model to use based on protein type
                                treat_model = get_protein_treat_model(target_protein)
                                treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                                
                                target_sequence = get_sequence_from_protein_code(target_protein)
                                model = PsichicWrapper(model_path=treat_model_path)
                                model.run_challenge_start(target_sequence)
                                state['psichic_models'][target_protein] = model
                                bt.logging.info(f"Initialized model ({treat_model}) for target: {target_protein}")

                        for antitarget_protein in state['current_challenge_antitargets']:
                            if antitarget_protein not in state['psichic_models']:
                                # Determine which TREAT model to use based on protein type
                                treat_model = get_protein_treat_model(antitarget_protein)
                                treat_model_path = os.path.join(BASE_DIR, "PSICHIC", "trained_weights", treat_model)
                                
                                antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                                model = PsichicWrapper(model_path=treat_model_path)
                                model.run_challenge_start(antitarget_sequence)
                                state['psichic_models'][antitarget_protein] = model
                                bt.logging.info(f"Initialized model ({treat_model}) for antitarget: {antitarget_protein}")
                        bt.logging.info("Models re-downloaded and initialized successfully.")
                    except Exception as e2:
                        bt.logging.error(f"Error initializing models after re-download attempt: {e2}")

                # Start new inference
                try:
                    state['inference_task'] = asyncio.create_task(run_two_stage_pipeline(state))
                    bt.logging.debug("New inference task started.")
                except Exception as e:
                    bt.logging.error(f"Error starting new inference: {e}")

            # Periodically update our knowledge of the network
            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Number of nodes: {metagraph.n} | "
                    f"Current epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)

            await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            break


# ----------------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------------

async def main() -> None:
    """
    Main entry point for asynchronous execution of the miner logic.
    """
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
