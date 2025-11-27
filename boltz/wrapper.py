import os
import yaml
import sys
import traceback
import json
import numpy as np
import random
import gc
import shutil
import hashlib
import math
import psutil
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Set multiprocessing start method to 'spawn' to avoid shared memory (shm) bus errors
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Start method already set

import torch
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for Tensor Cores
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN
torch.set_float32_matmul_precision("medium")  # Use Tensor Cores for 1.5-2x speedup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)

import bittensor as bt

from src.boltz.main import predict
from utils.proteins import get_sequence_from_protein_code
from utils.molecules import compute_maccs_entropy, is_boltz_safe_smiles
from src.boltz.model.models.boltz2 import Boltz2
from dataclasses import asdict
from src.boltz.main import (
    Boltz2DiffusionParams,
    PairformerArgsV2,
    MSAModuleArgs,
    BoltzSteeringParams,
)
from boltz.precompute_conformers import precompute_conformers_batch, load_precomputed_conformer
from boltz.quantize_model import quantize_model

def _snapshot_rng():
    return {
        "py":  random.getstate(),
        "np":  np.random.get_state(),
        "tc":  torch.random.get_rng_state(),
        "tcu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _restore_rng(snap):
    random.setstate(snap["py"])
    np.random.set_state(snap["np"])
    torch.random.set_rng_state(snap["tc"])
    if snap["tcu"] is not None:
        torch.cuda.set_rng_state_all(snap["tcu"])

def _seed_for_record(rec_id, base_seed):
    h = hashlib.sha256(str(rec_id).encode()).digest()
    return (int.from_bytes(h[:8], "little") ^ base_seed) % (2**31 - 1)

class BoltzWrapper:
    def __init__(self, device_id: int = 0):
        """
        Initialize BoltzWrapper.
        
        Args:
            device_id: GPU device ID to use (0 for GPU 0, 1 for GPU 1, etc.)
                       Default is 0. Set to 1 for GPU 1 in two-stage pipeline.
        """
        config_path = os.path.join(BASE_DIR, "boltz_config.yaml")
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.base_dir = BASE_DIR
        self.device_id = device_id  # Store device ID for use in predict()

        self.tmp_dir = os.path.join(PARENT_DIR, "boltz_tmp_files")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.input_dir = os.path.join(self.tmp_dir, "inputs")
        os.makedirs(self.input_dir, exist_ok=True)

        self.output_dir = os.path.join(self.tmp_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Precomputed conformers directory (for fast conformer loading)
        self.precomputed_dir = os.path.join(self.tmp_dir, "precomputed_conformers")
        os.makedirs(self.precomputed_dir, exist_ok=True)
        
        # Enable precomputation by default (can be disabled for debugging)
        self.use_precomputed_conformers = self.config.get('use_precomputed_conformers', True)

        bt.logging.debug(f"BoltzWrapper initialized with device_id={device_id}")
        self.per_molecule_metric = {}
        
        self.base_seed = 68
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.base_seed)

        self._rng0 = _snapshot_rng()
        bt.logging.debug("BoltzWrapper initialized with deterministic baseline")
        
        # Cache models to avoid reloading every iteration (saves 72.4s per iteration)
        self.structure_model = None
        self.affinity_model = None
        self._models_loaded = False
        
        # Memory monitoring settings
        self.enable_memory_monitoring = self.config.get('enable_memory_monitoring', True)
        self.max_gpu_memory_usage = self.config.get('max_gpu_memory_usage', 0.90)
        self.max_ram_usage = self.config.get('max_ram_usage', 0.85)

    # @profile
    def preprocess_data_for_boltz(self, valid_molecules_by_uid: dict, score_dict: dict, final_block_hash: str) -> None:
        # Get protein sequence
        self.protein_sequence = get_sequence_from_protein_code(self.subnet_config['weekly_target'])

        # Collect all unique molecules across all UIDs
        self.unique_molecules = {}  # {smiles: [(uid, mol_id), ...]}
        
        bt.logging.info("Preprocessing data for Boltz2")
        for uid, valid_molecules in valid_molecules_by_uid.items():
            # Select a subsample of n molecules to score
            if self.subnet_config['sample_selection'] == "random":
                seed = int(final_block_hash[2:], 16) + uid
                rng = random.Random(seed)

                unique_indices = rng.sample(range(len(valid_molecules['smiles'])), 
                                           k=self.subnet_config['num_molecules_boltz'])

                boltz_candidates_smiles = [valid_molecules['smiles'][i] for i in unique_indices]
            elif self.subnet_config['sample_selection'] == "first":
                boltz_candidates_smiles = valid_molecules['smiles'][:self.subnet_config['num_molecules_boltz']]
            else:
                bt.logging.error(f"Invalid sample selection method: {self.subnet_config['sample_selection']}")
                return None

            if self.subnet_config['num_molecules_boltz'] > 1:
                try:
                    score_dict[uid]["entropy_boltz"] = compute_maccs_entropy(boltz_candidates_smiles)
                except Exception as e:
                    bt.logging.error(f"Error computing Boltz subset entropy for UID={uid}: {e}")
                    score_dict[uid]["entropy_boltz"] = None
            else:
                score_dict[uid]["entropy_boltz"] = None

            for smiles in boltz_candidates_smiles:
                ok, reason = is_boltz_safe_smiles(smiles)
                if not ok:
                    bt.logging.warning(f"Skipping Boltz candidate {smiles} because it is not parseable: {reason}")
                    continue
                if smiles not in self.unique_molecules:
                    self.unique_molecules[smiles] = []
                rec_id = smiles + self.protein_sequence #+ final_block_hash
                mol_idx = _seed_for_record(rec_id, self.base_seed)

                self.unique_molecules[smiles].append((uid, mol_idx))
        bt.logging.info(f"Unique Boltz candidates: {self.unique_molecules}")

        bt.logging.info(f"Writing {len(self.unique_molecules)} unique molecules to input directory")
        
        # Prepare items for precomputation: (product_name, smiles)
        items_for_precompute = []
        for smiles, ids in self.unique_molecules.items():
            product_name = f"mol_{ids[0][1]}"  # Use mol_idx as product name
            items_for_precompute.append((product_name, smiles))
        
        # Precompute conformers in parallel (if enabled and not already computed)
        if self.use_precomputed_conformers:
            # Check if we need to precompute (check if index exists and has all molecules)
            index_path = os.path.join(self.precomputed_dir, "index.pt")
            need_precompute = True
            
            if os.path.exists(index_path):
                try:
                    import torch
                    index = torch.load(index_path)
                    # Check if all molecules are already precomputed
                    all_precomputed = all(
                        f"mol_{ids[0][1]}" in index 
                        for ids in self.unique_molecules.values()
                    )
                    if all_precomputed:
                        need_precompute = False
                        bt.logging.info("All conformers already precomputed, skipping...")
                except Exception:
                    pass
            
            if need_precompute:
                bt.logging.info(f"Precomputing conformers for {len(items_for_precompute)} molecules...")
                from boltz.precompute_conformers import precompute_conformers_batch
                stats = precompute_conformers_batch(
                    items_for_precompute,
                    Path(self.precomputed_dir),
                    max_workers=self.config.get('precompute_workers', 32),
                    shard_size=self.config.get('precompute_shard_size', 1000),
                )
                bt.logging.info(f"Precomputation stats: {stats}")
        
        # Write YAML files (still needed for Boltz input format, but conformers are precomputed)
        for smiles, ids in self.unique_molecules.items():
            yaml_content = self.create_yaml_content(smiles)
            with open(os.path.join(self.input_dir, f"{ids[0][1]}.yaml"), "w") as f:
                f.write(yaml_content)

        bt.logging.debug(f"Preprocessing data for Boltz2 complete")
            
    def create_yaml_content(self, ligand_smiles: str) -> str:
        """Create YAML content for Boltz2 prediction with no MSA"""

        yaml_content = f"""version: 1
sequences:
    - protein:
        id: A
        sequence: {self.protein_sequence}
        msa: empty
    - ligand:
        id: B
        smiles: '{ligand_smiles}'
        """

        if self.subnet_config['binding_pocket'] is not None:
            yaml_content += f"""
constraints:
    - pocket:
        binder: B
        contacts: {self.subnet_config['binding_pocket']}
        max_distance: {self.subnet_config['max_distance']}
        force: {self.subnet_config['force']}
        """

        yaml_content += f"""
properties:
    - affinity:
        binder: B
        """
        
        return yaml_content
    def _load_models_if_needed(self):
        """Load and cache models if not already loaded. Saves 72.4s per iteration after first."""
        if self._models_loaded:
            return
        
        cache = Path("~/.boltz").expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        
        bt.logging.info(f"Loading and caching Boltz-2 models (one-time cost ~72s)...")
        
        # Load structure model
        checkpoint = cache / "boltz2_conf.ckpt"
        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.5
        pairformer_args = PairformerArgsV2()
        msa_args = MSAModuleArgs(
            subsample_msa=True,
            num_subsampled_msa=1024,
            use_paired_feature=True,
        )
        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        
        predict_args_structure = {
            "recycling_steps": self.config['recycling_steps'],
            "sampling_steps": self.config['sampling_steps'],
            "diffusion_samples": self.config['diffusion_samples'],
            "max_parallel_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        
        # Set CUDA device context before loading to ensure models load on correct GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            # Clear cache on target device before loading
            torch.cuda.empty_cache()
            gc.collect()
        
        self.structure_model = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args_structure,
            map_location=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not self.config.get('no_kernels', False),
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        # Explicitly move model to target device (PyTorch Lightning may not respect map_location)
        if torch.cuda.is_available():
            self.structure_model = self.structure_model.to(f"cuda:{self.device_id}")
        self.structure_model.eval()
        
        # Load affinity model
        affinity_checkpoint = cache / "boltz2_aff.ckpt"
        predict_args_affinity = {
            "recycling_steps": 5,
            "sampling_steps": self.config['sampling_steps_affinity'],
            "diffusion_samples": self.config['diffusion_samples_affinity'],
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        steering_args_affinity = BoltzSteeringParams()
        steering_args_affinity.fk_steering = False
        steering_args_affinity.guidance_update = False
        steering_args_affinity.physical_guidance_update = False
        steering_args_affinity.contact_guidance_update = False
        
        self.affinity_model = Boltz2.load_from_checkpoint(
            affinity_checkpoint,
            strict=True,
            predict_args=predict_args_affinity,
            map_location=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args_affinity),
            affinity_mw_correction=self.config.get('affinity_mw_correction', False),
        )
        # Explicitly move model to target device
        if torch.cuda.is_available():
            self.affinity_model = self.affinity_model.to(f"cuda:{self.device_id}")
        self.affinity_model.eval()
        
        # Note: FP16 quantization is handled by PyTorch Lightning trainer with precision="16-mixed"
        # We don't need to manually convert models to .half() - AMP handles dtype conversions automatically
        quantization = self.config.get('quantization', 'none')
        if quantization == 'fp16':
            bt.logging.info("FP16 quantization enabled - PyTorch Lightning will use AMP (automatic mixed precision)")
            bt.logging.info("This provides ~2x speedup with minimal accuracy loss")
        elif quantization == 'int8':
            bt.logging.info("Applying INT8 (weight-only) quantization to models...")
            from boltz.quantize_model import quantize_model
            self.structure_model = quantize_model(self.structure_model, quantization)
            self.affinity_model = quantize_model(self.affinity_model, quantization)
            bt.logging.info("Models quantized to INT8 (weight-only)")
        
        self._models_loaded = True
        bt.logging.info("Boltz-2 models cached successfully (will reuse in future iterations)")
    
    # @profile
    def score_molecules_target(self, valid_molecules_by_uid: dict, score_dict: dict, subnet_config: dict, final_block_hash: str) -> None:
        # Preprocess data
        self.subnet_config = subnet_config

        self.preprocess_data_for_boltz(valid_molecules_by_uid, score_dict, final_block_hash)

        # Run Boltz2 for unique molecules
        bt.logging.info(f"Running Boltz2 on GPU {self.device_id}")
        try:
            # Ensure we're using the correct CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device_id)
            
            _restore_rng(self._rng0)
            
            # Load models if not cached (first iteration only)
            self._load_models_if_needed()
            
            # Pass devices as a list to specify which GPU to use
            # PyTorch Lightning uses devices=[1] to select GPU 1, devices=[0] for GPU 0
            # Set num_workers to avoid shared memory (shm) bus errors
            # Using 0 workers (main process only) avoids multiprocessing shared memory issues
            # This is safer and more stable, though slightly slower than multiprocessing
            base_num_workers = self.config.get('num_workers', 0)  # Default to 0 to avoid shm bus errors
            
            # Cap workers at 4 maximum to prevent shared memory issues
            # Even with spawn method, too many workers can exhaust /dev/shm
            if base_num_workers > 4:
                bt.logging.warning(f"num_workers={base_num_workers} is too high, capping at 4 to avoid shared memory issues")
                base_num_workers = 4
            
            # Adaptive worker adjustment based on available memory
            if self.enable_memory_monitoring:
                num_workers = self._adjust_workers_for_memory(base_num_workers)
            else:
                num_workers = base_num_workers
            
            # Ensure we don't exceed the cap (prevent shared memory bus errors)
            num_workers = min(num_workers, 4)
            
            # Log worker count for debugging
            if num_workers > 0:
                bt.logging.info(f"Using {num_workers} DataLoader workers (capped at 4 to avoid shared memory issues)")
            else:
                bt.logging.info("Using main process only (num_workers=0) to avoid shared memory bus errors")
            
            # Check GPU memory before processing
            if torch.cuda.is_available() and self.enable_memory_monitoring:
                gpu_memory_usage = self._check_gpu_memory()
                if gpu_memory_usage > self.max_gpu_memory_usage:
                    bt.logging.warning(f"GPU memory usage high ({gpu_memory_usage:.1%}), consider reducing batch size or workers")
                    # Clear cache before processing
                    torch.cuda.empty_cache()
                    gc.collect()
            
            predict(
                data = self.input_dir,
                out_dir = self.output_dir,
                recycling_steps = self.config['recycling_steps'],
                sampling_steps = self.config['sampling_steps'],
                diffusion_samples = self.config['diffusion_samples'],
                sampling_steps_affinity = self.config['sampling_steps_affinity'],
                diffusion_samples_affinity = self.config['diffusion_samples_affinity'],
                output_format = self.config['output_format'],
                seed = 68,
                affinity_mw_correction = self.config['affinity_mw_correction'],
                no_kernels = self.config['no_kernels'],
                batch_predictions = self.config['batch_predictions'],
                override = self.config['override'],
                devices = [self.device_id],  # Pass as list to specify GPU device
                num_workers = num_workers,  # Critical: increase workers to parallelize data loading
                structure_model = self.structure_model,  # Pass cached model
                affinity_model = self.affinity_model,  # Pass cached model
                precomputed_conformers_dir = self.precomputed_dir if self.use_precomputed_conformers else None,  # Pass precomputed conformers
                quantization = self.config.get('quantization', 'none'),  # Pass quantization setting to trainer
            )
            bt.logging.info(f"Boltz2 predictions complete")

        except Exception as e:
            bt.logging.error(f"Error running Boltz2: {e}")
            bt.logging.error(traceback.format_exc())
            return None

        # Collect scores and distribute results to all UIDs
        self.postprocess_data(score_dict)
        # Defer cleanup tp preserve unique_molecules for result submission
    # @profile
    def postprocess_data(self, score_dict: dict) -> None:
        # Collect scores - Results need to be saved to disk because of distributed predictions
        scores = {}
        for smiles, id_list in self.unique_molecules.items():
            mol_idx = id_list[0][1] # unique molecule identifier, same for all UIDs
            results_path = os.path.join(self.output_dir, 'boltz_results_inputs', 'predictions', f'{mol_idx}')
            if mol_idx not in scores:
                scores[mol_idx] = {}
            for filepath in os.listdir(results_path):
                if filepath.startswith('affinity'):
                    with open(os.path.join(results_path, filepath), 'r') as f:
                        affinity_data = json.load(f)
                    scores[mol_idx].update(affinity_data)
                elif filepath.startswith('confidence'):
                    with open(os.path.join(results_path, filepath), 'r') as f:
                        confidence_data = json.load(f)
                    scores[mol_idx].update(confidence_data)
        #bt.logging.debug(f"Collected scores: {scores}")

        if self.config['remove_files']:
            bt.logging.info("Removing files")
            # Optimized: Use shutil and pathlib instead of os.system for better performance
            results_dir = Path(self.output_dir) / 'boltz_results_inputs'
            if results_dir.exists():
                shutil.rmtree(results_dir)  # Faster than os.system("rm -r")
            
            # Remove YAML files more efficiently
            yaml_files = list(Path(self.input_dir).glob("*.yaml"))
            for yaml_file in yaml_files:
                yaml_file.unlink()  # Faster than os.system("rm")
            bt.logging.info("Files removed")

        # Distribute results to all UIDs
        self.per_molecule_metric = {}
        final_boltz_scores = {}
        for smiles, id_list in self.unique_molecules.items():
            for uid, mol_idx in id_list:
                if uid not in final_boltz_scores:
                    final_boltz_scores[uid] = []
                    
                metric_value = scores[mol_idx][self.subnet_config['boltz_metric']]
                final_boltz_scores[uid].append(metric_value)
                if uid not in self.per_molecule_metric:
                    self.per_molecule_metric[uid] = {}
                self.per_molecule_metric[uid][smiles] = metric_value
        bt.logging.debug(f"final_boltz_scores: {final_boltz_scores}")


        for uid, data in score_dict.items():
            if uid in final_boltz_scores:
                data['boltz_score'] = np.mean(final_boltz_scores[uid])
            else:
                data['boltz_score'] = -math.inf
    
    def _check_gpu_memory(self) -> float:
        """
        Check current GPU memory usage as a fraction of total VRAM.
        
        Returns:
            float: GPU memory usage as a fraction (0.0 to 1.0)
        """
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            device = torch.device(f"cuda:{self.device_id}")
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)
            
            # Use cached memory as it's more accurate for peak usage
            usage = cached_memory / total_memory if total_memory > 0 else 0.0
            return usage
        except Exception as e:
            bt.logging.warning(f"Error checking GPU memory: {e}")
            return 0.0
    
    def _check_ram_usage(self) -> float:
        """
        Check current system RAM usage as a fraction of total RAM.
        
        Returns:
            float: RAM usage as a fraction (0.0 to 1.0)
        """
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception as e:
            bt.logging.warning(f"Error checking RAM usage: {e}")
            return 0.0
    
    def _adjust_workers_for_memory(self, base_workers: int) -> int:
        """
        Adjust number of workers based on available RAM to prevent OOM.
        
        Args:
            base_workers: Base number of workers from config
            
        Returns:
            int: Adjusted number of workers (reduced if RAM is high)
        """
        try:
            ram_usage = self._check_ram_usage()
            
            # Each DataLoader worker uses ~50-100MB RAM
            # Estimate: base_workers * 75MB average
            estimated_worker_memory_mb = base_workers * 75
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / (1024 * 1024)
            
            # If RAM usage is above threshold, reduce workers
            if ram_usage > self.max_ram_usage:
                # Reduce workers proportionally
                reduction_factor = (1.0 - ram_usage) / (1.0 - self.max_ram_usage)
                adjusted_workers = max(4, int(base_workers * reduction_factor))
                bt.logging.warning(
                    f"RAM usage high ({ram_usage:.1%}), reducing workers from {base_workers} to {adjusted_workers}"
                )
                return adjusted_workers
            
            # If we don't have enough available memory for workers, reduce
            if available_memory_mb < estimated_worker_memory_mb * 1.5:  # 1.5x safety margin
                max_safe_workers = int(available_memory_mb / (75 * 1.5))
                adjusted_workers = max(4, min(base_workers, max_safe_workers))
                if adjusted_workers < base_workers:
                    bt.logging.warning(
                        f"Limited RAM available ({available_memory_mb:.0f}MB), reducing workers from {base_workers} to {adjusted_workers}"
                    )
                return adjusted_workers
            
            return base_workers
            
        except Exception as e:
            bt.logging.warning(f"Error adjusting workers for memory: {e}, using base workers")
            return base_workers
    
    def clear_gpu_memory(self):
        """Clear GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset CUDA context to allow other processes to initialize
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    def cleanup_model(self):
        """Clean up model and free GPU memory."""
        
        # Clear any model-specific attributes
        if hasattr(self, 'unique_molecules'):
            del self.unique_molecules
            self.unique_molecules = None
        if hasattr(self, 'protein_sequence'):
            del self.protein_sequence
            self.protein_sequence = None
        
        # Optionally clear cached models to free GPU memory
        # Uncomment if you need to free GPU memory between epochs
        # if self.structure_model is not None:
        #     del self.structure_model
        #     self.structure_model = None
        # if self.affinity_model is not None:
        #     del self.affinity_model
        #     self.affinity_model = None
        # self._models_loaded = False
            
        self.clear_gpu_memory()

