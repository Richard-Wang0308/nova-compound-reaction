import random
import requests
import bittensor as bt
from datasets import load_dataset


def get_protein_treat_model(protein_code: str) -> str:
    """
    Determine which TREAT model to use based on protein type.
    
    Returns:
        'TREAT1' for monoamine transporters (SERT, DAT, NET)
        'TREAT2' for HDACs (histone deacetylases)
        'TREAT1' as default for other proteins
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check protein name and description
            protein_name = ""
            description = ""
            
            # Get recommended name
            if "proteinDescription" in data:
                rec_name = data["proteinDescription"].get("recommendedName", {})
                if rec_name:
                    protein_name = rec_name.get("fullName", {}).get("value", "").lower()
                
                # Get alternative names
                alt_names = data["proteinDescription"].get("alternativeNames", [])
                for alt in alt_names:
                    alt_name = alt.get("fullName", {}).get("value", "").lower()
                    description += " " + alt_name
            
            # Get gene names
            genes = data.get("genes", [])
            for gene in genes:
                gene_names = gene.get("geneName", {})
                if gene_names:
                    value = gene_names.get("value", "").lower()
                    description += " " + value
            
            # Get keywords and features
            keywords = data.get("keywords", [])
            for kw in keywords:
                kw_value = kw.get("value", "").lower()
                description += " " + kw_value
            
            # Check for monoamine transporters (TREAT-1)
            monoamine_keywords = [
                "serotonin transporter", "sert", "slc6a4",
                "dopamine transporter", "dat", "slc6a3",
                "norepinephrine transporter", "net", "slc6a2",
                "monoamine transporter", "solute carrier family 6"
            ]
            
            # Check for HDACs (TREAT-2)
            hdac_keywords = [
                "histone deacetylase", "hdac", "histone deacetylase complex"
            ]
            
            full_text = (protein_name + " " + description).lower()
            
            # Check for HDACs first (more specific)
            for keyword in hdac_keywords:
                if keyword in full_text:
                    bt.logging.info(f"Protein {protein_code} identified as HDAC, using TREAT-2")
                    return "TREAT2"
            
            # Check for monoamine transporters
            for keyword in monoamine_keywords:
                if keyword in full_text:
                    bt.logging.info(f"Protein {protein_code} identified as monoamine transporter, using TREAT-1")
                    return "TREAT1"
            
            # Check Pfam/InterPro domains
            if "uniProtKBCrossReferences" in data:
                for ref in data["uniProtKBCrossReferences"]:
                    database = ref.get("database", "").lower()
                    if database in ["pfam", "interpro"]:
                        ref_id = ref.get("id", "").lower()
                        properties = ref.get("properties", [])
                        prop_values = " ".join([p.get("value", "").lower() for p in properties])
                        
                        # Check for HDAC domains
                        if "hist_deacetylase" in ref_id or "hist_deacetylase" in prop_values:
                            bt.logging.info(f"Protein {protein_code} has HDAC domain, using TREAT-2")
                            return "TREAT2"
                        
                        # Check for SLC6 family (monoamine transporters)
                        if "slc6" in ref_id or "solute carrier" in prop_values:
                            bt.logging.info(f"Protein {protein_code} has SLC6 domain, using TREAT-1")
                            return "TREAT1"
        
        # Default to TREAT-1 if we can't determine
        bt.logging.info(f"Could not determine protein type for {protein_code}, defaulting to TREAT-1")
        return "TREAT1"
        
    except Exception as e:
        bt.logging.warning(f"Error determining TREAT model for {protein_code}: {e}, defaulting to TREAT-1")
        return "TREAT1"


def get_sequence_from_protein_code(protein_code: str) -> str:
    """
    Get the amino acid sequence for a protein code.
    First tries to fetch from UniProt API, and if that fails,
    falls back to searching the Hugging Face dataset.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        lines = response.text.splitlines()
        sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
        amino_acid_sequence = ''.join(sequence_lines)
        # Check if the sequence is empty
        if not amino_acid_sequence:
            bt.logging.warning(f"Retrieved empty sequence for {protein_code} from UniProt API")
        else:
            return amino_acid_sequence
    
    bt.logging.info(f"Failed to retrieve sequence for {protein_code} from UniProt API. Trying Hugging Face dataset.")
    try:
        dataset = load_dataset("Metanova/Proteins", split="train")
        
        for i in range(len(dataset)):
            if dataset[i]["Entry"] == protein_code:
                sequence = dataset[i]["Sequence"]
                bt.logging.info(f"Found sequence for {protein_code} in Hugging Face dataset")
                return sequence
                
        bt.logging.error(f"Could not find protein {protein_code} in Hugging Face dataset")
        return None
        
    except Exception as e:
        bt.logging.error(f"Error accessing Hugging Face dataset: {e}")
        return None


def get_challenge_params_from_blockhash(block_hash: str, weekly_target: str, num_antitargets: int, include_reaction: bool = False) -> dict:
    """
    Use block_hash as a seed to pick 'num_targets' and 'num_antitargets' random entries
    from the 'Metanova/Proteins' dataset. Optionally also pick allowed reaction.
    Returns {'targets': [...], 'antitargets': [...], 'allowed_reaction': '...'}.
    """
    if not (isinstance(block_hash, str) and block_hash.startswith("0x")):
        raise ValueError("block_hash must start with '0x'.")
    if not weekly_target or num_antitargets < 0:
        raise ValueError("weekly_target must exist and num_antitargets must be non-negative.")

    # Convert block hash to an integer seed
    try:
        seed = int(block_hash[2:], 16)
    except ValueError:
        raise ValueError(f"Invalid hex in block_hash: {block_hash}")

    # Initialize random number generator
    rng = random.Random(seed)

    # Load huggingface protein dataset
    try:
        dataset = load_dataset("Metanova/Proteins", split="train")
    except Exception as e:
        raise RuntimeError("Could not load the 'Metanova/Proteins' dataset.") from e

    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty; cannot pick random entries.")

    # Grab all required indices at once, ensure uniqueness
    unique_indices = rng.sample(range(dataset_size), k=(num_antitargets))

    # Split indices for antitargets
    antitarget_indices = unique_indices[:num_antitargets]

    # Convert indices to protein codes
    targets = [weekly_target]
    antitargets = [dataset[i]["Entry"] for i in antitarget_indices]

    result = {
        "targets": targets,
        "antitargets": antitargets
    }

    if include_reaction:
        try:
            from .reactions import get_total_reactions
            total_reactions = get_total_reactions()
            allowed_option = seed % total_reactions
            if allowed_option == 0:
                result["allowed_reaction"] = "savi"
            else:
                result["allowed_reaction"] = f"rxn:{allowed_option}"
        except Exception as e:
            bt.logging.warning(f"Failed to determine allowed reaction: {e}, defaulting to all reactions allowed")

    return result
