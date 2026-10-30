"""Pure-Python helpers for PDB parsing and dihedral index extraction.

Ported from v1 ``asmsa.helpers`` with the TensorFlow / tensorflow-probability
imports removed. The GMM-on-circle helper lives in :mod:`asmsa_v2.priors.gmm`.
"""
import numpy as np


def extract_dihedral_indices_local(pdb_file):
    """Return local (0-based) atom-index quadruples for backbone + sidechain dihedrals."""

    CHI_DEFINITIONS = {
        'ARG': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
                ('CB', 'CG', 'CD', 'NE'), ('CG', 'CD', 'NE', 'CZ')],
        'ASN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')],
        'ASP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')],
        'CYS': [('N', 'CA', 'CB', 'SG')],
        'GLN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
                ('CB', 'CG', 'CD', 'OE1')],
        'GLU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
                ('CB', 'CG', 'CD', 'OE1')],
        'HIS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'ND1')],
        'ILE': [('N', 'CA', 'CB', 'CG1'), ('CA', 'CB', 'CG1', 'CD1')],
        'LEU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
        'LYS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
                ('CB', 'CG', 'CD', 'CE'), ('CG', 'CD', 'CE', 'NZ')],
        'MET': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'SD'),
                ('CB', 'CG', 'SD', 'CE')],
        'PHE': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
        'PRO': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD')],
        'SER': [('N', 'CA', 'CB', 'OG')],
        'THR': [('N', 'CA', 'CB', 'OG1')],
        'TRP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
        'TYR': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
        'VAL': [('N', 'CA', 'CB', 'CG1')],
    }

    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atoms.append({
                    'index': len(atoms),
                    'name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'res_seq': int(line[22:26].strip()),
                    'chain': line[21],
                })

    chains = {}
    for atom in atoms:
        chains.setdefault(atom['chain'], {}).setdefault(
            atom['res_seq'], {'res_name': atom['res_name'], 'atoms': {}}
        )['atoms'][atom['name']] = atom['index']

    phi_indices, psi_indices, omega_indices, chi_indices = [], [], [], []

    for residues in chains.values():
        res_list = sorted(residues.keys())
        for i, res_num in enumerate(res_list):
            curr = residues[res_num]
            curr_atoms = curr['atoms']
            res_name = curr['res_name']

            if i > 0:
                prev_atoms = residues[res_list[i - 1]]['atoms']
                if 'C' in prev_atoms and all(a in curr_atoms for a in ('N', 'CA', 'C')):
                    phi_indices.append((prev_atoms['C'], curr_atoms['N'],
                                        curr_atoms['CA'], curr_atoms['C']))
            if i < len(res_list) - 1:
                next_atoms = residues[res_list[i + 1]]['atoms']
                if all(a in curr_atoms for a in ('N', 'CA', 'C')) and 'N' in next_atoms:
                    psi_indices.append((curr_atoms['N'], curr_atoms['CA'],
                                        curr_atoms['C'], next_atoms['N']))
                if all(a in curr_atoms for a in ('CA', 'C')) and all(a in next_atoms for a in ('N', 'CA')):
                    omega_indices.append((curr_atoms['CA'], curr_atoms['C'],
                                          next_atoms['N'], next_atoms['CA']))

            if res_name in CHI_DEFINITIONS:
                for chi_def in CHI_DEFINITIONS[res_name]:
                    if all(a in curr_atoms for a in chi_def):
                        chi_indices.append(tuple(curr_atoms[a] for a in chi_def))

    all_dihedrals = phi_indices + psi_indices + omega_indices + chi_indices
    seen, unique = set(), []
    for q in all_dihedrals:
        if q not in seen:
            unique.append(q)
            seen.add(q)
    return unique


def extract_atom_indices(pdb_path):
    """Extract common atom-index subsets (backbone, nC, alpha, alphabeta) from a PDB file."""
    def _parse(target_atoms):
        indices = []
        with open(pdb_path) as f:
            atom_counter = 0
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                name = line[12:16].strip()
                if name in target_atoms:
                    indices.append(atom_counter)
                atom_counter += 1
        return indices

    return {
        "backbone": _parse({"N", "C", "CA"}),
        "nC": _parse({"N", "O", "S"}),
        "alpha": _parse({"CA"}),
        "alphabeta": _parse({"CA", "CB"}),
    }
