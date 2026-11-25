#!/usr/bin/env python3

import glob
import re
import os
import numpy as np

from ase.optimize import BFGS
from ase.constraints import FixAtoms, FixedPlane
from ase.io import read, write

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


# ==========================================================
# CONSTANTES
# ==========================================================

EV2KCAL = 23.06


# ==========================================================
# EXTRAI DISTÂNCIA DO NOME
#
# Exemplos:
# n-ni100-0.00.xyz -> 0.00
# n-ni100-2.50.xyz -> 2.50
# n-ni100-6.00.xyz -> 6.00
# ==========================================================

def parse_dist(path):

    m = re.search(
        r"n-ni100-([0-9]+(?:\.[0-9]+)?)\.xyz$",
        os.path.basename(path)
    )

    return float(m.group(1)) if m else None


# ==========================================================
# p_mean
#
# camada superior do Ni(100):
# índices 12, 13, 14, 15
# ==========================================================

TOP_NI_IDX = [12, 13, 14, 15]


def compute_p_mean(atoms):

    symbols = atoms.get_chemical_symbols()

    n_idx = [
        i for i, symbol in enumerate(symbols)
        if symbol == "N"
    ]

    if len(n_idx) == 0:
        return float("nan")

    n = n_idx[0]

    distances = atoms.get_distances(
        n,
        TOP_NI_IDX,
        mic=True
    )

    return float(
        np.mean(distances)
    )


# ==========================================================
# MODELO FAIR-CHEM
# ==========================================================

CHECKPOINT = (
    "/home/bruno/doutorado/fairchem/4-pt/resposta/"
    "finetuning-com-replay/checkpoint-nivel-dft/"
    "checkpoint-umlip/10-epochs/"
    "inference_ckpt.pt"
)

print("\nCarregando modelo...")
print(CHECKPOINT)

predictor = load_predict_unit(
    CHECKPOINT,
    device="cpu"
)

calc = FAIRChemCalculator(
    predictor,
    task_name="oc20"
)


# ==========================================================
# SLAB PURO
# ==========================================================

print("\n========================================")
print("Otimizando slab puro")
print("========================================")

slab = read(
    "ni-100-2x2-4l-opt.xyz"
)

slab.set_constraint(
    FixAtoms(indices=range(8))
)

slab.calc = calc

opt_slab = BFGS(
    slab,
    logfile="ni100-puro-opt.log"
)

opt_slab.run(
    fmax=0.01,
    steps=100
)

slab_e = float(
    slab.get_potential_energy()
)

print(
    f"Energia calculada do slab puro: "
    f"{slab_e:.9f} eV"
)

print(
    "Referência fixa usada no E_ads: "
    "-72.13752834604227 eV"
)


# ==========================================================
# ARQUIVO DE SAÍDA
# ==========================================================

OUTFILE = "adsorption_curve_n_ni100_sem_corr.dat"

with open(
    OUTFILE,
    "w",
    encoding="utf-8"
) as fout:

    fout.write(
        "# dist(Å)   "
        "E_tot(eV)   "
        "E_ads(kcal/mol)   "
        "p_mean(Å)\n"
    )


    # ======================================================
    # LOCALIZA E ORDENA OS ARQUIVOS
    # ======================================================

    files = glob.glob(
        "n-ni100-*.xyz"
    )

    files = [
        f for f in files
        if parse_dist(f) is not None
    ]

    files = sorted(
        files,
        key=parse_dist
    )

    print(
        f"\nEncontrados {len(files)} "
        "arquivos para o scan."
    )

    if len(files) == 0:

        raise RuntimeError(
            "Nenhum arquivo n-ni100-*.xyz "
            "compatível foi encontrado."
        )


    # ======================================================
    # LOOP DO SCAN
    # ======================================================

    for fname in files:

        dist = parse_dist(fname)

        print("\n========================================")
        print(
            f">>> Rodando {fname} "
            f"(dist={dist:.2f} Å)"
        )
        print("========================================")

        try:

            # --------------------------------------------------
            # lê estrutura
            # --------------------------------------------------

            adslab = read(fname)


            # --------------------------------------------------
            # identifica N
            # --------------------------------------------------

            n_idx = [
                i
                for i, atom in enumerate(adslab)
                if atom.symbol == "N"
            ]

            if len(n_idx) == 0:

                raise RuntimeError(
                    f"Nenhum N encontrado em {fname}"
                )


            # --------------------------------------------------
            # identifica Ni
            # --------------------------------------------------

            ni_idx = [
                i
                for i, atom in enumerate(adslab)
                if atom.symbol == "Ni"
            ]

            print(
                f"N index: {n_idx}"
            )

            print(
                f"Número de Ni: {len(ni_idx)}"
            )


            # --------------------------------------------------
            # RESTRIÇÕES
            #
            # primeiros 8 Ni fixos
            # N fixo
            # Ni sem movimento em z
            # --------------------------------------------------

            fix_bottom = FixAtoms(
                indices=range(8)
            )

            fix_N = FixAtoms(
                indices=n_idx
            )

            fix_Niz = FixedPlane(
                indices=ni_idx,
                direction=[0, 0, 1]
            )

            adslab.set_constraint(
                [
                    fix_bottom,
                    fix_N,
                    fix_Niz
                ]
            )


            # --------------------------------------------------
            # calculator
            # --------------------------------------------------

            adslab.calc = calc


            # --------------------------------------------------
            # otimização
            # --------------------------------------------------

            logfile = fname.replace(
                ".xyz",
                ".log"
            )

            opt = BFGS(
                adslab,
                logfile=logfile
            )

            opt.run(
                fmax=0.01,
                steps=200
            )


            # --------------------------------------------------
            # salva estrutura final
            # --------------------------------------------------

            trajfile = fname.replace(
                ".xyz",
                ".traj"
            )

            write(
                trajfile,
                adslab,
                format="traj"
            )


            # --------------------------------------------------
            # energia total
            # --------------------------------------------------

            E_tot = float(
                adslab.get_potential_energy()
            )


            # --------------------------------------------------
            # ENERGIA DE ADSORÇÃO
            #
            # REFERÊNCIA FIXA MANTIDA
            # --------------------------------------------------

            E_ads_kcal = (
                E_tot
                - (-72.13752834604227)
                - (-8.083)
                + (-4.877556)
            ) * EV2KCAL


            # --------------------------------------------------
            # p_mean
            # --------------------------------------------------

            pmean = compute_p_mean(
                adslab
            )


            # --------------------------------------------------
            # imprime
            # --------------------------------------------------

            print(
                f"E_tot = {E_tot:+.9f} eV"
            )

            print(
                f"E_ads = {E_ads_kcal:+.3f} "
                "kcal/mol"
            )

            print(
                f"p_mean = {pmean:.6f} Å"
            )


            # --------------------------------------------------
            # escreve no dat
            # --------------------------------------------------

            fout.write(
                f"{dist:7.2f}   "
                f"{E_tot:+.9f}   "
                f"{E_ads_kcal:+.3f}   "
                f"{pmean:7.4f}\n"
            )

            fout.flush()


        except Exception as e:

            print(
                f"[ERRO] {fname}: {e}"
            )


# ==========================================================
# FINAL
# ==========================================================

print("\n========================================")
print("SCAN FINALIZADO")
print("========================================")

print(
    f"Resultados escritos em: {OUTFILE}"
)
