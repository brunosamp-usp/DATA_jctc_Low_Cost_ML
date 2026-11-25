#!/usr/bin/env python3

import re
from pathlib import Path

from ase.io import read

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(".")

CHECKPOINT = (
    "/home/bruno/doutorado/fairchem/4-pt/resposta/finetuning-com-replay/checkpoint-nivel-dft/09-11-resposta/novo-teste-pertub-2-cl/inference_ckpt.pt"
)

OUT_DAT = "pec_n_ni100_ft10_rpbe.dat"

TASK_NAME = "oc20"


# ==========================================================
# CONSTANTES PARA Eads
# ==========================================================

E_SLAB = -72.137528
E_N = -8.083
F_CORR = -4.877556

EV_TO_KCAL = 23.06


# ==========================================================
# CORRIGE METADADOS FAIR-CHEM
# ==========================================================

def fix_fairchem_metadata(atoms):

    for key in [
        "charge",
        "spin",
        "multiplicity"
    ]:

        if key in atoms.info:

            try:
                atoms.info[key] = int(
                    atoms.info[key]
                )
            except Exception:
                pass

    atoms.info["charge"] = int(
        atoms.info.get("charge", 0)
    )

    return atoms


# ==========================================================
# CARREGA CHECKPOINT FT10
# ==========================================================

print("=" * 70)
print("Carregando checkpoint FT10")
print("=" * 70)

print(CHECKPOINT)

predictor = load_predict_unit(
    CHECKPOINT
)

calc = FAIRChemCalculator(
    predictor,
    task_name=TASK_NAME
)

print("Modelo carregado.\n")


# ==========================================================
# EXTRAI DISTÂNCIA DO NOME DA PASTA
#
# n-ni100-0.0A -> 0.0
# n-ni100-1.0A -> 1.0
# n-ni100-6.0A -> 6.0
# ==========================================================

def get_distance_from_folder(folder_name):

    m = re.fullmatch(
        r"n-ni100-(\d+(?:\.\d+)?)A",
        folder_name
    )

    if m is None:
        return None

    return float(
        m.group(1)
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    folders = []

    # ------------------------------------------------------
    # encontra todas as pastas n-ni100-*A
    # ------------------------------------------------------

    for path in BASE_DIR.iterdir():

        if not path.is_dir():
            continue

        dist = get_distance_from_folder(
            path.name
        )

        if dist is not None:

            folders.append(
                (
                    dist,
                    path
                )
            )


    folders.sort(
        key=lambda x: x[0]
    )

    print(
        f"Pastas encontradas: "
        f"{len(folders)}"
    )

    print()

    resultados = []


    # ======================================================
    # LOOP
    # ======================================================

    for dist, folder in folders:

        system_name = folder.name

        # exemplo:
        #
        # n-ni100-1.0A/
        # └── n-ni100-1.0A-qe.traj

        qe_traj = (
            folder /
            f"{system_name}-qe.traj"
        )


        print("=" * 70)

        print(
            f"Distância = {dist:.3f} Å"
        )

        print(
            f"Arquivo = {qe_traj}"
        )


        # --------------------------------------------------
        # verifica se existe
        # --------------------------------------------------

        if not qe_traj.exists():

            print(
                "[AVISO] Arquivo -qe.traj "
                "não encontrado. Pulando."
            )

            continue


        try:

            # ==============================================
            # ÚLTIMO FRAME DO QE
            # ==============================================

            atoms_qe = read(
                qe_traj,
                index=-1
            )


            # ==============================================
            # CÓPIA DA GEOMETRIA
            # ==============================================

            atoms_ft = atoms_qe.copy()

            atoms_ft.set_cell(
                atoms_qe.cell.copy()
            )

            atoms_ft.set_pbc(
                atoms_qe.pbc.copy()
            )

            atoms_ft = fix_fairchem_metadata(
                atoms_ft
            )


            # ==============================================
            # SINGLE POINT FT10
            # ==============================================

            atoms_ft.calc = calc

            E_tot = float(
                atoms_ft.get_potential_energy()
            )


            # ==============================================
            # ENERGIA DE ADSORÇÃO
            #
            # Eads =
            # (E_N/Ni - E_Ni - E_N + Fcorr)*23.06
            # ==============================================

            E_ads = (
                E_tot
                - E_SLAB
                - E_N
                + F_CORR
            ) * EV_TO_KCAL


            # ==============================================
            # GUARDA
            # ==============================================

            resultados.append(
                (
                    dist,
                    E_tot,
                    E_ads
                )
            )


            print(
                f"E_tot FT10 = "
                f"{E_tot:.10f} eV"
            )

            print(
                f"E_ads FT10 = "
                f"{E_ads:.6f} kcal/mol"
            )


        except Exception as exc:

            print(
                f"[ERRO] {folder.name}: {exc}"
            )


    # ======================================================
    # ORDENA
    # ======================================================

    resultados.sort(
        key=lambda x: x[0]
    )


    # ======================================================
    # SALVA DAT
    # ======================================================

    with open(
        OUT_DAT,
        "w",
        encoding="utf-8"
    ) as fout:

        fout.write(
            "# distance_A    "
            "E_FT10_eV    "
            "Eads_FT10_kcal_mol\n"
        )

        for dist, E_tot, E_ads in resultados:

            fout.write(
                f"{dist:10.3f}   "
                f"{E_tot:18.10f}   "
                f"{E_ads:18.8f}\n"
            )


    # ======================================================
    # FINAL
    # ======================================================

    print()
    print("=" * 70)
    print("FINALIZADO")
    print("=" * 70)

    print(
        f"Pontos calculados: "
        f"{len(resultados)}"
    )

    print(
        f"Arquivo salvo: "
        f"{OUT_DAT}"
    )


if __name__ == "__main__":
    main()
