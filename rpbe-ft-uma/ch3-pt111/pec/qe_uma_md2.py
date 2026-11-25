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
    "/home/bruno/doutorado/fairchem/4-pt/resposta/finetuning-com-replay/checkpoint-nivel-dft/09-11-resposta/novo-teste-pertub-2/inference_ckpt.pt"
)

OUT_DAT = "pec_ch3_ft10_rpbe.dat"

TASK_NAME = "oc20"


# ==========================================================
# CONSTANTES PARA Eads
# ==========================================================

E_SLAB = -331.9816186249
E_CH3 = -17.713
F_CORR = 0.235

EV_TO_KCAL = 23.06


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
# pt-ch3-1.6A -> 1.6
# pt-ch3-1.9A -> 1.9
# pt-ch3-6.0A -> 6.0
# ==========================================================

def get_distance_from_folder(folder_name):

    m = re.fullmatch(
        r"pt-ch3-(\d+(?:\.\d+)?)A",
        folder_name
    )

    if m is None:
        return None

    return float(
        m.group(1)
    )


# ==========================================================
# PROCURA O -qe.traj
#
# Em algumas pastas você pode ter:
#
# pt-ch3-1.9A/
# ├── pt-ch3-1.9A-qe.traj
#
# ou:
#
# pt-ch3-1.9A/
# └── pt-ch3-1.9A/
#     └── pt-ch3-1.9A-qe.traj
# ==========================================================

def encontrar_qe_traj(folder):

    system_name = folder.name

    # primeira opção: arquivo direto na pasta
    traj1 = (
        folder /
        f"{system_name}-qe.traj"
    )

    if traj1.exists():
        return traj1

    # segunda opção: subpasta com mesmo nome
    traj2 = (
        folder /
        system_name /
        f"{system_name}-qe.traj"
    )

    if traj2.exists():
        return traj2

    return None


# ==========================================================
# MAIN
# ==========================================================

def main():

    folders = []

    # ------------------------------------------------------
    # encontra todas as pastas pt-ch3-*A
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

        print("=" * 70)

        print(
            f"Distância = {dist:.3f} Å"
        )

        qe_traj = encontrar_qe_traj(
            folder
        )


        # --------------------------------------------------
        # verifica se existe
        # --------------------------------------------------

        if qe_traj is None:

            print(
                f"[AVISO] Nenhum -qe.traj encontrado "
                f"em {folder}"
            )

            continue


        print(
            f"Arquivo = {qe_traj}"
        )


        try:

            # ==============================================
            # ÚLTIMA GEOMETRIA OTIMIZADA PELO QE
            # ==============================================

            atoms_qe = read(
                qe_traj,
                index=-1
            )


            # ==============================================
            # CÓPIA EXATA DA GEOMETRIA QE
            # ==============================================

            atoms_ft = atoms_qe.copy()

            atoms_ft.set_cell(
                atoms_qe.cell.copy()
            )

            atoms_ft.set_pbc(
                atoms_qe.pbc.copy()
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
            # (E_CH3/Pt - E_Pt - E_CH3 + Fcorr) * 23.06
            # ==============================================

            E_ads = (
                E_tot
                - E_SLAB
                - E_CH3
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


            # ==============================================
            # PRINT
            # ==============================================

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
