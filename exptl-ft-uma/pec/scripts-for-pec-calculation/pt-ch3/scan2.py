import glob
import re
from ase.optimize import BFGS
from ase.constraints import FixAtoms, FixInternals
from ase.io import read, write
from ase.io.trajectory import Trajectory
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


# --- modelo fairchem ---
predictor = load_predict_unit(
    "/home/bruno/doutorado/fairchem/4-pt/resposta/"
    "finetuning-com-replay/checkpoint-nivel-dft/"
    "checkpoint-umlip/treino-proximo/teste-3/inference_ckpt.pt"
)

calc = FAIRChemCalculator(
    predictor,
    task_name="oc20"
)


# --- energia de referência da molécula isolada ---
E_ads_ref = -17.713
E_corr = 0.273


# --- otimização do slab puro (só uma vez) ---
slab = read("pt-16ml-opt.xyz")

slab.set_constraint(
    FixAtoms(indices=range(32))
)

slab.set_calculator(calc)

opt_slab = BFGS(slab)

opt_slab.run(
    fmax=0.01,
    steps=100
)

slab_e = slab.get_potential_energy()

print(
    "Energia do slab puro (eV):",
    slab_e
)


# ==========================================================
# ARQUIVO DE SAÍDA
# ==========================================================

with open("adsorption_curve.dat", "w") as fout:

    fout.write(
        "#dist(Å)   E_tot(eV)   E_ads(kcal/mol)\n"
    )


    # ======================================================
    # LOOP SOBRE TODOS OS XYZ
    # ======================================================

    for fname in sorted(
        glob.glob("pt-ch3-*.xyz")
    ):

        print(
            f"\n>>> Rodando {fname}"
        )


        # --------------------------------------------------
        # extrai distância:
        #
        # pt-ch3-1.50.xyz -> 1.50
        # --------------------------------------------------

        m = re.search(
            r"pt-ch3-([0-9]+(?:\.[0-9]+)?)\.xyz$",
            fname
        )

        if not m:

            print(
                f"[ERRO] Não consegui extrair "
                f"a distância de {fname}"
            )

            continue


        dist = float(
            m.group(1)
        )


        # --------------------------------------------------
        # lê estrutura
        # --------------------------------------------------

        adslab = read(
            fname
        )


        # --------------------------------------------------
        # restrições
        # --------------------------------------------------

        constraint = FixAtoms(
            indices=range(32)
        )


        # trava distância C-Pt inicial
        bond1 = [
            dist,
            [58, 64]
        ]

        c = FixInternals(
            bonds=[bond1]
        )


        adslab.set_constraint(
            [
                constraint,
                c
            ]
        )


        # --------------------------------------------------
        # calculator
        # --------------------------------------------------

        adslab.set_calculator(
            calc
        )


        # --------------------------------------------------
        # otimização
        # --------------------------------------------------

        trajfile = (
            f"pt-ch3-{dist:.2f}.traj"
        )

        opt = BFGS(
            adslab,
            trajectory=trajfile
        )

        opt.run(
            fmax=0.01,
            steps=200
        )


        adslab_e = (
            adslab.get_potential_energy()
        )


        # --------------------------------------------------
        # energia de adsorção
        # --------------------------------------------------

        eads = (
            adslab_e
            - (-332.626790792820)
            - E_ads_ref
            + E_corr
        ) * 23.06


        print(
            f"Distância {dist:.2f} Å: "
            f"E = {adslab_e:.4f} eV, "
            f"Eads = {eads:.2f} kcal/mol"
        )


        # --------------------------------------------------
        # salva json
        # --------------------------------------------------

        jsonfile = (
            f"pt-ch3-{dist:.2f}.json"
        )

        write(
            jsonfile,
            adslab,
            format="json"
        )


        # --------------------------------------------------
        # salva no dat
        # --------------------------------------------------

        fout.write(
            f"{dist:.2f}   "
            f"{adslab_e:.6f}   "
            f"{eads:.6f}\n"
        )

        fout.flush()


print(
    "\nScan finalizado. "
    "Resultados salvos em adsorption_curve.dat"
)
