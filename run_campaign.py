"""run_campaign.py — Orchestrateur de campagne V2 (grilles synchronisees avec les .bat)

Rejoue EXACTEMENT les 100 memes configurations que les 3 fichiers
lancer_grille_poids.bat / lancer_grille_cache.bat / lancer_grille_conjointe.bat :
  Phase 1 POIDS      : int4|int8|fp16 x flash off|on x 5 contextes  = 30
  Phase 2 CACHE      : int8|int4|int2 x flash off|on x 5 contextes  = 30
  Phase 3 CONJOINTE  : int4 x 3 caches x 2 flash + int8 x int4 x 2  = 40

Avantages sur le .bat :
  - Reprise : les configs deja presentes dans le CSV sont sautees.
  - Attend le GPU libre (nvidia-smi) avant chaque config (--gpu-wait).
  - Timeout par config (--timeout) : un run bloque n'arrete pas la campagne.
  - Meme thermique que le .bat : 15 s entre configs, 45 s entre blocs, 60 s en fin de phase.

Usage :
  python run_campaign.py --phase all --dry-run
  python run_campaign.py --phase all --gpu-wait 60
  python run_campaign.py --phase 3 --retry-oom
"""

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


CTXS = [512, 1024, 2048, 4096, 8192]
FLASH = [False, True]

PAUSE_INTER_CONFIG = 15
PAUSE_INTER_BLOC = 45
PAUSE_FIN_PHASE = 60

MY_MACHINE = socket.gethostname()
GPU_BUSY_MIB = 2000  # VRAM (Mo) au dela de laquelle on considere le GPU occupe
CODE_GPU_OCCUPE = 3   # code de retour signifiant "GPU occupe par un autre travail"
RESULTS_DIR = Path("results")
LOG_FILE = RESULTS_DIR / "campaign.log"


def grid_poids():
    return [("int4", "fp16", f) for f in FLASH] + \
           [("int8", "fp16", f) for f in FLASH] + \
           [("fp16", "fp16", f) for f in FLASH]


def grid_cache():
    return [(("fp16", k, f)) for k in ["int8", "int4", "int2"] for f in FLASH]


def grid_conjointe():
    base = [("int4", k, f) for k in ["int8", "int4", "int2"] for f in FLASH]
    base += [("int8", "int4", f) for f in FLASH]
    return base


PHASES = {
    "1": {
        "name": "POIDS (cache fixe FP16 natif)",
        "script": "scripts/01_quant_poids.py",
        "csv": "results_v2_poids.csv",
        "base": grid_poids(),
    },
    "2": {
        "name": "CACHE (poids FP16 fixes)",
        "script": "scripts/02_quant_cache_v2.py",
        "csv": "results_v2_cache.csv",
        "base": grid_cache(),
    },
    "3": {
        "name": "CONJOINTE (poids + cache quantifies)",
        "script": "scripts/03_quant_conjointe_v2.py",
        "csv": "results_v2_conjointe.csv",
        "base": grid_conjointe(),
    },
}


def find_base_dir() -> Path:
    """Racine du projet : dossier contenant scripts/ et results/."""
    here = Path(__file__).resolve().parent
    # Marqueur definitif du dossier projet : il contient le sous-dossier scripts/
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / "scripts").is_dir():
            return candidate
    # Repli : un dossier contenant results/
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / "results").is_dir():
            return candidate
    return here


def log(msg: str):
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# -------------------------------------------------------------------------
# GPU
# -------------------------------------------------------------------------
def gpu_busy() -> bool:
    """Sous Windows/WDDM, --query-compute-apps renvoie toujours une entree (bureau/DWM),
    meme GPU a l'ident. On se fie donc a la VRAM utilisee : un vrai workload en occupe des Go."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        mem_mib = max(float(x.strip()) for x in out.stdout.strip().splitlines() if x.strip())
        return mem_mib > GPU_BUSY_MIB
    except Exception:
        return False
    except Exception:
        return False


def wait_for_gpu(max_wait_min: int) -> bool:
    """Attend que la VRAM repasse sous GPU_BUSY_MIB.

    Retourne False si le GPU est encore occupe a la fin du delai : on ABANDONNE
    la session (code retour 1) au lieu de lancer forcement, ce qui finirait en
    OOM a repetition. Le superviseur interprete ce code comme une interruption
    et reessaie plus tard, donc la campagne finit par partir des que l'autre
    workload libere la carte.
    """
    if max_wait_min <= 0:
        return True
    deadline = time.time() + max_wait_min * 60
    polls = 0
    while gpu_busy():
        if time.time() > deadline:
            log(f"[GPU] Toujours occupe apres {max_wait_min} min -> abandon de la session.")
            return False
        polls += 1
        if polls == 1 or polls % 10 == 0:
            reste = int(max(0, deadline - time.time()) // 60)
            log(f"[GPU] Occupe, attente... ({reste} min restantes avant abandon)")
        time.sleep(30)
    time.sleep(5)
    return True


# -------------------------------------------------------------------------
# Reprise
# -------------------------------------------------------------------------
def load_done(csv_path: Path, retry_oom: bool) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (
                    row["machine"],
                    row["weight_quantization"],
                    row["kv_quantization"],
                    str(row["flash_attention"]).strip().lower() == "true",
                    int(row["context_length"]),
                )
            except (KeyError, ValueError, TypeError):
                continue
            if retry_oom and str(row.get("vram_inference_mb", "")).upper() == "OOM":
                continue
            done.add(key)
    return done


def build_grid(phase_cfg: dict):
    """Ordre identique au .bat : contexte blox par bloc, configs dans l'ordre du bloc."""
    grid = []
    for ctx in CTXS:
        for base in phase_cfg["base"]:
            grid.append(base + (ctx,))
    return grid


# -------------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------------
def run_one(phase_cfg: dict, combo, timeout_min: int) -> bool:
    q, k, flash, ctx = combo
    cmd = [sys.executable, phase_cfg["script"],
           "--flash-attention", "on" if flash else "off",
           "--context-len", str(ctx)]
    name = Path(phase_cfg["script"]).name
    if name.startswith("01"):
        cmd += ["--quant", q]
    elif name.startswith("02"):
        cmd += ["--kv-quant", k]
    else:
        cmd += ["--quant", q, "--kv-quant", k]

    log(f"--- {name} | quant={q} kv={k} flash={flash} ctx={ctx} ---")
    try:
        proc = subprocess.run(cmd, timeout=timeout_min * 60)
        if proc.returncode != 0:
            log(f"[ECHEC] code={proc.returncode} | {combo}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log(f"[TIMEOUT] {combo} apres {timeout_min} min — config abandonnee.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Orchestrateur de campagne V2 (100 configs)")
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--gpu-wait", type=int, default=0, help="minutes d'attente GPU libre")
    parser.add_argument("--timeout", type=int, default=90, help="minutes max par config")
    parser.add_argument("--retry-oom", action="store_true", help="relance les configs OOM")
    parser.add_argument("--dry-run", action="store_true", help="affiche la grille sans lancer")
    args = parser.parse_args()

    base = find_base_dir()
    os.chdir(base)
    RESULTS_DIR.mkdir(exist_ok=True)
    log(f"Repertoire projet : {base}")

    phases = ["1", "2", "3"] if args.phase == "all" else [args.phase]

    for p in phases:
        cfg = PHASES[p]
        grid = build_grid(cfg)
        csv_path = RESULTS_DIR / cfg["csv"]
        done = load_done(csv_path, args.retry_oom)
        todo = [c for c in grid if (MY_MACHINE,) + c not in done]

        log("=" * 70)
        log(f"PHASE {p} — {cfg['name']}")
        log(f"  Script : {cfg['script']}")
        log(f"  Total : {len(grid)} | deja faits : {len(grid) - len(todo)} | a faire : {len(todo)}")
        log("=" * 70)

        if args.dry_run:
            for c in todo:
                log(f"  [A FAIRE] quant={c[0]} kv={c[1]} flash={c[2]} ctx={c[3]}")
            continue

        for i, combo in enumerate(todo):
            if not wait_for_gpu(args.gpu_wait):
                log("Abandon de la session : la reprise se fera au lancement suivant.")
                sys.exit(CODE_GPU_OCCUPE)
            run_one(cfg, combo, args.timeout)

            if i < len(todo) - 1:
                if combo[3] != todo[i + 1][3]:
                    log(f"[Pause] {PAUSE_INTER_BLOC}s (changement de contexte)")
                    time.sleep(PAUSE_INTER_BLOC)
                else:
                    time.sleep(PAUSE_INTER_CONFIG)

        log(f"[Pause] {PAUSE_FIN_PHASE}s (fin de la phase {p})")
        time.sleep(PAUSE_FIN_PHASE)

    log("Campagne terminee.")


if __name__ == "__main__":
    main()
