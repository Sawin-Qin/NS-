"""
batch_run.py — FreeFEM++ 批量参数扫描调度器
用法: python run_batch.py
"""

import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import qmc
import os
import time
import csv

# ===================== 配置 =====================
N_CASES = 500
FREEFEM_EXE = r"FreeFem++"          # 如果在 PATH 里就直接写，否则写完整路径如 r"D:\FreeFEM\FreeFem++.exe"
EDP_FILE = r"fluid_fem.edp"
WORK_DIR = r"D:\FreeFEM_file\run_100file"
OUTPUT_DIR = os.path.join(WORK_DIR, "result")   # ← 注意：你脚本里写的是 result/ 不是 results/
TIMEOUT = 120  # 秒/case

# 参数范围
PARAM_BOUNDS = {
    "nu":    (0.01, 0.5),
    "rho":   (0.5,  50.0),
    "inlet": (0.1,  50.0),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== LHS 采样 =====================
def generate_lhs(n, bounds):
    """
    n: 样本数
    bounds: dict of (min, max)
    返回: DataFrame, columns = 参数名
    """
    dim = len(bounds)
    sampler = qmc.LatinHypercube(d=dim, seed=42)
    samples = sampler.random(n=n)

    param_names = list(bounds.keys())
    data = {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[name]
        data[name] = lo + samples[:, i] * (hi - lo)

    return pd.DataFrame(data)

'''
def write_csv_file(file_name,nu,rho,inlet):


    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nu", "rho", "inlet"])  # 表头
        writer.writerow([nu, rho, inlet])  # 数据行
'''
def write_csv_file(file_name, nu, rho, inlet):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nu", "rho", "inlet"])
        writer.writerow([nu, rho, inlet])

    print(file_name)
    return



# ===================== 单 case 执行 =====================
def run_single_case(case_id, nu, rho, inlet):
    """执行单个 FreeFEM case，返回 (case_id, success, message)"""
    output_name = f"case_{case_id:04d}.vtk"
    # 相对路径，正斜杠，cwd=WORK_DIR 下找 result/ 子目录
    output_rel = f"result/{output_name}"

    cmd = [
        FREEFEM_EXE, "-nw", "-nowait",
        EDP_FILE,
        "nu", str(nu),
        "rho", str(rho),
        "inlet", str(inlet),
        "outputname", output_rel,
    ]


    #param_file_name = f"param_{case_id:04d}.csv"
    #param_output_rel = f"result/{param_file_name}"

    param_output_rel = os.path.join(OUTPUT_DIR, f"param_{case_id:04d}.csv")
    write_csv_file(param_output_rel,nu,rho,inlet)


    print(f"[case {case_id:03d}] CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT,
        )

        if result.returncode != 0:
            # 保存错误日志
            err_log = os.path.join(OUTPUT_DIR, f"case_{case_id:04d}_err.log")
            with open(err_log, "w", encoding="utf-8") as f:
                f.write(f"CMD: {' '.join(cmd)}\n\n")
                if result.stdout:
                    f.write(result.stdout)
                f.write("\n--- STDERR ---\n")
                if result.stderr:
                    f.write(result.stderr)
            return (case_id, False, f"returncode={result.returncode}")

        # 检查 VTK 是否生成（检查绝对路径）
        expected_vtk = os.path.join(OUTPUT_DIR, output_name)
        if not os.path.exists(expected_vtk):
            # 也检查相对路径（cwd 下）
            alt = os.path.join(WORK_DIR, "result", output_name)
            if not os.path.exists(alt):
                return (case_id, False, f"VTK not found at {expected_vtk}")
            else:
                # VTK 生成在 cwd/result/ 下，OK
                pass

        return (case_id, True, "OK")

    except subprocess.TimeoutExpired:
        return (case_id, False, "TIMEOUT")
    except Exception as e:
        return (case_id, False, str(e))


# ===================== 主流程 =====================
def main():
    print(f"Generating {N_CASES} LHS samples ...")
    df_params = generate_lhs(N_CASES, PARAM_BOUNDS)

    # 保存参数表
    params_csv = os.path.join(OUTPUT_DIR, "params.csv")
    df_params.to_csv(params_csv, index=False)
    print(f"Params saved to {params_csv}")
    print(f"Work dir: {WORK_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    print(f"Launching {N_CASES} cases ...")
    t0 = time.time()

    # ============ 串行模式（调试用，稳定后改并行） ============
    results = []
    for case_id in range(N_CASES):
        nu = float(df_params.loc[case_id, "nu"])
        rho = float(df_params.loc[case_id, "rho"])
        inlet = float(df_params.loc[case_id, "inlet"])
        r = run_single_case(case_id, nu, rho, inlet)
        results.append(r)
        status = "OK" if r[1] else f"FAIL({r[2]})"
        print(f"  [{case_id+1}/{N_CASES}] case {case_id:3d}: {status}")
        # 前 3 个如果全挂就停，方便调试
        if case_id >= 2 and not r[1]:
            print("  >>> First 3 cases all failed, stopping for debug. Check err logs.")
            print(f"  >>> CMD was: {' '.join([FREEFEM_EXE, '-nw', '-nowait', EDP_FILE, '-nu', str(nu), '-rho', str(rho), '-in_let', str(inlet), '-output_name', f'result/case_{case_id:04d}.vtk'])}")
            break

    # ============ 并行模式（调试通了再开） ============
    # 把上面串行注释掉，取消下面注释
    # with ProcessPoolExecutor() as executor:
    #     futures = {}
    #     for case_id in range(N_CASES):
    #         nu = float(df_params.loc[case_id, "nu"])
    #         rho = float(df_params.loc[case_id, "rho"])
    #         inlet = float(df_params.loc[case_id, "inlet"])
    #         fut = executor.submit(run_single_case, case_id, nu, rho, inlet)
    #         futures[fut] = case_id
    #
    #     done_count = 0
    #     fail_count = 0
    #     for fut in as_completed(futures):
    #         case_id, success, msg = fut.result()
    #         done_count += 1
    #         if not success:
    #             fail_count += 1
    #         status = "OK" if success else f"FAIL({msg})"
    #         print(f"  [{done_count}/{N_CASES}] case {case_id:3d}: {status}")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r[1])
    n_fail = len(results) - n_ok
    print(f"\nDone in {elapsed:.1f}s. {n_ok} succeeded, {n_fail} failed.")
    print(f"Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()