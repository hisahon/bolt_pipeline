"""
Script de debug/exploração pro Spyder — trabalha só com os dados RAW.

Não usa input() nem chama main(): importa as funções e configs direto de
plot_lfr_days.py, então dá pra rodar célula por célula (Ctrl+Enter) e usar
o Variable Explorer / o debugger do Spyder sem passar pelo CLI toda vez.
Precisa estar na mesma pasta que plot_lfr_days.py.

── Como debugar de verdade (breakpoint + step) ──────────────────────────────
1. Clique à esquerda do número da linha, DENTRO da função que quer
   investigar (ex: dentro de load_day, em plot_lfr_days.py), pra criar um
   breakpoint (bolinha vermelha aparece na margem).
2. Rode a célula em modo debug — botão "Debug cell" (ao lado do botão de
   rodar célula, ícone de inseto) ou botão direito do mouse -> Debug cell.
3. A execução para no breakpoint. A partir daí:
       Ctrl+F10  -> step over  (próxima linha, sem entrar em sub-chamadas)
       Ctrl+F11  -> step into  (entra na função chamada na linha atual)
       Ctrl+F12  -> continue   (roda até o próximo breakpoint)
   e inspecione variáveis no Variable Explorer ou digitando direto no
   console (ele fica no contexto da função pausada).

Alternativa sem clicar em nada: escreva a linha `breakpoint()` dentro da
função que você quer investigar (em plot_lfr_days.py) e rode a célula ou o
arquivo normalmente — o Spyder cai automaticamente no debugger (pdb) ali.
Não esqueça de tirar o breakpoint() depois.
"""

# %% Imports — reaproveita tudo do script principal
from pathlib import Path

from plot_lfr_days import (
    INPUT_DIR,
    OUTPUT_DIR,
    discover_files,
    load_day,
    plot_raw_stack,
)

# %% Escolha manual do dia (evita o input() do CLI)
DEBUG_INDEX = 1  # ajuste pro número do dia que você quer testar

entries = discover_files(Path(INPUT_DIR))
for i, (d, p) in enumerate(entries, 1):
    print(f"  [{i:>3}] {d.date()}   ({p.name})")

date, path = entries[DEBUG_INDEX - 1]
print(f"\nDebugando: {date.date()}  <-  {path.name}")

# %% Carregar o dia — dê uma olhada em `df` no Variable Explorer depois de rodar
df = load_day(path, date)
print(df.head())
print(df.dtypes)
print(f"{len(df):,} linhas")

# %% Gerar a figura raw só pra esse dia
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
debug_out = out_dir / "debug_raw.png"

plot_raw_stack([(date, df)], debug_out)
print(f"salvo -> {debug_out}")