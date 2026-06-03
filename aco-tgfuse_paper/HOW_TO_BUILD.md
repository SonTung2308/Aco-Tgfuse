# How to build the paper

## 1. Regenerate architecture figure (required after code changes)

```bash
cd /home/iec/vstung/TGFuse
python gen_arch_fig.py
# → aco-tgfuse_paper/fig_architecture.jpg
```

## 2. Regenerate comparison figure

```bash
# Default stems 05/17/18 → includes PIAFusion (same IR/VIS pair for all methods)
python fig_compare_all.py \
    --stems 05 17 18 \
    --model models_aco_sdmi3/epoch5.model \
    --out aco-tgfuse_paper/fig_compare_all.jpg

# Other stems → PIAFusion excluded automatically (different subset)
python fig_compare_all.py --stems 01 08 20 --model models_aco_sdmi3/epoch5.model
```

## 3. Compile LaTeX

```bash
cd aco-tgfuse_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes
- fig_aco_convergence.jpg is NOT referenced in main.tex (removed per revision).
- SwinFusion on MSRS uses n=55 (official code limitation); footnote added in Table 3.
- PIAFusion quantitative table is excluded; qualitative only on stems 05/17/18.
