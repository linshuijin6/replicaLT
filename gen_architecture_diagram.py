#!/usr/bin/env python3
"""
Generate draw.io XML – v2 (crossing-free layout)
BiomedCLIP-CoCoOp Alignment Framework
with MRI-conditioned Context and Plasma Supervision

Key layout change vs v1:
  - MRI branch on LEFT   → T_ctx drops STRAIGHT DOWN to ⊕ nodes
  - TAU branch to RIGHT of MRI  → img_emb drops cleanly to loss
  - Batch outputs positioned above their downstream consumers
  - Curved edges for cross-layer connections
"""

import html as _html, pathlib

def esc(t):
    return _html.escape(str(t))

# ═══════════════════════════════════════════════════════════
#  Styles
# ═══════════════════════════════════════════════════════════

_base = "rounded=1;whiteSpace=wrap;html=1;arcSize=12;fontFamily=Helvetica;"

S_DATA = _base + "fillColor=#E8EDF2;strokeColor=#9EAFC0;strokeWidth=1.5;fontSize=11;"
S_TAU  = _base + "fillColor=#DCEEE6;strokeColor=#8CC0A0;strokeWidth=1.5;fontSize=11;"
S_MRI  = _base + "fillColor=#D6EDF5;strokeColor=#7BBDD4;strokeWidth=1.5;fontSize=11;"
S_CLS  = _base + "fillColor=#F5EDD6;strokeColor=#D4C48A;strokeWidth=1.5;fontSize=11;"
S_PLS  = _base + "fillColor=#E8DDF5;strokeColor=#B09AD4;strokeWidth=1.5;fontSize=11;"
S_LOSS = _base + "fillColor=#F5DDE0;strokeColor=#D49A9E;strokeWidth=1.5;fontSize=11;"

S_TAU_E = _base + "fillColor=#DCEEE6;strokeColor=#4A8E65;strokeWidth=2.5;fontSize=12;fontStyle=1;"
S_MRI_E = _base + "fillColor=#D6EDF5;strokeColor=#2A7D98;strokeWidth=2.5;fontSize=12;fontStyle=1;"
S_CLS_E = _base + "fillColor=#F5EDD6;strokeColor=#A4944A;strokeWidth=2.5;fontSize=12;fontStyle=1;"
S_PLS_E = _base + "fillColor=#E8DDF5;strokeColor=#8060A4;strokeWidth=2.5;fontSize=12;fontStyle=1;"

_cb = ("rounded=1;whiteSpace=wrap;html=1;strokeWidth=1;dashed=1;dashPattern=6 3;"
       "fontFamily=Helvetica;fontStyle=1;verticalAlign=top;align=left;"
       "spacingLeft=10;spacingTop=5;fontSize=13;")
S_C_TAU = _cb + "fillColor=#F2FAF5;strokeColor=#8CC0A0;"
S_C_MRI = _cb + "fillColor=#EAF5FA;strokeColor=#7BBDD4;"
S_C_CLS = _cb + "fillColor=#FAF5E8;strokeColor=#D4C48A;"
S_C_PLS = _cb + "fillColor=#F0EAF8;strokeColor=#B09AD4;"
S_C_BAT = _cb + "fillColor=#F5F7FA;strokeColor=#B0BCC8;fontSize=11;fontStyle=0;"

S_PLUS = _base + "fillColor=#FFFFFF;strokeColor=#555555;strokeWidth=1.5;fontSize=14;fontStyle=1;arcSize=50;"
S_CALLOUT = _base + "fillColor=#FFF8DC;strokeColor=#D8C850;strokeWidth=1;fontSize=10;fontStyle=3;arcSize=10;"
S_NOTE = ("text;html=1;align=left;verticalAlign=middle;resizable=0;points=[];autosize=0;"
          "strokeColor=none;fillColor=none;fontSize=9;fontFamily=Helvetica;fontColor=#888888;fontStyle=2;")
S_NOTE_BOX = (_base + "fillColor=#F8F8F0;strokeColor=#CCCCCC;strokeWidth=0.5;fontSize=9;"
              "fontColor=#555555;align=left;spacingLeft=8;spacingRight=8;spacingTop=6;spacingBottom=6;arcSize=8;")
S_TITLE = ("text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=0;"
           "strokeColor=none;fillColor=none;fontSize=15;fontFamily=Helvetica;fontStyle=1;")

# arrows
_ae = "rounded=1;orthogonalLoop=1;html=1;fontFamily=Helvetica;fontSize=9;"
S_A    = "edgeStyle=orthogonalEdgeStyle;" + _ae + "strokeWidth=1.5;endArrow=blockThin;endFill=1;"
S_A_MRI= "edgeStyle=orthogonalEdgeStyle;" + _ae + "strokeWidth=2;strokeColor=#2A7D98;endArrow=blockThin;endFill=1;fontColor=#2A7D98;"
S_A_C  = "curved=1;" + _ae + "strokeWidth=1.5;endArrow=blockThin;endFill=1;"          # curved
S_A_CL = "curved=1;" + _ae + "strokeWidth=1.5;endArrow=blockThin;endFill=1;strokeColor=#A4944A;"  # curved beige
S_A_CP = "curved=1;" + _ae + "strokeWidth=1.5;endArrow=blockThin;endFill=1;strokeColor=#8060A4;"  # curved purple
_de = "rounded=1;orthogonalLoop=1;html=1;fontFamily=Helvetica;fontSize=9;dashed=1;dashPattern=5 3;endArrow=open;endFill=0;"
S_D    = "edgeStyle=orthogonalEdgeStyle;" + _de + "strokeWidth=1;"
S_DG   = "curved=1;" + _de + "strokeWidth=1.2;strokeColor=#4A8E65;"
S_DB   = "curved=1;" + _de + "strokeWidth=1.2;strokeColor=#A4944A;"
S_DP   = "curved=1;" + _de + "strokeWidth=1.2;strokeColor=#8060A4;"

# ═══════════════════════════════════════════════════════════
#  Builder
# ═══════════════════════════════════════════════════════════

class G:
    def __init__(self):
        self.cells = []
        self._id = 2
    def n(self, lbl, x, y, w, h, s):
        i = self._id; self._id += 1
        self.cells.append(('n', i, lbl, x, y, w, h, s)); return i
    def e(self, s, t, st, lbl=""):
        i = self._id; self._id += 1
        self.cells.append(('e', i, s, t, st, lbl)); return i
    def xml(self):
        L = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<mxfile host="app.diagrams.net" agent="Claude-v2">',
             ' <diagram name="Architecture" id="arch">',
             '  <mxGraphModel dx="20" dy="20" grid="1" gridSize="10" guides="1" '
             'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
             'pageWidth="1900" pageHeight="1450" math="0" shadow="0" background="#FFFFFF">',
             '   <root>',
             '    <mxCell id="0"/>',
             '    <mxCell id="1" parent="0"/>']
        for c in self.cells:
            if c[0]=='n':
                _,i,lb,x,y,w,h,s = c
                L.append(f'    <mxCell id="{i}" value="{esc(lb)}" style="{s}" vertex="1" parent="1">')
                L.append(f'     <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>')
                L.append(f'    </mxCell>')
            else:
                _,i,s,t,st,lb = c
                L.append(f'    <mxCell id="{i}" value="{esc(lb)}" style="{st}" edge="1" parent="1" source="{s}" target="{t}">')
                L.append(f'     <mxGeometry relative="1" as="geometry"/>')
                L.append(f'    </mxCell>')
        L += ['   </root>','  </mxGraphModel>',' </diagram>','</mxfile>']
        return '\n'.join(L)

g = G()

# ═══════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════
g.n("BiomedCLIP-CoCoOp Alignment Framework with MRI-conditioned Context and Plasma Supervision",
    50, 8, 1800, 30, S_TITLE)

# ═══════════════════════════════════════════════════════════
#  LAYER 1 — DATA PIPELINE  (y 48–88)
# ═══════════════════════════════════════════════════════════
csv  = g.n("CSV<br/><span style='font-size:9px'>(pairs CSV)</span>",    130, 50, 105, 38, S_DATA)
cch  = g.n("Cache Check<br/><span style='font-size:9px'>(precompute)</span>", 275, 50, 130, 38, S_DATA)
ds   = g.n("TAUPlasma<br/>Dataset",                                     445, 50, 130, 38, S_DATA)
smp  = g.n("Subject<br/>BatchSampler",                                  615, 50, 140, 38, S_DATA)
bat  = g.n("Batch",                                                      795, 50, 80, 38, S_DATA)
g.e(csv,cch,S_A); g.e(cch,ds,S_A); g.e(ds,smp,S_A); g.e(smp,bat,S_A)

# ═══════════════════════════════════════════════════════════
#  BATCH OUTPUTS (y 120–160)  — spread above consumers
# ═══════════════════════════════════════════════════════════
# Container background
g.n("", 65, 110, 1340, 60, S_C_BAT)

b_mri = g.n("mri_tokens<br/><span style='font-size:9px'>(B,N,768)</span>",
            80, 117, 165, 40, S_MRI)
b_tau = g.n("tau_tokens<br/><span style='font-size:9px'>(B,N,768)</span>",
            380, 117, 165, 40, S_TAU)
b_lbl = g.n("label_idx<br/><span style='font-size:9px'>(B,)</span>",
            810, 117, 115, 40, S_CLS)
b_pls = g.n("plasma_vals+mask<br/><span style='font-size:9px'>(B,5)</span>",
            1100, 117, 170, 40, S_PLS)

g.e(bat, b_mri, S_A_C); g.e(bat, b_tau, S_A_C)
g.e(bat, b_lbl, S_A_C); g.e(bat, b_pls, S_A_C)

# ═══════════════════════════════════════════════════════════
#  LAYER 2 — FEATURE EXTRACTION  (y 200–560)
#   MRI on LEFT      TAU on RIGHT of MRI
# ═══════════════════════════════════════════════════════════

# ─── MRI Context Branch (LEFT) ────────────────────────────
g.n("MRI Context Branch", 50, 195, 370, 370, S_C_MRI)

mri_pool = g.n("MRI TokenAttentionPool<br/>"
               "<span style='font-size:9px'>dedicated MRI pooling</span>",
               90, 240, 260, 42, S_MRI)
g_mri    = g.n("g_mri<br/><span style='font-size:9px'>(B, 512)</span>",
               140, 315, 150, 38, S_MRI)
ctx_net  = g.n("ContextNet<br/><span style='font-size:9px'>LN → FC → GELU → FC</span>",
               90, 388, 250, 42, S_MRI)
t_ctx    = g.n("<b>T_ctx</b><br/><span style='font-size:9px'>(B, ctx_len, 768)</span>",
               95, 468, 240, 50, S_MRI_E)

g.e(b_mri, mri_pool, S_A)
g.e(mri_pool, g_mri, S_A)
g.e(g_mri, ctx_net, S_A)
g.e(ctx_net, t_ctx, S_A)

# Callout
g.n("★ MRI-conditioned context", 350, 475, 175, 26, S_CALLOUT)

# ─── TAU Visual Branch (RIGHT of MRI) ─────────────────────
g.n("TAU Visual Branch", 460, 195, 360, 370, S_C_TAU)

tau_pool = g.n("TokenAttentionPool<br/>"
               "<span style='font-size:9px'>learned attn. pooling</span>",
               500, 240, 250, 42, S_TAU)
g_tau    = g.n("g_tau<br/><span style='font-size:9px'>(B, 512)</span>",
               545, 315, 150, 38, S_TAU)
proj_img = g.n("Proj Img (MLP)<br/><span style='font-size:9px'>FC→GELU→Drop→FC</span>",
               500, 388, 230, 42, S_TAU)
img_emb  = g.n("<b>img_emb</b><br/><span style='font-size:9px'>(B, 512)</span>",
               520, 475, 210, 48, S_TAU_E)

g.e(b_tau, tau_pool, S_A)
g.e(tau_pool, g_tau, S_A)
g.e(g_tau, proj_img, S_A)
g.e(proj_img, img_emb, S_A)

# Annotate: TAU only → img_emb
g.n("<i>TAU pooled feature<br/>(→ image emb. only)</i>", 700, 318, 145, 28, S_NOTE)

# ═══════════════════════════════════════════════════════════
#  LAYER 3A — CLASS TEXT BRANCH  (y 600–755)
# ═══════════════════════════════════════════════════════════
g.n("Class Text Branch", 40, 598, 1520, 157, S_C_CLS)

ry = 650   # main row y
rh = 48

cls_base  = g.n("base_ctx_cls<br/><span style='font-size:9px'>(ctx_len,768)<br/>[learnable]</span>",
                55, ry, 130, 52, S_CLS)
cls_plus  = g.n("⊕", 210, ry+8, 48, 36, S_PLUS)
cls_prmpt = g.n("Class Prompt<br/><span style='font-size:9px'>×3 classes<br/>[CLS]+ctx+suf</span>",
                285, ry, 140, 52, S_CLS)
cls_tenc  = g.n("Text Encoder<br/><span style='font-size:9px'>BiomedCLIP<br/>(frozen)</span>",
                450, ry, 150, 52, S_CLS)
cls_feat  = g.n("class_feat<br/><span style='font-size:9px'>(B,3,512)</span>",
                625, ry, 135, 52, S_CLS)
gt_sel    = g.n("GT Select<br/><span style='font-size:9px'>diagnosis-<br/>matched</span>",
                785, ry, 120, 52, S_CLS)
cls_proj  = g.n("Proj Cls<br/><span style='font-size:9px'>MLP + L2</span>",
                930, ry, 115, 52, S_CLS)
cls_emb   = g.n("<b>class_emb</b><br/><span style='font-size:9px'>(B, 512)</span>",
                1075, ry, 155, 52, S_CLS_E)

g.e(cls_base, cls_plus, S_A)
g.e(cls_plus, cls_prmpt, S_A)
g.e(cls_prmpt, cls_tenc, S_A)
g.e(cls_tenc, cls_feat, S_A)
g.e(cls_feat, gt_sel, S_A)
g.e(gt_sel, cls_proj, S_A)
g.e(cls_proj, cls_emb, S_A)

# T_ctx → ⊕ Class  (straight down! MRI cyan)
g.e(t_ctx, cls_plus, S_A_MRI, "T_ctx")

# label_idx → GT Select  (curved, colored)
g.e(b_lbl, gt_sel, S_A_CL, "label_idx")

# Annotation
g.n("<i>diagnosis-matched selection</i>", 775, 710, 155, 18, S_NOTE)

# ═══════════════════════════════════════════════════════════
#  LAYER 3B — PLASMA TEXT BRANCH  (y 790–1000)
# ═══════════════════════════════════════════════════════════
g.n("Plasma Text Branch", 40, 788, 1520, 212, S_C_PLS)

py = 855   # main row y

pls_base  = g.n("base_ctx_pls<br/><span style='font-size:9px'>(ctx_len,768)<br/>[learnable]</span>",
                55, py, 130, 52, S_PLS)
pls_plus  = g.n("⊕", 210, py+8, 48, 36, S_PLUS)
pls_prmpt = g.n("Plasma Prompt<br/><span style='font-size:9px'>×5 biomarkers<br/>[CLS]+ctx+suf</span>",
                285, py, 150, 52, S_PLS)
pls_tenc  = g.n("Text Encoder<br/><span style='font-size:9px'>BiomedCLIP<br/>(frozen)</span>",
                460, py, 150, 52, S_PLS)
pls_feat  = g.n("plasma_feat<br/><span style='font-size:9px'>(B,5,512)</span>",
                635, py, 135, 52, S_PLS)
wt_sum    = g.n("Weighted Sum<br/><span style='font-size:9px'>mask-aware<br/>aggregation</span>",
                795, py, 132, 52, S_PLS)
pls_proj  = g.n("Proj Pls<br/><span style='font-size:9px'>MLP + L2</span>",
                955, py, 115, 52, S_PLS)
pls_emb   = g.n("<b>plasma_emb</b><br/><span style='font-size:9px'>(B, 512)</span>",
                1100, py, 160, 52, S_PLS_E)

g.e(pls_base, pls_plus, S_A)
g.e(pls_plus, pls_prmpt, S_A)
g.e(pls_prmpt, pls_tenc, S_A)
g.e(pls_tenc, pls_feat, S_A)
g.e(pls_feat, wt_sum, S_A)
g.e(wt_sum, pls_proj, S_A)
g.e(pls_proj, pls_emb, S_A)

# T_ctx → ⊕ Plasma  (straight down! MRI cyan)
g.e(t_ctx, pls_plus, S_A_MRI, "T_ctx")

# Sub-chain: Masked Softmax → Weighted Sum
msk_sm = g.n("Masked Softmax<br/><span style='font-size:9px'>vals/τ, mask→−∞</span>",
             985, 805, 148, 36, S_PLS)
g.e(msk_sm, wt_sum, S_A, "weights (B,5)")

# plasma_vals+mask → Masked Softmax  (curved)
g.e(b_pls, msk_sm, S_A_CP, "plasma_vals")

# ═══════════════════════════════════════════════════════════
#  LAYER 4 — LOSS & OPTIMIZATION  (y 1050–1130)
# ═══════════════════════════════════════════════════════════
ly = 1050
lh = 62

l_ic = g.n("L<sub>img↔class</sub><br/><span style='font-size:9px'>per-sample 3-way<br/>CrossEntropy</span>",
           100, ly, 160, lh, S_LOSS)
l_ip = g.n("L<sub>img↔plasma</sub><br/><span style='font-size:9px'>batch bidir.<br/>InfoNCE</span>",
           300, ly, 155, lh, S_LOSS)
l_cp = g.n("L<sub>class↔plasma</sub><br/><span style='font-size:9px'>batch bidir.<br/>InfoNCE</span>",
           495, ly, 165, lh, S_LOSS)
l_reg= g.n("L<sub>reg</sub><br/><span style='font-size:9px'>ReLU(cos−0.8)²</span>",
           700, ly, 130, lh, S_LOSS)
l_tot= g.n("<b>Total Loss</b><br/><span style='font-size:9px'>Σ λ-weighted</span>",
           900, ly, 140, lh, S_LOSS)
l_opt= g.n("Backward<br/>+ AdamW",
           1110, ly, 140, lh, S_LOSS)

g.e(l_ic, l_tot, S_A, "λ=1.0")
g.e(l_ip, l_tot, S_A, "λ=0.5")
g.e(l_cp, l_tot, S_A, "λ=0.1")
g.e(l_reg,l_tot, S_A, "λ=0.01")
g.e(l_tot,l_opt, S_A)

# ─── Dashed: embeddings → losses ──────────────────────────
# img_emb (center ≈ 625) → L_ic, L_ip
g.e(img_emb, l_ic, S_DG, "img")
g.e(img_emb, l_ip, S_DG)

# class_emb (center ≈ 1153) → L_ic, L_cp, L_reg
g.e(cls_emb, l_ic, S_DB, "cls")
g.e(cls_emb, l_cp, S_DB)
g.e(cls_emb, l_reg,S_DB)

# plasma_emb (center ≈ 1180) → L_ip, L_cp, L_reg
g.e(pls_emb, l_ip, S_DP, "pls")
g.e(pls_emb, l_cp, S_DP)
g.e(pls_emb, l_reg,S_DP)

# ═══════════════════════════════════════════════════════════
#  ANNOTATIONS
# ═══════════════════════════════════════════════════════════

# Note box (bottom right)
g.n("<b>Note:</b> The conditional context <b>T_ctx</b> is generated from "
    "<b>MRI</b> features via dedicated MRI TokenAttentionPool + ContextNet, "
    "while <b>TAU</b> tokens are used exclusively to produce <b>img_emb</b>.",
    1050, 1150, 480, 55, S_NOTE_BOX)

# Legend (top right)
g.n("<b>Legend</b><br/>"
    "── solid = data flow &nbsp; - - dash = loss signal<br/>"
    "<span style='color:#7BBDD4'>■</span> MRI "
    "<span style='color:#8CC0A0'>■</span> TAU "
    "<span style='color:#D4C48A'>■</span> Class "
    "<span style='color:#B09AD4'>■</span> Plasma "
    "<span style='color:#D49A9E'>■</span> Loss",
    1420, 48, 310, 68, S_NOTE_BOX)

# ═══════════════════════════════════════════════════════════
#  WRITE
# ═══════════════════════════════════════════════════════════
out = pathlib.Path(__file__).parent / "architecture_diagram.drawio"
out.write_text(g.xml(), encoding="utf-8")
print(f"OK  {out}  ({out.stat().st_size:,} bytes)")
