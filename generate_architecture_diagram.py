#!/usr/bin/env python3
"""
Generate BiomedCLIP-CoCoOp model architecture diagram in draw.io XML format.
Output: model_architecture.drawio
"""

import html as html_mod

# ============================================================================
# Style definitions
# ============================================================================

STYLES = {
    # --- Node styles ---
    "data": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8EDF2;strokeColor=#9EAFC0;"
        "fontSize=11;fontFamily=Helvetica;arcSize=12;"
    ),
    "tau": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#DCEEE6;strokeColor=#8CC0A0;"
        "fontSize=11;fontFamily=Helvetica;arcSize=12;"
    ),
    "mri": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#D6EDF5;strokeColor=#7BBDD4;"
        "fontSize=11;fontFamily=Helvetica;arcSize=12;"
    ),
    "cls": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#F5EDD6;strokeColor=#D4C48A;"
        "fontSize=11;fontFamily=Helvetica;arcSize=12;"
    ),
    "pls": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8DDF5;strokeColor=#B09AD4;"
        "fontSize=11;fontFamily=Helvetica;arcSize=12;"
    ),
    "loss": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#F5DDE0;strokeColor=#D49A9E;"
        "fontSize=11;fontFamily=Helvetica;arcSize=12;"
    ),
    # --- Emphasized embedding nodes ---
    "emb_tau": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#DCEEE6;strokeColor=#5A9E75;"
        "fontSize=12;fontStyle=1;fontFamily=Helvetica;arcSize=12;strokeWidth=2;"
    ),
    "emb_mri": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#D6EDF5;strokeColor=#4A8EAD;"
        "fontSize=12;fontStyle=1;fontFamily=Helvetica;arcSize=12;strokeWidth=2;"
    ),
    "emb_cls": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#F5EDD6;strokeColor=#B8A45A;"
        "fontSize=12;fontStyle=1;fontFamily=Helvetica;arcSize=12;strokeWidth=2;"
    ),
    "emb_pls": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8DDF5;strokeColor=#8A6CB0;"
        "fontSize=12;fontStyle=1;fontFamily=Helvetica;arcSize=12;strokeWidth=2;"
    ),
    # --- Container styles ---
    "cont_batch": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F7FA;strokeColor=#BCC5D0;"
        "dashed=1;dashPattern=6 3;fontSize=12;fontStyle=1;fontFamily=Helvetica;"
        "verticalAlign=top;align=left;spacingLeft=8;spacingTop=2;"
    ),
    "cont_tau": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#EFF7F2;strokeColor=#8CC0A0;"
        "dashed=1;dashPattern=6 3;fontSize=13;fontStyle=1;fontFamily=Helvetica;"
        "verticalAlign=top;align=left;spacingLeft=8;spacingTop=2;"
    ),
    "cont_mri": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#EBF5F9;strokeColor=#7BBDD4;"
        "dashed=1;dashPattern=6 3;fontSize=13;fontStyle=1;fontFamily=Helvetica;"
        "verticalAlign=top;align=left;spacingLeft=8;spacingTop=2;"
    ),
    "cont_cls": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#FAF3E0;strokeColor=#D4C48A;"
        "dashed=1;dashPattern=6 3;fontSize=13;fontStyle=1;fontFamily=Helvetica;"
        "verticalAlign=top;align=left;spacingLeft=8;spacingTop=2;"
    ),
    "cont_pls": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#F0E8F8;strokeColor=#B09AD4;"
        "dashed=1;dashPattern=6 3;fontSize=13;fontStyle=1;fontFamily=Helvetica;"
        "verticalAlign=top;align=left;spacingLeft=8;spacingTop=2;"
    ),
    # --- Annotation styles ---
    "ann": (
        "text;html=1;fontSize=9;fontFamily=Helvetica;fontColor=#777777;"
        "fontStyle=2;align=center;verticalAlign=middle;"
    ),
    "ann_bold": (
        "text;html=1;fontSize=10;fontFamily=Helvetica;fontColor=#4A8EAD;"
        "fontStyle=3;align=center;verticalAlign=middle;"
    ),
    "ann_lambda": (
        "text;html=1;fontSize=9;fontFamily=Helvetica;fontColor=#D49A9E;"
        "fontStyle=1;align=center;verticalAlign=middle;"
    ),
    "note": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8F8F0;strokeColor=#CCCCCC;"
        "fontSize=9;fontFamily=Helvetica;fontColor=#555555;align=left;"
        "spacingLeft=8;spacingRight=8;spacingTop=4;spacingBottom=4;arcSize=8;"
    ),
    "title": (
        "text;html=1;fontSize=16;fontFamily=Helvetica;fontStyle=1;"
        "align=center;verticalAlign=middle;fontColor=#333333;"
    ),
    # --- Edge styles ---
    "edge_solid": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.5;endArrow=block;endFill=1;strokeColor=#666666;"
    ),
    "edge_solid_down": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.5;endArrow=block;endFill=1;strokeColor=#666666;"
        "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
    ),
    "edge_solid_mri": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.8;endArrow=block;endFill=1;strokeColor=#4A8EAD;"
    ),
    "edge_solid_beige": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.2;endArrow=block;endFill=1;strokeColor=#D4C48A;"
    ),
    "edge_solid_purple": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.2;endArrow=block;endFill=1;strokeColor=#B09AD4;"
    ),
    "edge_dashed_green": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.2;dashed=1;dashPattern=6 3;endArrow=open;endFill=0;"
        "strokeColor=#5A9E75;"
    ),
    "edge_dashed_beige": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.2;dashed=1;dashPattern=6 3;endArrow=open;endFill=0;"
        "strokeColor=#B8A45A;"
    ),
    "edge_dashed_purple": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.2;dashed=1;dashPattern=6 3;endArrow=open;endFill=0;"
        "strokeColor=#8A6CB0;"
    ),
    "edge_loss": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;"
        "strokeWidth=1.2;endArrow=block;endFill=1;strokeColor=#D49A9E;"
    ),
}


def esc(text):
    """Escape HTML entities for XML attribute values."""
    return html_mod.escape(text, quote=True)


def node(nid, label, style_key, x, y, w, h):
    """Generate a vertex mxCell XML string."""
    s = STYLES[style_key]
    return (
        f'        <mxCell id="{nid}" value="{esc(label)}" '
        f'style="{s}" vertex="1" parent="1">\n'
        f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>\n'
        f'        </mxCell>\n'
    )


def edge(eid, src, tgt, style_key, label=None):
    """Generate an edge mxCell XML string."""
    s = STYLES[style_key]
    lbl = f' value="{esc(label)}"' if label else ""
    return (
        f'        <mxCell id="{eid}"{lbl} '
        f'style="{s}" edge="1" source="{src}" target="{tgt}" parent="1">\n'
        f'          <mxGeometry relative="1" as="geometry"/>\n'
        f'        </mxCell>\n'
    )


def build_xml():
    cells = []

    # ------------------------------------------------------------------
    # TITLE
    # ------------------------------------------------------------------
    cells.append(node(
        "title", "BiomedCLIP-CoCoOp Alignment Framework with MRI-conditioned Context and Plasma Supervision",
        "title", 300, 8, 1800, 35,
    ))

    # ==================================================================
    # LAYER 1: DATA PIPELINE  (y: 55 - 240)
    # ==================================================================
    # Pipeline boxes
    cells.append(node("d_csv",     "CSV<br/>(pairs_180d)",      "data", 200, 62, 125, 44))
    cells.append(node("d_cache",   "Cache Check<br/>(precompute)", "data", 375, 62, 150, 44))
    cells.append(node("d_dataset", "TAUPlasma<br/>Dataset",     "data", 575, 62, 150, 44))
    cells.append(node("d_sampler", "Subject<br/>BatchSampler",  "data", 775, 62, 160, 44))
    cells.append(node("d_batch",   "Batch",                     "data", 985, 62, 100, 44))

    # Pipeline arrows
    cells.append(edge("e_d1", "d_csv",     "d_cache",   "edge_solid"))
    cells.append(edge("e_d2", "d_cache",   "d_dataset", "edge_solid"))
    cells.append(edge("e_d3", "d_dataset", "d_sampler", "edge_solid"))
    cells.append(edge("e_d4", "d_sampler", "d_batch",   "edge_solid"))

    # Batch outputs container
    cells.append(node("d_bout_bg", "Batch Outputs", "cont_batch", 250, 148, 930, 78))

    # Batch output items
    cells.append(node("b_tau",    "tau_tokens<br/>(B, N, 768)",     "tau", 275, 168, 180, 44))
    cells.append(node("b_mri",    "mri_tokens<br/>(B, N, 768)",     "mri", 480, 168, 180, 44))
    cells.append(node("b_label",  "label_idx<br/>(B,)",             "cls", 685, 168, 135, 44))
    cells.append(node("b_plasma", "plasma_vals + mask<br/>(B, 5)",  "pls", 845, 168, 190, 44))

    # Batch → outputs
    cells.append(edge("e_d5", "d_batch", "d_bout_bg", "edge_solid_down"))

    # ==================================================================
    # LAYER 2: FEATURE EXTRACTION  (y: 270 - 640)
    # ==================================================================

    # --- TAU Visual Branch container ---
    cells.append(node("tau_bg", "TAU Visual Branch", "cont_tau", 100, 275, 490, 370))

    cells.append(node("t_pool",  "TokenAttentionPool",         "tau", 225, 325, 220, 42))
    cells.append(node("t_gtau",  "g_tau (B, 512)",             "tau", 265, 400, 155, 38))
    cells.append(node("t_proj",  "Proj Img (MLP)",             "tau", 275, 470, 135, 38))
    cells.append(node("t_l2",   "L2 Norm",                    "tau", 300, 540, 95, 30))
    cells.append(node("t_imgemb","img_emb (B, 512)",           "emb_tau", 255, 600, 175, 44))

    # TAU annotations
    cells.append(node("ann_tpool", "learned attn. pooling",    "ann", 450, 330, 140, 20))
    cells.append(node("ann_gtau",  "TAU pooled feature",       "ann", 420, 408, 120, 20))

    # TAU arrows
    cells.append(edge("e_t0", "b_tau",   "t_pool",   "edge_solid_down"))
    cells.append(edge("e_t1", "t_pool",  "t_gtau",   "edge_solid_down"))
    cells.append(edge("e_t2", "t_gtau",  "t_proj",   "edge_solid_down"))
    cells.append(edge("e_t3", "t_proj",  "t_l2",     "edge_solid_down"))
    cells.append(edge("e_t4", "t_l2",    "t_imgemb", "edge_solid_down"))

    # --- MRI Context Branch container ---
    cells.append(node("mri_bg", "MRI Context Branch", "cont_mri", 670, 275, 520, 370))

    cells.append(node("m_pool",   "MRI TokenAttentionPool",    "mri", 780, 325, 240, 42))
    cells.append(node("m_gmri",   "g_mri (B, 512)",            "mri", 830, 400, 150, 38))
    cells.append(node("m_ctxnet", "ContextNet",                "mri", 795, 470, 210, 42))
    cells.append(node("m_tctx",   "T_ctx<br/>(B, ctx_len, 768)", "emb_mri", 795, 555, 220, 52))

    # MRI annotations
    cells.append(node("ann_mpool",  "dedicated MRI pooling",   "ann", 1025, 330, 150, 20))
    cells.append(node("ann_ctxnet", "LN → FC → GELU → FC",    "ann", 1010, 478, 140, 20))
    cells.append(node("ann_tctx",   "★ MRI-conditioned context", "ann_bold", 1020, 568, 180, 20))

    # MRI arrows
    cells.append(edge("e_m0", "b_mri",    "m_pool",   "edge_solid_down"))
    cells.append(edge("e_m1", "m_pool",   "m_gmri",   "edge_solid_down"))
    cells.append(edge("e_m2", "m_gmri",   "m_ctxnet", "edge_solid_down"))
    cells.append(edge("e_m3", "m_ctxnet", "m_tctx",   "edge_solid_down"))

    # Context source callout
    cells.append(node("ann_src", "Context source: MRI", "note", 1020, 600, 150, 28))

    # ==================================================================
    # LAYER 3A: CLASS TEXT BRANCH  (y: 690 - 870)
    # ==================================================================
    cells.append(node("cls_bg", "Class Text Branch", "cont_cls", 100, 690, 1600, 170))

    CY = 745  # center y for class flow
    cells.append(node("c_basectx", "base_ctx_cls<br/>(ctx_len, 768)<br/>[learnable]",
                       "cls", 125, CY, 150, 52))
    cells.append(node("c_plus",    "⊕ T_ctx",           "mri", 305, CY+5, 80, 38))
    cells.append(node("c_prompt",  "Class Prompt<br/>(3 prompts)<br/>[CLS]+ctx+text",
                       "cls", 415, CY, 155, 52))
    cells.append(node("c_txtenc",  "Text Encoder<br/>(BiomedCLIP, frozen)",
                       "cls", 600, CY, 170, 52))
    cells.append(node("c_feat",    "class_feat<br/>(B, 3, 512)",
                       "cls", 805, CY+3, 145, 46))
    cells.append(node("c_gtsel",   "GT Select",         "cls", 985, CY+5, 120, 38))
    cells.append(node("c_proj",    "Proj Cls<br/>(MLP)", "cls", 1140, CY+3, 115, 46))
    cells.append(node("c_l2",      "L2 Norm",           "cls", 1290, CY+8, 80, 34))
    cells.append(node("c_emb",     "class_emb<br/>(B, 512)", "emb_cls", 1400, CY, 170, 50))

    # Class annotations
    cells.append(node("ann_gtsel", "diagnosis-matched<br/>selection",
                       "ann", 975, CY+48, 130, 28))

    # Class arrows (horizontal flow)
    cells.append(edge("e_c1", "c_basectx", "c_plus",   "edge_solid"))
    cells.append(edge("e_c2", "c_plus",    "c_prompt",  "edge_solid"))
    cells.append(edge("e_c3", "c_prompt",  "c_txtenc",  "edge_solid"))
    cells.append(edge("e_c4", "c_txtenc",  "c_feat",    "edge_solid"))
    cells.append(edge("e_c5", "c_feat",    "c_gtsel",   "edge_solid"))
    cells.append(edge("e_c6", "c_gtsel",   "c_proj",    "edge_solid"))
    cells.append(edge("e_c7", "c_proj",    "c_l2",      "edge_solid"))
    cells.append(edge("e_c8", "c_l2",      "c_emb",     "edge_solid"))

    # T_ctx → ⊕ (Class branch)  -- key cross-layer connection
    cells.append(edge("e_tctx_c", "m_tctx", "c_plus", "edge_solid_mri"))

    # label_idx → GT Select
    cells.append(edge("e_label_gt", "b_label", "c_gtsel", "edge_solid_beige"))

    # ==================================================================
    # LAYER 3B: PLASMA TEXT BRANCH  (y: 900 - 1140)
    # ==================================================================
    cells.append(node("pls_bg", "Plasma Text Branch", "cont_pls", 100, 900, 1600, 240))

    PY = 955  # center y for plasma flow
    cells.append(node("p_basectx", "base_ctx_pls<br/>(ctx_len, 768)<br/>[learnable]",
                       "pls", 125, PY, 150, 52))
    cells.append(node("p_plus",    "⊕ T_ctx",           "mri", 305, PY+5, 80, 38))
    cells.append(node("p_prompt",  "Plasma Prompt<br/>(5 prompts)<br/>[CLS]+ctx+text",
                       "pls", 415, PY, 160, 52))
    cells.append(node("p_txtenc",  "Text Encoder<br/>(BiomedCLIP, frozen)",
                       "pls", 605, PY, 170, 52))
    cells.append(node("p_feat",    "plasma_feat<br/>(B, 5, 512)",
                       "pls", 810, PY+3, 145, 46))
    cells.append(node("p_wsum",    "Weighted<br/>Sum",   "pls", 995, PY+3, 115, 46))
    cells.append(node("p_proj",    "Proj Pls<br/>(MLP)", "pls", 1150, PY+3, 115, 46))
    cells.append(node("p_l2",      "L2 Norm",           "pls", 1300, PY+8, 80, 34))
    cells.append(node("p_emb",     "plasma_emb<br/>(B, 512)", "emb_pls", 1410, PY, 170, 50))

    # Plasma sub-chain: Masked Softmax → weights → Weighted Sum
    cells.append(node("p_msoftmax", "Masked Softmax",      "pls", 985, PY+75, 140, 36))
    cells.append(node("p_weights",  "weights (B, 5)",      "pls", 1005, PY+130, 105, 30))

    # Plasma annotations
    cells.append(node("ann_wsum", "mask-aware<br/>aggregation",
                       "ann", 985, PY+52, 110, 28))

    # Plasma arrows (horizontal flow)
    cells.append(edge("e_p1", "p_basectx", "p_plus",   "edge_solid"))
    cells.append(edge("e_p2", "p_plus",    "p_prompt",  "edge_solid"))
    cells.append(edge("e_p3", "p_prompt",  "p_txtenc",  "edge_solid"))
    cells.append(edge("e_p4", "p_txtenc",  "p_feat",    "edge_solid"))
    cells.append(edge("e_p5", "p_feat",    "p_wsum",    "edge_solid"))
    cells.append(edge("e_p6", "p_wsum",    "p_proj",    "edge_solid"))
    cells.append(edge("e_p7", "p_proj",    "p_l2",      "edge_solid"))
    cells.append(edge("e_p8", "p_l2",      "p_emb",     "edge_solid"))

    # Sub-chain arrows
    cells.append(edge("e_pw1", "b_plasma",    "p_msoftmax", "edge_solid_purple"))
    cells.append(edge("e_pw2", "p_msoftmax",  "p_weights",  "edge_solid_down"))
    cells.append(edge("e_pw3", "p_weights",   "p_wsum",     "edge_solid"))

    # T_ctx → ⊕ (Plasma branch)
    cells.append(edge("e_tctx_p", "m_tctx", "p_plus", "edge_solid_mri"))

    # ==================================================================
    # LAYER 4: LOSS & OPTIMIZATION  (y: 1190 - 1360)
    # ==================================================================
    LY = 1205
    cells.append(node("l_ic",  "L_img↔class<br/><font style='font-size:9px'>CrossEntropy<br/>(3-way, per-sample)</font>",
                       "loss", 200, LY, 170, 62))
    cells.append(node("l_ip",  "L_img↔plasma<br/><font style='font-size:9px'>InfoNCE<br/>(bidirectional)</font>",
                       "loss", 410, LY, 170, 62))
    cells.append(node("l_cp",  "L_class↔plasma<br/><font style='font-size:9px'>InfoNCE<br/>(bidirectional)</font>",
                       "loss", 620, LY, 180, 62))
    cells.append(node("l_reg", "L_reg<br/><font style='font-size:9px'>ReLU(cos−0.8)²<br/>collapse prevention</font>",
                       "loss", 840, LY, 160, 62))
    cells.append(node("l_total", "Total Loss<br/><font style='font-size:9px'>Σ weighted</font>",
                       "loss", 1060, LY, 140, 62))
    cells.append(node("l_backward", "Backward<br/>+ AdamW",
                       "loss", 1260, LY, 150, 62))

    # Loss internal arrows with lambda labels
    cells.append(edge("e_l1", "l_ic",    "l_total", "edge_loss"))
    cells.append(edge("e_l2", "l_ip",    "l_total", "edge_loss"))
    cells.append(edge("e_l3", "l_cp",    "l_total", "edge_loss"))
    cells.append(edge("e_l4", "l_reg",   "l_total", "edge_loss"))
    cells.append(edge("e_l5", "l_total", "l_backward", "edge_loss"))

    # Lambda annotations
    cells.append(node("ann_l1", "λ=1.0",  "ann_lambda", 280, LY-15, 50, 16))
    cells.append(node("ann_l2", "λ=0.5",  "ann_lambda", 500, LY-15, 50, 16))
    cells.append(node("ann_l3", "λ=0.1",  "ann_lambda", 720, LY-15, 50, 16))
    cells.append(node("ann_l4", "λ=0.01", "ann_lambda", 920, LY-15, 55, 16))

    # Dashed loss connections from embeddings
    cells.append(edge("e_dl1", "t_imgemb", "l_ic", "edge_dashed_green"))
    cells.append(edge("e_dl2", "t_imgemb", "l_ip", "edge_dashed_green"))
    cells.append(edge("e_dl3", "c_emb",    "l_ic", "edge_dashed_beige"))
    cells.append(edge("e_dl4", "c_emb",    "l_cp", "edge_dashed_beige"))
    cells.append(edge("e_dl5", "p_emb",    "l_ip", "edge_dashed_purple"))
    cells.append(edge("e_dl6", "p_emb",    "l_cp", "edge_dashed_purple"))
    cells.append(edge("e_dl7", "p_emb",    "l_reg", "edge_dashed_purple"))
    cells.append(edge("e_dl8", "c_emb",    "l_reg", "edge_dashed_beige"))

    # ==================================================================
    # NOTE BOX
    # ==================================================================
    cells.append(node(
        "note_box",
        "<b>Note:</b> In the current implementation, the conditional context "
        "<b>T_ctx</b> is generated from <b>MRI</b> features via a dedicated "
        "MRI TokenAttentionPool + ContextNet pathway, while <b>TAU</b> tokens "
        "are used exclusively to produce the image embedding <b>img_emb</b>.",
        "note", 1200, 1310, 480, 68,
    ))

    # ==================================================================
    # LEGEND (top-right)
    # ==================================================================
    LGX, LGY = 1780, 55
    cells.append(node("leg_bg", "", "cont_batch", LGX, LGY, 200, 185))
    cells.append(node("leg_title", "<b>Legend</b>", "ann", LGX+10, LGY+2, 80, 18))
    cells.append(node("leg1", "Data / Pipeline",  "data", LGX+10, LGY+28, 130, 20))
    cells.append(node("leg2", "TAU Visual",        "tau",  LGX+10, LGY+54, 130, 20))
    cells.append(node("leg3", "MRI Context",       "mri",  LGX+10, LGY+80, 130, 20))
    cells.append(node("leg4", "Class Text",        "cls",  LGX+10, LGY+106, 130, 20))
    cells.append(node("leg5", "Plasma Text",       "pls",  LGX+10, LGY+132, 130, 20))
    cells.append(node("leg6", "Loss / Optim",      "loss", LGX+10, LGY+158, 130, 20))

    return cells


def main():
    cells = build_xml()
    cell_xml = "".join(cells)

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2026-03-17T00:00:00.000Z" agent="Claude" version="24.0.0" type="device">
  <diagram name="Model Architecture" id="arch">
    <mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2400" pageHeight="1500" math="0" shadow="0" background="#FFFFFF">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{cell_xml}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

    outpath = "model_architecture.drawio"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[OK] Generated {outpath}")
    print(f"     Page: 2400 x 1500 px")
    print(f"     Nodes: ~{len([c for c in cells if 'vertex' in c])} vertices")
    print(f"     Edges: ~{len([c for c in cells if 'edge' in c])} edges")


if __name__ == "__main__":
    main()
