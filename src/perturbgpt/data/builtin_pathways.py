"""Small curated gene sets used as an offline fallback for pathway enrichment.

These sets are **approximations** of the MSigDB Hallmark collections, hand
curated from the canonical member genes so that pathway enrichment, retrieval
ground truth, and the test suite work with no network access and no
``gseapy`` installation.  They are *not* the official MSigDB lists: for
publication-quality results run ``scripts/build_pathway_sets.py`` with
``gseapy`` installed and use the generated
``data/pathways/msigdb_hallmark.json`` instead.

Schema matches :func:`perturbgpt.eval.pathway_metrics.load_gene_sets`:
``{pathway_name: [gene_symbol, ...]}``.
"""

from __future__ import annotations

#: Provenance tag written into the gene-set metadata sidecar.
FALLBACK_SOURCE = "perturbgpt-curated-fallback"

HALLMARK_FALLBACK: dict[str, list[str]] = {
    "HALLMARK_ERYTHROID": [
        "ALAS2", "AHSP", "ANK1", "BCL2L1", "BPGM", "CA1", "CD47", "EPB42",
        "EPOR", "FECH", "GATA1", "GATA2", "GLDC", "GYPA", "GYPC", "HBA1",
        "HBA2", "HBB", "HMBS", "KLF1", "LMO2", "NFE2", "OSBP2", "RHAG",
        "RHCE", "SLC4A1", "SNCA", "SOX6", "SPTA1", "SPTB", "TAL1", "TFRC",
        "UROD", "UROS", "XK",
    ],
    "HALLMARK_HYPOXIA": [
        "ADM", "ALDOA", "ANGPTL4", "ANXA2", "BHLHE40", "BNIP3", "BNIP3L",
        "CA9", "DDIT4", "EGLN1", "EGLN3", "ENO1", "EPAS1", "GAPDH", "HIF1A",
        "HK2", "LDHA", "MIF", "MXI1", "NDRG1", "PDK1", "PFKL", "PFKP",
        "PGF", "PGK1", "PGM1", "PLIN2", "SERPINE1", "SLC2A1", "TXNIP",
        "UGP2", "VEGFA",
    ],
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING": [
        "AXIN2", "CCND1", "CD44", "CTNNB1", "DKK1", "FZD1", "FZD7", "LEF1",
        "LGR5", "LRP5", "LRP6", "MMP7", "MYC", "NKD1", "NOTUM", "RNF43",
        "SOX9", "TCF7", "TCF7L1", "TCF7L2", "WNT1", "WNT2", "WNT3", "WNT3A",
        "WNT5A", "WNT7A", "ZNRF3",
    ],
    "HALLMARK_NOTCH_SIGNALING": [
        "ADAM10", "ADAM17", "APH1A", "DLL1", "DLL3", "DLL4", "DTX1", "DTX2",
        "DTX3", "DTX4", "HES1", "HEY1", "HEY2", "HEYL", "JAG1", "JAG2",
        "LFNG", "MAML1", "MFNG", "NCSTN", "NOTCH1", "NOTCH2", "NOTCH3",
        "NOTCH4", "PSEN1", "PSEN2", "RFNG",
    ],
    "HALLMARK_P53_PATHWAY": [
        "AEN", "BAX", "BBC3", "BTG2", "CCNG1", "CDKN1A", "DDB2", "E2F7",
        "FAS", "FDXR", "GADD45A", "GADD45B", "MDM2", "PMAIP1", "PLK3",
        "RRM2B", "SESN1", "SESN2", "TIGAR", "TNFRSF10B", "TP53", "TP53I3",
        "TRIAP1", "XPC", "ZMAT3",
    ],
    "HALLMARK_APOPTOSIS": [
        "APAF1", "BAD", "BAK1", "BAX", "BCL2", "BCL2L1", "BCL2L11", "BID",
        "BIRC2", "BIRC3", "CASP10", "CASP3", "CASP6", "CASP7", "CASP8",
        "CASP9", "CYCS", "DIABLO", "FADD", "FAS", "PMAIP1", "TNF",
        "TNFRSF10A", "TNFRSF10B", "TP53", "XIAP",
    ],
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": [
        "ACTA2", "CDH1", "CDH2", "COL1A1", "COL1A2", "COL3A1", "COL5A2",
        "DCN", "FBN1", "FN1", "LUM", "MMP2", "MMP9", "SNAI1", "SNAI2",
        "SPARC", "TAGLN", "TGFB1", "TGFBR1", "TIMP1", "TWIST1", "TWIST2",
        "VIM", "ZEB1", "ZEB2",
    ],
    "HALLMARK_INTERFERON_ALPHA_RESPONSE": [
        "BST2", "DDX58", "EIF2AK2", "GBP1", "GBP2", "GBP4", "GBP5", "IFI27",
        "IFI35", "IFI44", "IFI44L", "IFI6", "IFIH1", "IFIT1", "IFIT2",
        "IFIT3", "IRF7", "IRF9", "ISG15", "ISG20", "JAK1", "MX1", "MX2",
        "OAS1", "OAS2", "OAS3", "OASL", "RSAD2", "STAT1", "STAT2", "TYK2",
        "USP18", "XAF1",
    ],
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": [
        "B2M", "CD74", "CXCL10", "CXCL11", "CXCL9", "FCGR1A", "GBP1", "GBP4",
        "GBP5", "HLA-A", "HLA-B", "HLA-C", "HLA-DPA1", "HLA-DPB1", "HLA-DRA",
        "IDO1", "IFNG", "IRF1", "JAK2", "NOS2", "PSMB8", "PSMB9", "SOCS1",
        "STAT1", "TAP1", "TAP2",
    ],
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": [
        "ATF3", "BIRC3", "CCL2", "CXCL1", "CXCL2", "CXCL3", "CXCL8", "EGR1",
        "FOS", "FOSB", "ICAM1", "IL1B", "IL6", "JUN", "JUNB", "NFKB1",
        "NFKB2", "NFKBIA", "NFKBIB", "NFKBIE", "RELA", "RELB", "SOD2",
        "TNF", "TNFAIP3", "TNFRSF1A", "TNFRSF1B", "VCAM1", "ZFP36",
    ],
    "HALLMARK_MYC_TARGETS_V1": [
        "CAD", "CCT2", "CCT3", "CCT4", "CCT5", "EIF2S1", "EIF4E", "ENO1",
        "GAPDH", "GART", "HK2", "LDHA", "MAX", "MCM2", "MCM3", "MCM4",
        "MCM5", "MCM6", "MCM7", "MYC", "NAP1L1", "NCL", "NME1", "NPM1",
        "ODC1", "PAICS", "PCNA", "PFKL", "PGK1", "PRDX3", "RPL3", "RPS5",
        "RRM2", "SRM", "TXN", "TYMS",
    ],
}
