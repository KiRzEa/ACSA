#!/usr/bin/env bash
# CAGE ablations, 5 seeds each, fixed fusion, description text, otherwise the protocol of run_cage_5seeds_description.sh
# (its "fixed" runs, outputs/cage_<domain>_fixed, are the matched reference).
#   design      gate_hard   sentiment head reads z * 1[p_acd >= 0.5]
#               gate_none   sentiment head reads z (no ACD -> sentiment interaction)
#               query_id    learned free embedding per category instead of the category text
#   components  lw_fixed    fixed loss weights instead of GradNorm
#               acd_focal   focal loss for the ACD head instead of BCE
#               acd_asl     asymmetric loss for the ACD head instead of BCE
#               child_tuning  Child-Tuning of the encoder
#   heads       heads2 heads4 heads12 heads16 heads24     (cross-attention heads; default 8)
#   adapter     adapter64 adapter96 adapter256 adapter384 adapter512  (adapter bottleneck; default 192)
# Usage: bash run_cage_ablation.sh <name|group|all|everything> [domain ...]
#   all = design; everything = design + components + heads + adapter. Default domains: restaurant hotel.
#   Each ablation x domain = 5 trainings.  DRY=1 prints the commands.  Resumable.
#   Outputs: outputs/cage_abl_<name>_<domain>_fixed/
ABL="${1:-all}"; shift || true
DOMS=("$@"); [ ${#DOMS[@]} -eq 0 ] && DOMS=(restaurant hotel)
DESIGN="gate_hard gate_none query_id"; COMPONENTS="lw_fixed acd_focal acd_asl child_tuning"
HEADS="heads2 heads4 heads12 heads16 heads24"; ADAPTER="adapter64 adapter96 adapter256 adapter384 adapter512"
case "$ABL" in
  all|design) list="$DESIGN" ;;
  components) list="$COMPONENTS" ;;
  heads) list="$HEADS" ;;
  adapter) list="$ADAPTER" ;;
  everything) list="$DESIGN $COMPONENTS $HEADS $ADAPTER" ;;
  *) list="$ABL" ;;
esac
extra_args() {
  case "$1" in
    gate_hard) echo "--gate_mode hard" ;;      gate_none) echo "--gate_mode none" ;;
    query_id) echo "--category_query id" ;;    lw_fixed) echo "--loss_weighting fixed" ;;
    acd_focal) echo "--acd_loss_fn focal" ;;   acd_asl) echo "--acd_loss_fn asl" ;;
    child_tuning) echo "--child_tuning" ;;
    heads*) echo "--num_attention_heads ${1#heads}" ;;
    adapter*) echo "--adapter_dim ${1#adapter}" ;;
    *) echo "" ;;
  esac
}
for a in $list; do
  x="$(extra_args "$a")"; [ -z "$x" ] && { echo "unknown ablation: $a"; exit 1; }
  export EXTRA_ARGS="$x" OUT_PREFIX="cage_abl_${a}" CATEGORY_TEXT=description VARIANTS=fixed
  bash -c 'source "$0/scripts/cage_common.sh"' "$(cd "$(dirname "$0")" && pwd)" "${DOMS[@]}"
done
