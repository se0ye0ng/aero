#!/usr/bin/env bash
# Pretrained checkpoints used by the registration and matching diagnostics.
#
# These are third-party weights. This repository does not redistribute them: the script
# fetches each file from its official URL and verifies the SHA-256 recorded from the
# artifact this project actually ran against. Licences are the upstream ones; see
# THIRD_PARTY_NOTICES.md before redistributing any derivative.
set -euo pipefail

DEST="${AERO_EXTERNAL_ROOT:-experiments/external}"

# path-relative-to-DEST | sha256 | url
ASSETS=(
"sam_vit_b_01ec64.pth|ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
"dinov2_vits14_reg4_pretrain.pth|f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb|https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
"loftr_outdoor.ckpt|21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec|https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
"minima_roma_0d3fd22/minima_roma.pth|17f3923bd780e8f1450792706e0643f70c8864ad11bf5fd4f2b54714bac23538|https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth"
"minima_roma_0d3fd22/dinov2_vitl14_pretrain.pth|d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428|https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
"minima_796e772/minima_xoftr.ckpt|551eaa814713b3c60ca380eda022be9b7609e51737302e12370f8943a1e864d7|https://github.com/LSXI7/storage/releases/download/MINIMA/minima_xoftr.ckpt"
)

cat <<MSG
Pretrained diagnostic checkpoints
---------------------------------
Destination : $DEST
Total size  : ~2.3 GB
Not fetched here, because they have no stable public direct-download URL:
  - weights_xoftr_640.ckpt   see https://github.com/OnderT/XoFTR (official release assets)
  - raftstereo-middlebury.pth see https://github.com/princeton-vl/RAFT-Stereo (official download script)
MSG

mkdir -p "$DEST"
for entry in "${ASSETS[@]}"; do
    IFS='|' read -r name want url <<<"$entry"
    target="$DEST/$name"
    mkdir -p "$(dirname "$target")"

    if [ -f "$target" ]; then
        got=$(sha256sum "$target" | cut -d' ' -f1)
        if [ "$got" = "$want" ]; then
            echo "ok (cached): $name"
            continue
        fi
        echo "checksum mismatch, refetching: $name" >&2
    fi

    echo "fetching: $name"
    curl -fL --retry 3 --continue-at - -o "$target.part" "$url"
    got=$(sha256sum "$target.part" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "checksum mismatch for $name" >&2
        echo "  expected $want" >&2
        echo "  actual   $got" >&2
        rm -f "$target.part"
        exit 1
    fi
    mv "$target.part" "$target"
    echo "ok: $name"
done

echo "all pinned checkpoints verified under $DEST"
