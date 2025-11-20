"""Visualise flagged annotations on mock pose skeletons.
Usage: python scripts/visualize/visualize_flags.py --annotations data/qa_output/flagged.json
"""
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SKELETON = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,11),(11,12),(12,13),(1,8),(8,9),(9,10)]

def draw_skeleton(ax, kps, color="steelblue", alpha=1.0):
    kps = np.array(kps).reshape(-1, 3)
    for i, j in SKELETON:
        if kps[i,2] > 0.3 and kps[j,2] > 0.3:
            ax.plot([kps[i,0], kps[j,0]], [kps[i,1], kps[j,1]], color=color, lw=2, alpha=alpha)
    visible = kps[kps[:,2] > 0.3]
    ax.scatter(visible[:,0], visible[:,1], c=color, s=30, zorder=5, alpha=alpha)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--n", type=int, default=4)
    a = p.parse_args()
    data = json.load(open(a.annotations))
    anns = data["annotations"][:a.n]
    fig, axes = plt.subplots(1, len(anns), figsize=(4*len(anns), 4))
    if len(anns) == 1: axes = [axes]
    for ax, ann in zip(axes, anns):
        ax.set_facecolor("#f8f8f8")
        x,y,w,h = ann["bbox"]
        rect = patches.Rectangle((x,y), w, h, lw=1.5, edgecolor="orange", facecolor="none")
        ax.add_patch(rect)
        draw_skeleton(ax, ann["keypoints"], color="tomato")
        ax.set_title(f"id={ann["id"]}", fontsize=9)
        ax.set_xlim(x-20, x+w+20); ax.set_ylim(y+h+20, y-20)
        ax.axis("off")
    plt.suptitle("Flagged Annotations", fontsize=12)
    plt.tight_layout()
    plt.savefig("assets/flagged_preview.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/flagged_preview.png")
    plt.show()
