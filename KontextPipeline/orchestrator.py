"""
orchestrator.py — DAG-based multi-step edit orchestrator.

Replaces the procedural loop in run.py with a Directed Acyclic Graph (DAG).
Each edit is a node that:
  - declares its dependencies (which nodes must execute before it)
  - stores its own state: result image, bbox, SAM2 mask
  - can fail without losing the calculated latents of prior nodes

A linear sequence [A, B, C, D] is the common case but the graph also handles:
  - branching: two inserts from the same base (independent nodes, same dep)
  - removal targeting a specific named prior node
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .loader   import load_kontext_pipeline, load_vlm, load_sam2
from .sketch   import generate_from_sketch, LORA_ID
from .utils    import run_standard, save_grid
from .grounding   import VLMGrounder
from .segmentation import SAM2Segmenter
from .collage  import paste_object, build_removal_reference
from .flow_inject  import run_flow_guided_injection, run_flow_removal
from .kv_inject    import capture_obj_kv, run_kv_guided_insertion


# ── Edit definitions ──────────────────────────────────────────────────────────

EDITS: List[dict] = [
    {"name": "bicycle", "description": "yellow mountain bicycle",         "action": "insert"},
    {"name": "vase",    "description": "black ceramic vase with flowers", "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",              "action": "insert"},
    {"name": "ball",    "description": "yellow rubber ball",              "action": "remove"},
    {"name": "bicycle", "description": "yellow mountain bicycle",         "action": "remove"},
]

BASE_PROMPT = (
    "A empty room with a wooden floor, white walls, "
    "and a window letting in natural light."
)

_SEP = "═" * 60


# ── DAG data structures ───────────────────────────────────────────────────────

@dataclass
class EditNode:
    """
    A single edit step in the graph.

    deps: list of node IDs (name_action strings) that must complete before this one.
    When deps is empty the node runs from the base scene.

    State filled during execution:
      scene_in      — input scene image fed to this node
      result        — output image after this node's edit
      bbox_pixels   — (x1, y1, x2, y2) placement or removal bbox
      sam2_mask     — (H, W) uint8 SAM2 mask; 1 = object, 0 = background
    """
    name:        str
    description: str
    action:      str          # "insert" | "remove"
    deps:        List[str] = field(default_factory=list)

    # Execution state (populated during run)
    scene_in:    Optional[Image.Image]     = field(default=None, repr=False)
    result:      Optional[Image.Image]     = field(default=None, repr=False)
    bbox_pixels: Optional[Tuple[int, int, int, int]] = None
    sam2_mask:   Optional[np.ndarray]     = field(default=None, repr=False)

    @property
    def node_id(self) -> str:
        return f"{self.name}_{self.action}"


class EditGraph:
    """
    Manages the DAG of edit nodes and executes them in topological order.
    """

    def __init__(self, nodes: List[EditNode]):
        self.nodes:  Dict[str, EditNode] = {n.node_id: n for n in nodes}
        self._order: List[str]           = self._topo_sort()

    def _topo_sort(self) -> List[str]:
        """Kahn's algorithm for topological ordering."""
        in_deg = {k: 0 for k in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.deps:
                if dep in in_deg:
                    in_deg[nid] = in_deg.get(nid, 0) + 1
        queue  = [k for k, d in in_deg.items() if d == 0]
        order  = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for nid, node in self.nodes.items():
                if cur in node.deps:
                    in_deg[nid] -= 1
                    if in_deg[nid] == 0:
                        queue.append(nid)
        if len(order) != len(self.nodes):
            raise ValueError("Edit graph has a cycle — check 'deps' fields.")
        return order

    def get_input_scene(self, node: EditNode, base: Image.Image) -> Image.Image:
        """Return the output of the last dependency, or base if no deps."""
        if not node.deps:
            return base
        last_dep = node.deps[-1]
        dep_node = self.nodes.get(last_dep)
        if dep_node is None or dep_node.result is None:
            return base
        return dep_node.result

    def find_insertion_node(self, name: str, before_node_id: str) -> Optional[EditNode]:
        """Find the insert node for an object (used by remove to get stored bbox/mask)."""
        target_id = f"{name}_insert"
        for nid in self._order:
            if nid == before_node_id:
                break
            if nid == target_id:
                return self.nodes[target_id]
        return None

    def run(
        self,
        pipe,
        base: Image.Image,
        sketch_dir: str,
        lora_id: str,
        grounder: Optional[VLMGrounder],
        segmenter: Optional[SAM2Segmenter],
        seed: int,
        num_steps: int,
        lora_guidance: float,
        scene_guidance: float,
        height: int,
        width: int,
        out_dir: str,
        device: str,
        use_flow: bool = True,
        alpha_k: float = 0.85,
        alpha_v: float = 0.50,
    ) -> List[Image.Image]:
        results = [base]

        for i, nid in enumerate(self._order):
            node  = self.nodes[nid]
            scene = self.get_input_scene(node, base)
            node.scene_in = scene

            print(f"\n{'─'*60}")
            print(f"  Node {i+1}/{len(self._order)}  —  [{node.node_id}]")
            print(f"{'─'*60}")

            try:
                if node.action == "insert":
                    result = self._run_insert(
                        node, scene, pipe, sketch_dir, lora_id,
                        grounder, segmenter, seed, num_steps,
                        lora_guidance, scene_guidance, height, width,
                        out_dir, device, use_flow, alpha_k, alpha_v,
                    )
                else:
                    result = self._run_remove(
                        node, scene, pipe, grounder, segmenter,
                        seed, num_steps, scene_guidance, height, width,
                        out_dir, use_flow,
                    )
                node.result = result
                result.save(os.path.join(out_dir, f"result_{nid}.png"))
                print(f"    Saved: result_{nid}.png")

            except Exception as exc:
                print(f"  !! Node {nid} FAILED: {exc}")
                print(f"     Carrying forward input scene unchanged.")
                node.result = scene   # fail-safe: pass through

            results.append(node.result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    # ── Insert ────────────────────────────────────────────────────────────

    def _run_insert(
        self, node, scene, pipe, sketch_dir, lora_id,
        grounder, segmenter, seed, num_steps,
        lora_guidance, scene_guidance, height, width,
        out_dir, device, use_flow, alpha_k, alpha_v,
    ) -> Image.Image:
        # A: Generate object from sketch
        sketch_path = self._find_sketch(sketch_dir, node.name)
        print(f"  [A] Generating '{node.description}' from sketch ...")
        obj_img = generate_from_sketch(
            pipe=pipe, sketch_path=sketch_path, description=node.description,
            seed=seed, num_steps=num_steps, guidance=lora_guidance,
            height=height, width=width, lora_id=lora_id, device=device,
        )
        obj_img.save(os.path.join(out_dir, f"obj_{node.name}.png"))

        # B: Capture K/V from obj_img for appearance injection
        print(f"  [KV] Capturing appearance K/V from obj_img ...")
        kv_store, n_tok = capture_obj_kv(pipe, obj_img, height, width)

        # C: Kontext scene pass with K/V injection (placement via prompt)
        insert_prompt = (
            f"Place a {node.description} naturally on the floor of this room. "
            f"Preserve the room exactly as-is; only add the object."
        )
        print(f"  [KV] Running K/V-guided insertion  α_K={alpha_k}  α_V={alpha_v} ...")
        result = run_kv_guided_insertion(
            pipe=pipe, scene=scene, prompt=insert_prompt,
            obj_name=node.name,
            kv_store=kv_store, n_tok=n_tok,
            seed=seed, num_steps=num_steps,
            guidance=scene_guidance, height=height, width=width,
            alpha_k=alpha_k, alpha_v=alpha_v,
        )
        return result

    # ── Remove ────────────────────────────────────────────────────────────

    def _run_remove(
        self, node, scene, pipe, grounder, segmenter,
        seed, num_steps, scene_guidance, height, width,
        out_dir, use_flow,
    ) -> Image.Image:
        # Find the corresponding insert node for bbox and mask
        insert_node = self.find_insertion_node(node.name, node.node_id)

        if insert_node is not None and insert_node.sam2_mask is not None:
            # Use stored SAM2 mask from insertion (most accurate)
            sam2_mask = insert_node.sam2_mask
            bbox_px   = insert_node.bbox_pixels
            print(f"  [SAM2] Using stored mask from insertion step.")
        elif grounder is not None and segmenter is not None:
            # Re-detect in current scene
            print(f"  [GND] Re-detecting '{node.description}' for removal ...")
            bbox_norm = grounder.predict_bbox(scene, node.description)
            bbox_px   = grounder.to_pixels(bbox_norm, width, height)
            print(f"  [SAM2] Re-segmenting for removal ...")
            sam2_mask = segmenter.segment(scene, bbox_px)
        else:
            # Rect fallback
            bbox_px   = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
            sam2_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = bbox_px
            sam2_mask[y1:y2, x1:x2] = 1

        node.sam2_mask   = sam2_mask
        node.bbox_pixels = bbox_px
        Image.fromarray(sam2_mask * 255).save(
            os.path.join(out_dir, f"mask_{node.node_id}.png")
        )

        # Build removal reference: blur-fill the object region
        print(f"  [COL] Building removal reference ...")
        scene_masked = build_removal_reference(scene, sam2_mask)
        scene_masked.save(os.path.join(out_dir, f"collage_remove_{node.name}.png"))

        # Prompt for removal
        removal_prompt = (
            f"{BASE_PROMPT} "
            f"The {node.description} has been removed. "
            f"Seamless wooden floor and white walls fill the area."
        )

        print(f"  [FLOW] Running removal pass ...")
        result = run_flow_removal(
            pipe=pipe, scene_masked=scene_masked, prompt=removal_prompt,
            seed=seed, num_steps=num_steps, guidance=scene_guidance,
            height=height, width=width,
        )
        return result

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _find_sketch(sketch_dir: str, name: str) -> str:
        for fname in (f"{name}.png", f"sketch_{name}.png", f"{name}.jpg"):
            p = os.path.join(sketch_dir, fname)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            f"Sketch not found for '{name}' in {sketch_dir!r}. "
            f"Expected '{name}.png' or 'sketch_{name}.png'."
        )


# ── Build graph from EDITS list ────────────────────────────────────────────────

def build_graph(edits: List[dict]) -> EditGraph:
    """
    Convert a linear EDITS list into an EditGraph.

    Each edit depends on the previous one (linear chain).
    Removal of an object depends on the corresponding insert AND the previous edit.

    edits format: [{"name": str, "description": str, "action": "insert"|"remove"}]
    """
    nodes: List[EditNode] = []
    prev_id: Optional[str] = None

    for edit in edits:
        name   = edit["name"]
        action = edit.get("action", "insert")
        nid    = f"{name}_{action}"
        deps   = [prev_id] if prev_id else []

        # Explicitly provided deps override the linear default
        if "deps" in edit:
            deps = edit["deps"]

        nodes.append(EditNode(
            name=name,
            description=edit["description"],
            action=action,
            deps=deps,
        ))
        prev_id = nid

    return EditGraph(nodes)


# ── CLI entry point ───────────────────────────────────────────────────────────

import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="V2 Sketch-Based Multi-Step Editing — Graph Orchestrator + Flow Injection."
    )
    p.add_argument("--sketch_dir",    required=True)
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--cache_dir",     default="./models")
    p.add_argument("--out_dir",       default="results/kontext_v2")
    p.add_argument("--config",        default=None,
                   help="JSON file with list of {name, description, action}.")
    p.add_argument("--lora_id",       default=LORA_ID)
    p.add_argument("--lora_guidance", type=float, default=4.0)
    p.add_argument("--guidance",      type=float, default=2.5)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_steps",     type=int,   default=28)
    p.add_argument("--height",        type=int,   default=1024)
    p.add_argument("--width",         type=int,   default=1024)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--vlm_model",     default=None,
                   help="VLM for grounding (e.g. Qwen/Qwen2-VL-2B-Instruct). "
                        "Omit to use fallback centroid placement.")
    p.add_argument("--vlm_device",    default="cpu")
    p.add_argument("--sam2_model",    default="facebook/sam2-hiera-small",
                   help="SAM2 model ID. Omit (or set --no_sam2) for rect-mask fallback.")
    p.add_argument("--no_sam2",       action="store_true",
                   help="Skip SAM2. Use rectangular bounding-box masks.")
    p.add_argument("--no_flow",       action="store_true",
                   help="Skip flow injection. Use standard pipe(collage) fallback.")
    p.add_argument("--alpha_k",       type=float, default=0.85,
                   help="K injection strength (0–1). Higher = more obj appearance in attention routing.")
    p.add_argument("--alpha_v",       type=float, default=0.50,
                   help="V injection strength (0–1). Higher = more obj content; keep below 0.6.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    edits = EDITS
    if args.config:
        with open(args.config) as f:
            edits = json.load(f)

    print(f"\n{_SEP}")
    print(f"  KontextPipeline V2 — Graph Orchestrator + Flow-Guided Injection")
    print(f"{_SEP}")
    print(f"  Objects    : {[e['name'] for e in edits]}")
    print(f"  VLM        : {args.vlm_model or 'none (fallback bbox)'}")
    print(f"  SAM2       : {'disabled' if args.no_sam2 else args.sam2_model}")
    print(f"  Injection  : K/V appearance  α_K={args.alpha_k}  α_V={args.alpha_v}")
    print(f"  Output     : {args.out_dir}")
    print(f"{_SEP}\n")

    # ── Load models ──────────────────────────────────────────────────────
    grounder  = None
    segmenter = None

    if args.vlm_model:
        print("Loading VLM ...")
        vlm_model, vlm_proc = load_vlm(args.vlm_model, cache_dir=args.cache_dir,
                                        device=args.vlm_device)
        grounder = VLMGrounder(vlm_model, vlm_proc)

    if not args.no_sam2:
        print("Loading SAM2 ...")
        try:
            sam2_pred = load_sam2(model_id=args.sam2_model, cache_dir=args.cache_dir,
                                   device=args.device)
            segmenter = SAM2Segmenter(sam2_pred)
        except ImportError as e:
            print(f"  SAM2 unavailable ({e}). Using rect-mask fallback.")

    print("\nLoading FLUX.1-Kontext-dev ...")
    pipe = load_kontext_pipeline(
        hf_token=args.hf_token, device=args.device, cache_dir=args.cache_dir,
    )

    # ── Base scene ───────────────────────────────────────────────────────
    print("\n=== Step 0: Base scene ===")
    grey = Image.new("RGB", (args.width, args.height), (200, 200, 190))
    base = run_standard(
        pipe=pipe, canvas=grey, prompt=BASE_PROMPT,
        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
        height=args.height, width=args.width,
    )
    base.save(os.path.join(args.out_dir, "base_scene.png"))
    print(f"  Saved: base_scene.png")

    # ── Build and run graph ───────────────────────────────────────────────
    graph   = build_graph(edits)
    results = graph.run(
        pipe=pipe, base=base,
        sketch_dir=args.sketch_dir, lora_id=args.lora_id,
        grounder=grounder, segmenter=segmenter,
        seed=args.seed, num_steps=args.num_steps,
        lora_guidance=args.lora_guidance,
        scene_guidance=args.guidance,
        height=args.height, width=args.width,
        out_dir=args.out_dir, device=args.device,
        use_flow=not args.no_flow,
        alpha_k=args.alpha_k,
        alpha_v=args.alpha_v,
    )

    # ── Save grid ─────────────────────────────────────────────────────────
    labels = ["base"] + [f"{e['name']} ({e.get('action','insert')})" for e in edits]
    save_grid(results, labels,
              os.path.join(args.out_dir, "chain_grid.png"),
              ncols=len(results))
    print(f"\n{_SEP}")
    print(f"  Chain complete. Grid: {args.out_dir}/chain_grid.png")
    print(f"{_SEP}")


if __name__ == "__main__":
    main()
