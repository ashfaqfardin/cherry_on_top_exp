"""
KontextPipeline V2 — DAG-based sketch editing with Flow-Guided Trajectory Injection.

Public API
----------
  from KontextPipeline.loader       import load_kontext_pipeline, load_vlm, load_sam2
  from KontextPipeline.grounding    import VLMGrounder
  from KontextPipeline.segmentation import SAM2Segmenter
  from KontextPipeline.collage      import paste_object, build_removal_reference
  from KontextPipeline.flow_inject  import run_flow_guided_injection, run_flow_removal
  from KontextPipeline.sketch       import generate_from_sketch
  from KontextPipeline.utils        import run_standard, save_grid
  from KontextPipeline.orchestrator import EditNode, EditGraph, build_graph, EDITS, main
"""

from .loader        import load_kontext_pipeline, load_vlm, load_sam2
from .grounding     import VLMGrounder
from .segmentation  import SAM2Segmenter
from .collage       import paste_object, build_removal_reference
from .flow_inject   import run_flow_guided_injection, run_flow_removal
from .kv_inject     import capture_obj_kv, run_kv_guided_insertion
from .sketch        import generate_from_sketch
from .utils         import run_standard, save_grid
from .orchestrator  import EditNode, EditGraph, build_graph, EDITS

__all__ = [
    # Loaders
    "load_kontext_pipeline", "load_vlm", "load_sam2",
    # VLM grounding
    "VLMGrounder",
    # SAM2 segmentation
    "SAM2Segmenter",
    # Image compositing
    "paste_object", "build_removal_reference",
    # Flow-guided ODE injection
    "run_flow_guided_injection", "run_flow_removal",
    # Sketch → object generation
    "generate_from_sketch",
    # Utilities
    "run_standard", "save_grid",
    # DAG orchestrator
    "EditNode", "EditGraph", "build_graph", "EDITS",
]
