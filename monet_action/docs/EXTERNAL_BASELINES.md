# Optional external baseline (D5b): official GR00T N1.7 on LIBERO-Spatial

The proposal allows "an external GR00T policy as a separate baseline", reported separately from the
matched D0–D5 variants (different backbone, pre-training, data pipeline and evaluator). **Not run in this
repository.**

NVIDIA's instructions live in Isaac-GR00T `examples/LIBERO/README.md`
(pinned commit `7aa0bca50f1097f8b17f5c274c98264b72a1419f`). As of that commit:

* the reported LIBERO-Spatial result is 195/200, from a 20K-step fine-tune with global batch 640 on 8 GPUs;
* the fine-tuned checkpoint is released as `nvidia/GR00T-N1.7-LIBERO` (per-suite sub-folders);
* evaluation uses a client/server rollout script with `--n-action-steps 8` in GR00T's own uv environment.

Use a **separate** environment; do not install GR00T into `~/venvs/monet_action`:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T && cd Isaac-GR00T && git checkout 7aa0bca50f1097f8b17f5c274c98264b72a1419f
```

Then follow `examples/LIBERO/README.md` ("Evaluate checkpoint"), using the `libero_spatial` sub-folder of
`nvidia/GR00T-N1.7-LIBERO`. Check the exact folder names on the Hub before downloading.

**Comparability caveats.** Report these alongside any number:

* GR00T executes 8-step chunks open-loop. Our principal setting executes E=1 or E=2 actions.
* Its environment wrapper, init-state handling and step caps differ from `monet_action.evaluate`.
* It was trained on 50 demos/task via the IPEC-COMMUNITY LeRobot conversion. We use
  `lerobot/libero_spatial_image` with an episode-disjoint 10% hold-out.

A faithful comparison needs GR00T's policy behind our `CALL_ACTION` interface so it runs on the same init
states (0..29) and step caps. That adapter is not part of Phase 1.
