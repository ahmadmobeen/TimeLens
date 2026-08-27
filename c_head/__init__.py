"""NS-P1 method bet C: a boundary-regression head on a FROZEN TimeLens-7B.

C step 1 (this package) builds the *unconditioned* AdaVTG-LLM-style baseline head
the method must beat: a plain MLP that reads the frozen LLM's last-layer hidden
state at the generation position and regresses (center, width). See
``docs/research/projects/NS-P1-short-moment-video-llm/{runbook-C-boundary-head,
C-baseline-adavtg-spec}.md``.
"""
