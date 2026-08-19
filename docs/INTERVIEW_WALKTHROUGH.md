# 5-Minute Interview Walkthrough — AI Engineer (Recycling Robotics)

> **Target call**: 30-min screen, you get ~5 min for the demo/walkthrough.
> **The job**: convince them you can ship a production perception stack, not
> just train a model. The 2 paired pain points in the JD are
> **predictive-maintenance perception** and **L1 triage for ROC scaling** —
> the demo should land both.
>
> **The one-line pitch (memorize this)**:
> *"Industrial CV is 4 plumbing problems, not a model problem. I built a
> framework with 4 abstractions — detector, tracker, triage surface,
> drift monitor — and 7 modules on top, one per JD bullet, with 171
> tests passing. The 2 things I want to talk about are predictive
> maintenance and L1 triage."*
>
> **Pace**: 150 words/min. This script is ~750 words = 5:00.
> **If interrupted**: don't apologize. *"Good question — let me answer
> that, then come back to where I was."* Mark the spot mentally.

---

## 0:00–0:30 — Opening (60 words)

> Thanks for the time. I want to use the demo to make one point: **the
> hard part of industrial CV isn't the model, it's the plumbing around
> it.** Drift, triage, ROS 2, encoder health, vacuum-pressure monitoring,
> catching when the conveyor's running slow because the belt's about to
> slip. That's where the real engineering is. So I built a framework
> that treats all of that as first-class, with the model as a swappable
> backend. Let me show you.

**What you're doing**: anchoring the JD's "beyond sorting" bullet
before they ask. They've read your resume; they want to see how you
think, not what you've done.

---

## 0:30–1:30 — The 4 Abstractions (framework story) (150 words)

> Four abstractions. Detector, tracker, triage surface, drift monitor.
> Each one is a Python class with a clear interface and tests. The
> detector is YOLO26 + OpenCV DNN today, but the interface doesn't
> care — swap in Ultralytics, TensorRT, or a custom head and the rest
> of the system doesn't change. The tracker is ByteTrack with stable
> IDs across frames. The triage surface is MCP — a 5-tool server that
> an L1 operator agent can call. The drift monitor runs a KS test on
> per-class confidence, plus count anomalies, plus latency regression.
>
> Why these four? Because they're the four things every industrial
> perception system needs, regardless of what's on the conveyor.
> Plastic, metal, glass, e-waste, parcels, fruit — doesn't matter.

**If asked "why these four and not [X]"**: *"I picked the four that
cover 90% of the failure modes I'd see in production. Specificity
comes from the modules on top."*

**Demo action**: open `src/conveyor_perception/core/` — 4 files,
well-commented. The point is *smallest surface, biggest leverage*.

---

## 1:30–2:30 — The 7 Modules (JD coverage) (150 words)

> Seven modules on top of the framework, one per JD bullet. Let me run
> through them fast:
>
> 1. **Perception** — detector + tracker + the inference loop.
> 2. **Predictive maintenance** — encoder health from pick-pattern
>    deltas, vacuum pressure trends, conveyor-belt speed variance.
> 3. **Multitask** — classifier, anomaly, time-series share a pipeline
>    so a single frame is processed once and routed to all heads.
> 4. **Integration** — real ROS 2 node + mock node for CI. Same
>    interface.
> 5. **Robustness** — 13 augmentations including motion blur, occlusion,
>    domain shift; the test suite runs them all and reports mAP drop.
> 6. **Monitoring** — drift trigger, retrain recommendation, shift
>    report. FastAPI dashboard at `/snapshot`, `/shift-report`, `/alerts`.
> 7. **Optimization** — PyTorch vs ONNX vs TensorRT benchmark script.
>
> **171 tests pass, 1 skipped** (the rclpy one — mock runs everywhere).
> Every module has unit tests, every module has the same interface
> pattern, every module is independently swappable.

**If asked "what about the model itself?"**: *"YOLO26s, 15 epochs on a
2,400-image recycling dataset, mAP50=0.671. 30 epochs on Colab T4
would push it to 0.75+."* (Be honest about not running the full
training locally — that's a maturity signal.)

---

## 2:30–3:30 — Live demo (60 sec) (90 words)

> Let me show you the dashboard. *(open the FastAPI monitor at
> localhost:8000/dashboard, or the Colab notebook at cell 7)*.
>
> Top-left: real-time detections with class + confidence. Top-right:
> the shift report — current mAP, drift indicator, retrain
> recommendation. Bottom: the alert feed. The 5-tool MCP triage
> agent is what the L1 operator at the ROC uses. They get an alert:
> "plastic class dropped 12% confidence in the last hour" — they
> click the tool, the system explains why (likely cause: new supplier
> batch with different surface finish), and the operator confirms
> the diagnosis in 30 seconds instead of pulling an engineer.

**If live demo fails**: *"Let me show the test output instead — 171
tests passing is the actual artifact."* Run `pytest -q` and let the
green scroll by. The tests *are* the demo.

---

## 3:30–4:15 — The 2 paired pain points (the close) (120 words)

> Two things I want to land specifically, because they're in your JD:
>
> **Predictive maintenance as a perception problem.** Right now
> recycling robots fail silently — the encoder skips, the vacuum
> drops, the belt slows. By the time a human notices, you've lost
> an hour of throughput. The `predictive_maintenance/` module reads
> the pick-pattern from the detector's track history and surfaces a
> "Belt slipping in 4 hours" hint before it becomes a shutdown. I
> made it rule-based, not ML — auditable, explainable, what an ROC
> actually needs to trust it.
>
> **L1 triage for ROC scaling.** Your ROCs run 24/7 across 3 shifts.
> The bottleneck isn't the model — it's the L1 operator making the
> call. The 5-tool MCP agent turns a 10-minute debugging session
> into a 30-second confirmation. That's the multiplier on every
> recycling line you operate.

**If asked "why rule-based, not ML?"**: *"Because the operator has to
defend the decision to the customer. 'The KS test flagged a
distribution shift' is harder to fight than 'vacuum pressure dropped
8% over 2 hours, and last time that happened, the seal failed.' ML
is right when the signal is hidden; rules are right when the signal
is obvious and the audit trail matters."*

---

## 4:15–5:00 — Closing + handoff (90 words)

> The repo is public, MIT, on GitHub — `conveyor-perception`. The
> README is the front door, the docs folder is the depth. 171 tests
> pass, all 7 JD bullets are covered, the demo runs in Colab free
> tier in about 20 minutes.
>
> I'm not going to pretend this is a year of production work — it's
> a week of focused build to show how I think. What I *am* claiming
> is that I can ship the system: detector + tracker + drift +
> triage + ROS 2 + dashboard + tests, end-to-end, in a stack I'd
> actually want to maintain.
>
> Open to questions, or to dig into any one module.

**What you're doing**: signaling self-awareness (not overselling)
while making the strength claim crisply. The "I'd want to maintain
it" line is the real closer — that's the engineering judgment they
can't test in an interview.

---

## Anticipated Questions (have answers ready)

| Question | One-line answer |
|---|---|
| Why YOLO26 over YOLOv8? | +1.6 mAP, NMS-free output, current Ultralytics default. |
| Why OpenCV DNN, not just Ultralytics? | Portability. ONNX runs anywhere; Ultralytics is a runtime dep. Have a `UltralyticsDetector` fallback for seg-trained models. |
| Why MCP, not REST? | The operator is increasingly an LLM agent, not a human UI. MCP is the standard surface for that. |
| Why rule-based predictive maintenance? | Auditable. ROC has to defend the call to the customer. |
| Why FastAPI, not Flask? | Async, Pydantic-native, type-safe. Plays well with ROS 2 bridge. |
| What's the hardest part? | Knowing when the model is wrong. Drift detection is the real engineering. |
| How would you scale to 10 lines? | Same code, separate processes, shared drift monitor + retrain queue. |
| What would you ship first at EverestLabs? | Wire the existing RecycleOS perception to the drift monitor. Most leverage for least change. |
| What did you skip? | Multi-camera calibration. Real conveyor hardware. Production-grade MLOps. The hard things that need a real plant. |
| What would you add in 30 days? | Encoder health + vacuum pressure + belt-speed fusion in predictive_maintenance/. Your spec calls it out. |

---

## The 1-line pitch (alt versions, pick one)

1. *"Industrial CV is 4 plumbing problems, not a model problem. I built a framework for the plumbing, with 7 modules for the JD."* ← default
2. *"The bottleneck at a recycling plant isn't the model — it's the L1 operator. I built the system to make them 10x faster."* ← if they lead with the ROC pain
3. *"Your JD asks for 7 things. I shipped 7 modules, 171 tests, one repo, MIT-licensed. Let me show you."* ← if you sense they want brisk

---

## What you want to come across as

- **A systems engineer, not a model trainer.** The 4 abstractions framing makes this.
- **Production-grade discipline.** The 171 tests, the pinned versions, the Colab notebook, the upgrade paths doc.
- **Self-aware about scope.** The "week of focused build" line. Don't oversell.
- **Genuinely curious about the ROC.** One real question near the end: *"What does the L1 operator's day look like right now? What takes the longest?"* It signals you think about humans in the loop.
- **Easy to work with.** Short sentences. Don't interrupt. When they ask a question, answer *that* question first.

---

## Pre-call checklist (5 min before)

- [ ] Repo URL in chat: `github.com/roniejosephv-star/conveyor-perception`
- [ ] `pytest -q` runs clean (171 + 1 skipped)
- [ ] Colab notebook cached or local demo ready
- [ ] Dashboard running on `localhost:8000` or screenshot ready
- [ ] This script on second monitor
- [ ] The 10 anticipated-question answers rehearsed once
